#!/usr/bin/env python3
"""Isaac Sim Franka Panda + ROS2 Bridge Script.

PREREQUISITE: Source system ROS2 BEFORE running (provides rclpy + native libs):
    source /opt/ros/jazzy/setup.bash

Then run with Isaac Sim's bundled Python:
    ~/isaac-sim-standalone-6.0.1-linux-x86_64/python.sh scripts/run_franka_bridge.py

What this does:
    1. Starts Isaac Sim (headless or GUI)
    2. Enables ROS2 Bridge extension
    3. Loads Franka Panda robot
    4. Publishes /joint_states (sensor_msgs/JointState)
    5. Publishes /ee_pose (geometry_msgs/PoseStamped) for end-effector
    6. Subscribes to /joint_trajectory (trajectory_msgs/JointTrajectory) to control robot
    7. Sets up RGB + Depth camera via ROS2 Camera Helper graph nodes

Usage:
    # GUI mode (default) — remember to source ROS2 first!
    source /opt/ros/jazzy/setup.bash
    ~/isaac-sim-standalone-6.0.1-linux-x86_64/python.sh scripts/run_franka_bridge.py

    # Headless mode (no GUI, for servers)
    source /opt/ros/jazzy/setup.bash
    ~/isaac-sim-standalone-6.0.1-linux-x86_64/python.sh scripts/run_franka_bridge.py --headless

    # Test mode (auto-exit after 100 frames)
    source /opt/ros/jazzy/setup.bash
    ~/isaac-sim-standalone-6.0.1-linux-x86_64/python.sh scripts/run_franka_bridge.py --test
"""

from __future__ import annotations

import argparse
import sys
import time

# ---------------------------------------------------------------------------
# Parse args BEFORE importing SimulationApp (Isaac Sim requirement)
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Franka Panda + ROS2 Bridge")
parser.add_argument("--test", default=False, action="store_true",
                    help="Run in test mode (auto-exit after N frames)")
parser.add_argument("--headless", default=False, action="store_true",
                    help="Run headless (no GUI)")
parser.add_argument("--dt", type=float, default=1.0 / 60.0,
                    help="Simulation timestep (default: 1/60)")
args, unknown = parser.parse_known_args()

# ---------------------------------------------------------------------------
# Start Isaac Sim
# ---------------------------------------------------------------------------
from isaacsim import SimulationApp

simulation_app = SimulationApp({
    "renderer": "RealTimePathTracing",
    "headless": args.headless,
})

import carb
import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.stage as stage_utils
import numpy as np
from isaacsim.core.experimental.objects import DistantLight, GroundPlane
from isaacsim.core.simulation_manager import SimulationManager

# Enable ROS2 bridge extension (must be before rclpy import)
app_utils.enable_extension("isaacsim.ros2.bridge")

# Enable Franka robot manipulator extension (must be before Franka import)
app_utils.enable_extension("isaacsim.robot.experimental.manipulators.examples")

simulation_app.update()

# ---------------------------------------------------------------------------
# ROS2 imports (from Isaac Sim's bundled rclpy, NOT system rclpy)
# ---------------------------------------------------------------------------
# Ensure Isaac Sim's internal rclpy is on sys.path as a fallback.
# If you have system ROS2 sourced (recommended), system rclpy takes precedence.
# Run:  source /opt/ros/jazzy/setup.bash && python.sh scripts/run_franka_bridge.py
import os as _os
_rclpy_base = _os.path.join(
    _os.path.expanduser("~"),
    "isaac-sim-standalone-6.0.1-linux-x86_64",
    "exts", "isaacsim.ros2.core", "jazzy", "rclpy",
)
if _rclpy_base not in sys.path:
    sys.path.append(_rclpy_base)  # append, not insert — system rclpy wins if sourced

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

# ---------------------------------------------------------------------------
# Camera setup imports
# ---------------------------------------------------------------------------
import omni
import omni.graph.core as og
import usdrt.Sdf
from pxr import Gf, Sdf, UsdGeom

# ---------------------------------------------------------------------------
# Franka robot class (from Isaac Sim experimental manipulators)
# Note: This import works because Isaac Sim's python.sh sets up sys.path
#       to include all extension packages automatically.
# ---------------------------------------------------------------------------
from isaacsim.robot.experimental.manipulators.examples.franka.franka import Franka


# ===================================================================
# Camera Setup
# ===================================================================
CAMERA_STAGE_PATH = "/World/Camera"
ROS_CAMERA_GRAPH_PATH = "/ROS_Camera"


def setup_camera(width: int = 640, height: int = 480,
                 position: tuple = (0.5, 0.0, 1.2),
                 rgb_topic: str = "rgb",
                 depth_topic: str = "depth",
                 camera_info_topic: str = "camera_info",
                 frame_id: str = "sim_camera") -> None:
    """Create a camera prim and set up ROS2 image publishing via Action Graph.

    Publishes three ROS2 topics:
        - /rgb          (sensor_msgs/Image)
        - /depth        (sensor_msgs/Image)
        - /camera_info  (sensor_msgs/CameraInfo)

    Args:
        width, height: Image resolution.
        position: Camera position in world frame (x, y, z).
        rgb_topic: ROS2 topic name for RGB images.
        depth_topic: ROS2 topic name for depth images.
        camera_info_topic: ROS2 topic name for camera info.
        frame_id: ROS2 frame_id for the camera.
    """
    # Create camera prim
    camera_prim = UsdGeom.Camera(
        omni.usd.get_context().get_stage().DefinePrim(CAMERA_STAGE_PATH, "Camera")
    )
    xform_api = UsdGeom.XformCommonAPI(camera_prim)
    xform_api.SetTranslate(Gf.Vec3d(*position))
    # Point camera forward and slightly downward
    xform_api.SetRotate((90, 0, 0), UsdGeom.XformCommonAPI.RotationOrderXYZ)

    camera_prim.GetHorizontalApertureAttr().Set(21)
    camera_prim.GetVerticalApertureAttr().Set(16)
    camera_prim.GetProjectionAttr().Set("perspective")
    camera_prim.GetFocalLengthAttr().Set(24)
    camera_prim.GetFocusDistanceAttr().Set(400)

    simulation_app.update()

    # Build ROS2 Camera Action Graph
    keys = og.Controller.Keys
    ros_camera_graph, _, _, _ = og.Controller.edit(
        {
            "graph_path": ROS_CAMERA_GRAPH_PATH,
            "evaluator_name": "push",
            "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_ONDEMAND,
        },
        {
            keys.CREATE_NODES: [
                ("OnTick", "omni.graph.action.OnTick"),
                ("createRenderProduct", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
                ("cameraHelperRgb", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                ("cameraHelperInfo", "isaacsim.ros2.bridge.ROS2CameraInfoHelper"),
                ("cameraHelperDepth", "isaacsim.ros2.bridge.ROS2CameraHelper"),
            ],
            keys.CONNECT: [
                ("OnTick.outputs:tick", "createRenderProduct.inputs:execIn"),
                ("createRenderProduct.outputs:execOut", "cameraHelperRgb.inputs:execIn"),
                ("createRenderProduct.outputs:execOut", "cameraHelperInfo.inputs:execIn"),
                ("createRenderProduct.outputs:execOut", "cameraHelperDepth.inputs:execIn"),
                ("createRenderProduct.outputs:renderProductPath",
                 "cameraHelperRgb.inputs:renderProductPath"),
                ("createRenderProduct.outputs:renderProductPath",
                 "cameraHelperInfo.inputs:renderProductPath"),
                ("createRenderProduct.outputs:renderProductPath",
                 "cameraHelperDepth.inputs:renderProductPath"),
            ],
            keys.SET_VALUES: [
                ("createRenderProduct.inputs:cameraPrim",
                 [usdrt.Sdf.Path(CAMERA_STAGE_PATH)]),
                ("createRenderProduct.inputs:width", width),
                ("createRenderProduct.inputs:height", height),
                ("cameraHelperRgb.inputs:frameId", frame_id),
                ("cameraHelperRgb.inputs:topicName", rgb_topic),
                ("cameraHelperRgb.inputs:type", "rgb"),
                ("cameraHelperInfo.inputs:frameId", frame_id),
                ("cameraHelperInfo.inputs:topicName", camera_info_topic),
                ("cameraHelperDepth.inputs:frameId", frame_id),
                ("cameraHelperDepth.inputs:topicName", depth_topic),
                ("cameraHelperDepth.inputs:type", "depth"),
            ],
        },
    )

    # Evaluate once to create publishers
    og.Controller.evaluate_sync(ros_camera_graph)
    simulation_app.update()

    carb.log_info(f"Camera set up: {width}x{height} at {position}")
    carb.log_info(f"  RGB topic: /{rgb_topic}")
    carb.log_info(f"  Depth topic: /{depth_topic}")
    carb.log_info(f"  Camera info topic: /{camera_info_topic}")


# ===================================================================
# Franka + ROS2 Bridge Node
# ===================================================================
class FrankaBridgeNode(Node):
    """ROS2 node that bridges Isaac Sim Franka robot with ROS2 topics.

    Publishers:
        /joint_states   (sensor_msgs/JointState)  — robot joint positions
        /ee_pose        (geometry_msgs/PoseStamped) — end-effector pose

    Subscribers:
        /joint_trajectory (trajectory_msgs/JointTrajectory) — control commands
    """

    # Franka joint names (order matches articulation DOFs)
    JOINT_NAMES = [
        "panda_joint1", "panda_joint2", "panda_joint3",
        "panda_joint4", "panda_joint5", "panda_joint6",
        "panda_joint7", "panda_finger_joint1", "panda_finger_joint2",
    ]

    def __init__(self, robot_path: str = "/World/Franka",
                 joint_state_hz: float = 30.0,
                 ee_pose_hz: float = 30.0) -> None:
        """Initialize Franka + ROS2 bridge.

        Args:
            robot_path: USD path for the Franka robot.
            joint_state_hz: Publishing rate for /joint_states.
            ee_pose_hz: Publishing rate for /ee_pose.
        """
        super().__init__("franka_bridge")

        # --- Create Franka robot ---
        self.get_logger().info("Loading Franka Panda robot...")
        self.franka = Franka(robot_path=robot_path, create_robot=True)
        self.get_logger().info("Franka Panda loaded.")

        # Joint state index map: articulation DOF indices
        self._joint_count = 9  # 7 arm + 2 gripper

        # --- ROS2 Publishers ---
        self.joint_state_pub = self.create_publisher(
            JointState, "/joint_states", 10
        )
        self.ee_pose_pub = self.create_publisher(
            PoseStamped, "/ee_pose", 10
        )

        # --- ROS2 Subscribers ---
        self.trajectory_sub = self.create_subscription(
            JointTrajectory, "/joint_trajectory", self._trajectory_callback, 10
        )

        # --- Timer for periodic publishing ---
        self._js_period = 1.0 / joint_state_hz
        self._ee_period = 1.0 / ee_pose_hz
        self._last_js_time = time.time()
        self._last_ee_time = time.time()

        # --- Latest trajectory command ---
        self._latest_trajectory: JointTrajectory | None = None
        self._traj_point_idx: int = 0
        self._traj_start_time: float = 0.0

        self.get_logger().info("FrankaBridgeNode ready.")
        self.get_logger().info("  Publishers:  /joint_states, /ee_pose")
        self.get_logger().info("  Subscribers: /joint_trajectory")

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    def _trajectory_callback(self, msg: JointTrajectory) -> None:
        """Receive a JointTrajectory command and start executing it."""
        self._latest_trajectory = msg
        self._traj_point_idx = 0
        self._traj_start_time = time.time()
        self.get_logger().info(
            f"Received trajectory: {len(msg.points)} points "
            f"for {len(msg.joint_names)} joints"
        )

    # ------------------------------------------------------------------
    # Periodic publish
    # ------------------------------------------------------------------
    def publish_state(self) -> None:
        """Publish joint states and end-effector pose at configured rates."""
        now = time.time()

        # Publish joint states
        if now - self._last_js_time >= self._js_period:
            self._publish_joint_state(now)
            self._last_js_time = now

        # Publish end-effector pose
        if now - self._last_ee_time >= self._ee_period:
            self._publish_ee_pose(now)
            self._last_ee_time = now

    def _publish_joint_state(self, now: float) -> None:
        """Read current DOF positions and publish as JointState."""
        try:
            dof_positions = self.franka.get_dof_positions().numpy().flatten()
        except Exception:
            return  # Physics may not be ready yet

        msg = JointState()
        msg.header = Header()
        msg.header.stamp.sec = int(now)
        msg.header.stamp.nanosec = int((now % 1) * 1e9)
        msg.header.frame_id = "franka_base"
        msg.name = self.JOINT_NAMES[:len(dof_positions)]
        msg.position = dof_positions.tolist()
        msg.velocity = []
        msg.effort = []

        self.joint_state_pub.publish(msg)

    def _publish_ee_pose(self, now: float) -> None:
        """Read end-effector pose and publish as PoseStamped."""
        try:
            pos, quat = self.franka.end_effector_link.get_world_poses()
            pos_np = pos.numpy().flatten()
            quat_np = quat.numpy().flatten()
        except Exception:
            return

        msg = PoseStamped()
        msg.header = Header()
        msg.header.stamp.sec = int(now)
        msg.header.stamp.nanosec = int((now % 1) * 1e9)
        msg.header.frame_id = "world"
        msg.pose.position.x = float(pos_np[0])
        msg.pose.position.y = float(pos_np[1])
        msg.pose.position.z = float(pos_np[2])
        msg.pose.orientation.w = float(quat_np[0])
        msg.pose.orientation.x = float(quat_np[1])
        msg.pose.orientation.y = float(quat_np[2])
        msg.pose.orientation.z = float(quat_np[3])

        self.ee_pose_pub.publish(msg)

    # ------------------------------------------------------------------
    # Execute trajectory commands
    # ------------------------------------------------------------------
    def execute_trajectory(self, now: float) -> None:
        """Step through the latest trajectory command.

        Each JointTrajectoryPoint is applied as a DOF position target.
        Timing follows the point's time_from_start.
        """
        if self._latest_trajectory is None:
            return
        if self._traj_point_idx >= len(self._latest_trajectory.points):
            return

        elapsed = now - self._traj_start_time
        point = self._latest_trajectory.points[self._traj_point_idx]

        # Check if it's time to advance to next point
        from builtins import float as builtin_float
        target_time = point.time_from_start.sec + point.time_from_start.nanosec * 1e-9
        if elapsed >= target_time:
            # Apply this point's positions as DOF targets
            positions = np.array(point.positions[:self._joint_count], dtype=np.float32)
            self.franka.set_dof_position_targets(
                positions.reshape(1, -1),
                dof_indices=list(range(min(len(point.positions), self._joint_count))),
            )
            self._traj_point_idx += 1


# ===================================================================
# Main simulation loop
# ===================================================================
def main() -> None:
    """Run the Franka + ROS2 bridge simulation."""
    carb.log_info("=" * 50)
    carb.log_info("  VLARM — Franka Panda + ROS2 Bridge")
    carb.log_info("=" * 50)

    # --- World setup ---
    stage_utils.set_stage_units(meters_per_unit=1.0)
    GroundPlane("/World/GroundPlane")
    DistantLight("/World/DistantLight").set_intensities(intensities=[3000])
    simulation_app.update()

    # --- Camera ---
    setup_camera(
        width=640, height=480,
        position=(0.5, 0.0, 1.2),
        rgb_topic="rgb",
        depth_topic="depth",
        camera_info_topic="camera_info",
        frame_id="sim_camera",
    )

    # --- Initialize rclpy (Isaac Sim bundled) ---
    rclpy.init(args=unknown)
    bridge_node = FrankaBridgeNode()

    # --- Setup physics ---
    SimulationManager.setup_simulation(dt=args.dt, device="cuda" if not args.headless else "cpu")

    # --- Run ---
    app_utils.play()
    simulation_app.update()

    frame = 0
    carb.log_info("Simulation running. Press Ctrl+C to stop.")

    try:
        while simulation_app.is_running():
            simulation_app.update()

            if app_utils.is_playing():
                # Spin ROS2 to process callbacks
                rclpy.spin_once(bridge_node, timeout_sec=0.0)

                # Publish state
                bridge_node.publish_state()

                # Execute trajectory commands
                bridge_node.execute_trajectory(time.time())

                frame += 1
                if args.test and frame >= 100:
                    carb.log_info("Test mode: exiting after 100 frames.")
                    break

    except KeyboardInterrupt:
        carb.log_info("Interrupted by user.")

    finally:
        carb.log_info("Shutting down...")
        app_utils.stop()
        bridge_node.destroy_node()
        rclpy.shutdown()
        simulation_app.close()
        carb.log_info("Done.")


if __name__ == "__main__":
    main()
