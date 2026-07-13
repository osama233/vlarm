#!/usr/bin/env python3
"""VLARM Isaac Sim Environment Wrapper.

Provides a Gym-style RL environment interface for the Franka Panda
tabletop manipulation task.

    env = IsaacEnv(headless=False)
    obs = env.reset()
    for _ in range(100):
        action = policy(obs)
        obs, reward, done, info = env.step(action)
        if done:
            obs = env.reset()
    env.close()

Observation space (dict):
    joint_positions   (9,)  float32  — robot DOF positions  [rad]
    joint_velocities  (9,)  float32  — robot DOF velocities [rad/s]
    ee_position       (3,)  float32  — end-effector position [m]
    ee_orientation    (4,)  float32  — end-effector quaternion (w,x,y,z)
    gripper_width     (1,)  float32  — gripper opening [m]
    rgb               (480, 640, 3) uint8 — RGB image (optional; requires ROS2 camera)
    depth             (480, 640, 1) float32 — depth image (optional)

Action space (Box):
    joint_targets     (7,)  float32  — target arm joint positions [rad]
    OR (9,) if gripper included

Reward:
    Dense: negative L2 distance from gripper to nearest cube
           + bonus when cube is grasped
           + bonus when cube is placed in basket

Usage (inside Isaac Sim Python):
    source /opt/ros/jazzy/setup.bash
    ~/isaac-sim-standalone-6.0.1-linux-x86_64/python.sh scripts/03_test_task_scene.py
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Optional

import numpy as np


# ===================================================================
# Environment
# ===================================================================
class IsaacEnv:
    """Gym-style environment wrapping Isaac Sim + Franka + task scene.

    The environment owns:
      - Task scene (table, cubes, basket, cameras) via ``task_scene``
      - Franka Panda robot
      - Physics stepping

    It does NOT own ``SimulationApp`` — the caller must create that first
    and pass it in (or create it before constructing this env).

    Parameters
    ----------
    headless : bool
        Run without GUI.
    dt : float
        Physics timestep in seconds (default 1/60).
    substeps : int
        Physics steps per ``step()`` call (so each RL step = substeps * dt s).
    robot_path : str
        USD prim path for the Franka robot.
    enable_ros2 : bool
        If True, create ROS2 publishers/subscribers for external control.
    """

    # Franka joint names (order matches articulation DOFs)
    JOINT_NAMES = [
        "panda_joint1", "panda_joint2", "panda_joint3",
        "panda_joint4", "panda_joint5", "panda_joint6",
        "panda_joint7", "panda_finger_joint1", "panda_finger_joint2",
    ]

    # Arm-only joints (first 7)
    ARM_DOF = 7
    # Gripper joints (last 2; move symmetrically)
    GRIPPER_DOF = 2
    TOTAL_DOF = 9

    # Joint limits (rad) — Franka Panda nominal
    JOINT_LIMITS_LOW = np.array([
        -2.8973, -1.7628, -2.8973, -3.0718, -2.8973,
        -0.0175, -2.8973, 0.0, 0.0,
    ], dtype=np.float32)
    JOINT_LIMITS_HIGH = np.array([
        2.8973, 1.7628, 2.8973, -0.0698, 2.8973,
        3.7525, 2.8973, 0.04, 0.04,
    ], dtype=np.float32)

    # Default home position
    HOME_POSITION = np.array([
        0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04,
    ], dtype=np.float32)

    def __init__(
        self,
        headless: bool = False,
        dt: float = 1.0 / 60.0,
        substeps: int = 10,
        robot_path: str = "/World/Franka",
        enable_ros2: bool = False,
        simulation_app: Any = None,
    ) -> None:
        self._headless = headless
        self._dt = dt
        self._substeps = substeps
        self._robot_path = robot_path
        self._enable_ros2 = enable_ros2
        self._simulation_app = simulation_app

        # References set up in _setup()
        self._franka: Any = None
        self._scene: dict[str, Any] = {}
        self._ros2_node: Any = None
        self._step_count: int = 0
        self._episode_count: int = 0
        self._max_episode_steps: int = 200

        # Cached observations
        self._last_joint_positions: np.ndarray = np.zeros(self.TOTAL_DOF, dtype=np.float32)
        self._last_joint_velocities: np.ndarray = np.zeros(self.TOTAL_DOF, dtype=np.float32)
        self._last_ee_position: np.ndarray = np.zeros(3, dtype=np.float32)
        self._last_ee_orientation: np.ndarray = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self._last_gripper_width: float = 0.08  # fully open = ~8 cm

        # Cube positions (set during reset)
        self._cube_positions: list[tuple[float, float, float]] = []
        self._cube_positions_np: np.ndarray = np.zeros((0, 3), dtype=np.float32)
        self._target_center: np.ndarray = np.zeros(3, dtype=np.float32)

        # Pre-allocated empty camera images (filled in when ROS2 camera is active)
        self._empty_rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        self._empty_depth = np.zeros((480, 640, 1), dtype=np.float32)

        # Bind simulation_app.update once to avoid repeated attribute lookups
        if simulation_app is not None:
            self._app_update = simulation_app.update
        else:
            self._app_update = None

        self._setup()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def _setup(self) -> None:
        """Initialize scene, robot, and optional ROS2 bridge."""
        import carb
        import sys
        from pathlib import Path

        # Ensure project src/ is importable
        _src = str(Path(__file__).resolve().parent.parent)
        if _src not in sys.path:
            sys.path.insert(0, _src)

        # Lazy imports (only work inside Isaac Sim Python env)
        import isaacsim.core.experimental.utils.app as app_utils
        import isaacsim.core.experimental.utils.stage as stage_utils
        from isaacsim.core.experimental.objects import DistantLight, GroundPlane
        from isaacsim.core.simulation_manager import SimulationManager

        # Extensions (idempotent — enable_extension is a no-op if already on)
        app_utils.enable_extension("isaacsim.ros2.bridge")
        app_utils.enable_extension("isaacsim.robot.experimental.manipulators.examples")

        # Stage units
        stage_utils.set_stage_units(meters_per_unit=1.0)

        # Ground & lighting (skip if already exist)
        try:
            GroundPlane("/World/GroundPlane")
        except Exception:
            pass
        try:
            DistantLight("/World/DistantLight").set_intensities(intensities=[3000])
        except Exception:
            pass

        # Build task scene
        from envs.task_scene import build_task_scene

        self._scene = build_task_scene()
        carb.log_info("[IsaacEnv] Task scene built.")

        # Cache scene parameters from config
        cfg = self._scene["config"]
        self._target_center = np.array(cfg["target_position"], dtype=np.float32)
        self._table_surface_z = cfg["table_position"][2] + cfg["table_size"][2] / 2

        # Load Franka robot
        from isaacsim.robot.experimental.manipulators.examples.franka.franka import Franka

        carb.log_info(f"[IsaacEnv] Loading Franka at {self._robot_path}...")
        self._franka = Franka(robot_path=self._robot_path, create_robot=True)
        carb.log_info("[IsaacEnv] Franka loaded.")

        # Physics
        SimulationManager.setup_simulation(
            dt=self._dt,
            device="cuda" if not self._headless else "cpu",
        )

        # Start the simulation timeline (idempotent if already playing)
        app_utils.play()

        # Optional ROS2 bridge node
        if self._enable_ros2:
            self._setup_ros2()

    def _setup_ros2(self) -> None:
        """Create a ROS2 node for external communication."""
        import carb
        try:
            import rclpy
            from envs.isaac_env import _IsaacEnvROS2Node
            # rclpy.init() should already be called by the caller
            if not rclpy.ok():
                rclpy.init()
            self._ros2_node = _IsaacEnvROS2Node(self)
            carb.log_info("[IsaacEnv] ROS2 node created.")
        except Exception as e:
            carb.log_warn(f"[IsaacEnv] ROS2 setup failed (non-fatal): {e}")

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(
        self,
        seed: Optional[int] = None,
        cube_positions: Optional[list[tuple[float, float, float]]] = None,
    ) -> dict[str, np.ndarray]:
        """Reset the environment for a new episode.

        Actions performed:
        1. Randomise cube positions (or use provided positions).
        2. Reset Franka to home joint configuration.
        3. Let physics settle.
        4. Return initial observation.

        Args:
            seed:  Random seed for reproducibility.
            cube_positions:  Override random cube placement.

        Returns:
            Observation dict (see class docstring).
        """
        import carb

        if seed is not None:
            np.random.seed(seed)

        self._step_count = 0
        self._episode_count += 1

        # --- Randomise cube positions ---
        if cube_positions is not None:
            self._cube_positions = list(cube_positions)
        else:
            self._cube_positions = self._randomize_cube_positions()
        self._cube_positions_np = np.array(self._cube_positions, dtype=np.float32)

        self._move_cubes(self._cube_positions)

        # Ensure timeline is playing after modifying stage prims.
        # _move_cubes modifies USD prims directly which can invalidate
        # physics tensor entities — play() re-initializes them.
        import isaacsim.core.experimental.utils.app as _app_utils
        _app_utils.play()
        if self._app_update is not None:
            self._app_update()

        # --- Reset robot to home ---
        self._franka.set_dof_position_targets(
            self.HOME_POSITION.reshape(1, -1),
            dof_indices=list(range(self.TOTAL_DOF)),
        )
        # Step physics to let the arm reach home (skip if arm already there)
        for _ in range(60):
            self._physics_step()

        carb.log_info(f"[IsaacEnv] Episode {self._episode_count} started "
                      f"(seed={seed}, cubes={self._cube_positions}).")

        return self._get_obs()

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    def step(
        self,
        action: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """Apply an action and advance the simulation.

        Args:
            action:  Joint position targets.
                     Shape (7,) for arm-only or (9,) for arm+gripper.

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        action = np.asarray(action, dtype=np.float32)

        # Clip to joint limits
        if len(action) == self.ARM_DOF:
            action = np.clip(action, self.JOINT_LIMITS_LOW[:7], self.JOINT_LIMITS_HIGH[:7])
            # Keep gripper open by default
            full_action = np.concatenate([action, np.array([0.04, 0.04], dtype=np.float32)])
        elif len(action) == self.TOTAL_DOF:
            action = np.clip(action, self.JOINT_LIMITS_LOW, self.JOINT_LIMITS_HIGH)
            full_action = action
        else:
            raise ValueError(
                f"Action must be shape ({self.ARM_DOF},) or ({self.TOTAL_DOF},), "
                f"got {action.shape}"
            )

        # Apply target
        self._franka.set_dof_position_targets(
            full_action.reshape(1, -1).astype(np.float32),
            dof_indices=list(range(self.TOTAL_DOF)),
        )

        # Step physics N times
        for _ in range(self._substeps):
            self._physics_step()

        self._step_count += 1

        # Compute reward + success in one pass
        reward, terminated = self._compute_reward_and_done()
        truncated = self._step_count >= self._max_episode_steps

        obs = self._get_obs()
        info = {
            "step": self._step_count,
            "episode": self._episode_count,
            "success": bool(terminated),
            "truncated": bool(truncated),
            "cube_positions": list(self._cube_positions),
        }

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Close
    # ------------------------------------------------------------------
    def close(self) -> None:
        """Clean up resources and break reference cycles."""
        if self._ros2_node is not None:
            try:
                self._ros2_node.destroy_node()
            except Exception:
                pass
            self._ros2_node = None
        self._franka = None
        self._scene = {}
        self._app_update = None

    # ------------------------------------------------------------------
    # Physics
    # ------------------------------------------------------------------
    def _physics_step(self) -> None:
        """Advance physics by one timestep via the bound SimulationApp.update."""
        if self._app_update is not None:
            self._app_update()

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------
    def _get_obs(self) -> dict[str, np.ndarray]:
        """Collect and return the current observation."""
        # Joint positions & velocities
        try:
            dof_pos = self._franka.get_dof_positions().numpy().flatten().astype(np.float32)
            self._last_joint_positions = dof_pos.copy()
        except Exception:
            dof_pos = self._last_joint_positions

        try:
            dof_vel = self._franka.get_dof_velocities().numpy().flatten().astype(np.float32)
            self._last_joint_velocities = dof_vel.copy()
        except Exception:
            dof_vel = self._last_joint_velocities

        # End-effector pose
        try:
            pos, quat = self._franka.end_effector_link.get_world_poses()
            pos_np = pos.numpy().flatten().astype(np.float32)
            quat_np = quat.numpy().flatten().astype(np.float32)
            self._last_ee_position = pos_np.copy()
            self._last_ee_orientation = quat_np.copy()  # (w, x, y, z)
        except Exception:
            pos_np = self._last_ee_position
            quat_np = self._last_ee_orientation

        # Gripper width (approximate from finger joint positions)
        try:
            finger_sum = float(dof_pos[7] + dof_pos[8])
            self._last_gripper_width = max(0.0, 0.08 - finger_sum)  # 8cm max open
        except Exception:
            pass

        obs = OrderedDict([
            ("joint_positions", dof_pos.astype(np.float32)),
            ("joint_velocities", dof_vel.astype(np.float32)),
            ("ee_position", pos_np.astype(np.float32)),
            ("ee_orientation", quat_np.astype(np.float32)),  # (w, x, y, z)
            ("gripper_width", np.array([self._last_gripper_width], dtype=np.float32)),
        ])

        # RGB & Depth — pre-allocated empty; populated when ROS2 camera is active
        obs["rgb"] = self._empty_rgb
        obs["depth"] = self._empty_depth

        return obs

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------
    def _compute_reward_and_done(self) -> tuple[float, bool]:
        """Compute dense reward and success flag in one pass.

        Reward shaping (dense → sparse):
          - Reach:  -L2(gripper, nearest_cube)
          - Grasp:  +1.0  when gripper closes near a cube
          - Lift:   +2.0  when cube is lifted above table surface
          - Place:  +5.0  when cube is on the target pad
          - Success: +10.0 when any cube is on the target pad
        """
        if len(self._cube_positions_np) == 0:
            return 0.0, False

        ee_pos = self._last_ee_position

        # --- Distances to all cubes ---
        cube_to_ee = np.linalg.norm(self._cube_positions_np - ee_pos.reshape(1, 3), axis=1)
        nearest_idx = int(np.argmin(cube_to_ee))
        nearest_dist = float(cube_to_ee[nearest_idx])

        # --- Target pad distances (for both reward and success) ---
        cube_to_target = np.linalg.norm(
            self._cube_positions_np - self._target_center.reshape(1, 3), axis=1)
        any_on_target = bool(np.any(cube_to_target < self._scene["config"]["target_radius"]))
        nearest_on_target = cube_to_target[nearest_idx] < self._scene["config"]["target_radius"]

        # --- Reward shaping ---
        reward = -nearest_dist                                      # reach
        gripper_closed = self._last_gripper_width < 0.02
        if nearest_dist < 0.05 and gripper_closed:
            reward += 1.0                                           # grasp
            cube_z = float(self._cube_positions_np[nearest_idx, 2])
            if cube_z > self._table_surface_z + 0.05:
                reward += 2.0                                       # lift
            if nearest_on_target:
                reward += 5.0                                       # place
        if any_on_target:
            reward += 10.0                                          # success

        return float(reward), any_on_target

    # ------------------------------------------------------------------
    # Cube manipulation helpers
    # ------------------------------------------------------------------
    def _randomize_cube_positions(
        self,
        num_cubes: int = 3,
        x_range: tuple[float, float] = (0.45, 0.65),
        y_range: tuple[float, float] = (-0.15, 0.15),
        z_table: float = 0.26,  # table surface + half cube height
    ) -> list[tuple[float, float, float]]:
        """Generate random cube positions on the table surface.

        Ensures minimum separation between cubes to avoid overlap.
        """
        min_sep_sq = 0.0036  # 0.06² m²
        positions: list[tuple[float, float, float]] = []
        attempts = 0
        max_attempts = 100

        while len(positions) < num_cubes and attempts < max_attempts:
            x = np.random.uniform(*x_range)
            y = np.random.uniform(*y_range)
            candidate = (float(x), float(y), z_table)

            # Check squared separation from already-placed cubes
            too_close = any(
                (candidate[0] - p[0]) ** 2 + (candidate[1] - p[1]) ** 2 < min_sep_sq
                for p in positions
            )
            if not too_close:
                positions.append(candidate)
            attempts += 1

        # Fallback: use fixed positions if randomization fails
        if len(positions) < num_cubes:
            positions = [
                (0.50, -0.10, z_table),
                (0.55, 0.05, z_table),
                (0.60, -0.05, z_table),
            ][:num_cubes]

        return positions

    def _move_cubes(self, positions: list[tuple[float, float, float]]) -> None:
        """Move existing cube prims to new world positions."""
        import omni
        from pxr import Gf, UsdGeom

        stage = omni.usd.get_context().get_stage()
        for i, (px, py, pz) in enumerate(positions):
            prim = stage.GetPrimAtPath(f"/World/Cube{i}")
            if prim.IsValid():
                UsdGeom.XformCommonAPI(prim).SetTranslate(Gf.Vec3d(px, py, pz))

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def franka(self) -> Any:
        """The Franka robot instance."""
        return self._franka

    @property
    def scene(self) -> dict[str, Any]:
        """The task scene dict (table, cubes, basket, cameras)."""
        return self._scene

    @property
    def joint_positions(self) -> np.ndarray:
        """Latest cached joint positions."""
        return self._last_joint_positions

    @property
    def ee_position(self) -> np.ndarray:
        """Latest cached end-effector position (x, y, z)."""
        return self._last_ee_position

    @property
    def ee_orientation(self) -> np.ndarray:
        """Latest cached end-effector orientation quaternion (w, x, y, z)."""
        return self._last_ee_orientation


# ===================================================================
# Internal ROS2 Node (used when enable_ros2=True)
# ===================================================================
class _IsaacEnvROS2Node:
    """Lightweight ROS2 node that mirrors the env state onto topics.

    Publishes /joint_states and /ee_pose; subscribes to /joint_trajectory
    for external action injection (e.g. from action_pub.py).
    """

    def __init__(self, env: IsaacEnv) -> None:
        self._env = env

        import rclpy
        from geometry_msgs.msg import PoseStamped
        from rclpy.node import Node
        from sensor_msgs.msg import JointState
        from std_msgs.msg import Header
        from trajectory_msgs.msg import JointTrajectory

        self._node = Node("isaac_env_bridge")

        # Publishers
        self._js_pub = self._node.create_publisher(JointState, "/joint_states", 10)
        self._ee_pub = self._node.create_publisher(PoseStamped, "/ee_pose", 10)

        # Subscribers
        self._traj_sub = self._node.create_subscription(
            JointTrajectory, "/joint_trajectory", self._trajectory_callback, 10,
        )

        # Latest received action
        self._latest_action: Optional[np.ndarray] = None

        # Timer
        self._timer = self._node.create_timer(1.0 / 30.0, self._publish_state)

    def _publish_state(self) -> None:
        """Publish current joint states and ee pose."""
        now = self._node.get_clock().now().nanoseconds / 1e9
        env = self._env

        # JointState
        js = JointState()
        js.header = Header()
        js.header.stamp.sec = int(now)
        js.header.stamp.nanosec = int((now % 1) * 1e9)
        js.header.frame_id = "franka_base"
        js.name = IsaacEnv.JOINT_NAMES
        js.position = env.joint_positions.tolist()
        js.velocity = []
        js.effort = []
        self._js_pub.publish(js)

        # PoseStamped
        ps = PoseStamped()
        ps.header = Header()
        ps.header.stamp.sec = int(now)
        ps.header.stamp.nanosec = int((now % 1) * 1e9)
        ps.header.frame_id = "world"
        ps.pose.position.x = float(env.ee_position[0])
        ps.pose.position.y = float(env.ee_position[1])
        ps.pose.position.z = float(env.ee_position[2])
        ps.pose.orientation.w = float(env.ee_orientation[0])
        ps.pose.orientation.x = float(env.ee_orientation[1])
        ps.pose.orientation.y = float(env.ee_orientation[2])
        ps.pose.orientation.z = float(env.ee_orientation[3])
        self._ee_pub.publish(ps)

    def _trajectory_callback(self, msg) -> None:
        """Store latest trajectory command for env to apply."""
        if msg.points:
            positions = msg.points[0].positions
            self._latest_action = np.array(positions, dtype=np.float32)

    def get_latest_action(self) -> Optional[np.ndarray]:
        """Return the most recently received trajectory action, if any."""
        action = self._latest_action
        self._latest_action = None  # consume once
        return action

    def destroy_node(self) -> None:
        self._timer.cancel()
        self._node.destroy_node()
        self._env = None  # break reference cycle


# ===================================================================
# Standalone test (run inside Isaac Sim)
# ===================================================================
def _standalone_test() -> None:
    """Quick smoke test of the environment.

    Usage:
        source /opt/ros/jazzy/setup.bash
        ~/isaac-sim-standalone-6.0.1-linux-x86_64/python.sh src/envs/isaac_env.py
    """
    import argparse
    import sys
    from pathlib import Path

    # Ensure project src/ is on sys.path
    _project_root = Path(__file__).resolve().parent.parent.parent
    if str(_project_root / "src") not in sys.path:
        sys.path.insert(0, str(_project_root / "src"))

    parser_ = argparse.ArgumentParser()
    parser_.add_argument("--headless", action="store_true")
    parser_.add_argument("--episodes", type=int, default=2,
                         help="Number of reset cycles to test.")
    parser_.add_argument("--steps", type=int, default=20,
                         help="Steps per episode.")
    args_, unknown_ = parser_.parse_known_args()

    from isaacsim import SimulationApp

    sim_app_ = SimulationApp({"renderer": "RayTracedLighting",
                              "headless": args_.headless})

    import carb
    import isaacsim.core.experimental.utils.app as app_utils

    # Enable extensions
    app_utils.enable_extension("isaacsim.ros2.bridge")
    app_utils.enable_extension("isaacsim.robot.experimental.manipulators.examples")
    sim_app_.update()

    # Create env (pass simulation_app so physics stepping works)
    env = IsaacEnv(headless=args_.headless, enable_ros2=False,
                   simulation_app=sim_app_)
    sim_app_.update()

    # Start simulation
    app_utils.play()
    sim_app_.update()

    carb.log_info(f"[isaac_env test] Running {args_.episodes} episodes "
                  f"× {args_.steps} steps each...")

    for ep in range(args_.episodes):
        obs = env.reset(seed=42 + ep)
        carb.log_info(f"  Episode {ep} — joint_positions: "
                      f"{obs['joint_positions'].round(3)}")
        carb.log_info(f"               ee_position: "
                      f"{obs['ee_position'].round(3)}")

        for step in range(args_.steps):
            # Simple action: add small random noise around current position
            action = obs["joint_positions"][:7].copy()
            action += np.random.normal(0, 0.02, size=7).astype(np.float32)

            obs, reward, terminated, truncated, info = env.step(action)

            if step % 5 == 0:
                carb.log_info(f"    step {step:3d}  reward={reward:.3f}  "
                             f"ee_pos={obs['ee_position'].round(3)}")

            if terminated:
                carb.log_info(f"    ✅ SUCCESS at step {step}")
                break
            if truncated:
                carb.log_info(f"    ⏰ Truncated at step {step}")
                break

        carb.log_info(f"  Episode {ep} done ({info['step']} steps, "
                      f"success={info['success']})")

    env.close()
    app_utils.stop()
    sim_app_.close()
    carb.log_info("[isaac_env test] Done.")


if __name__ == "__main__":
    _standalone_test()
