#!/usr/bin/env python3
"""ROS2 Action Publisher — Publishes joint trajectory commands to Isaac Sim.

Run this OUTSIDE Isaac Sim, using system ROS2:
    source /opt/ros/jazzy/setup.bash
    python src/ros2_bridge/action_pub.py

This node publishes JointTrajectory messages to /joint_trajectory,
which the Franka bridge inside Isaac Sim subscribes to.

Usage:
    # Send a predefined "home" trajectory
    python src/ros2_bridge/action_pub.py --home

    # Send a predefined "pick" trajectory (move to grasp pose)
    python src/ros2_bridge/action_pub.py --pick

    # Send a predefined "place" trajectory
    python src/ros2_bridge/action_pub.py --place

    # Open/close gripper
    python src/ros2_bridge/action_pub.py --gripper open
    python src/ros2_bridge/action_pub.py --gripper close

    # Send custom joint positions (7 arm + 2 gripper)
    python src/ros2_bridge/action_pub.py --positions 0.0 -0.5 0.0 -2.5 0.0 2.5 0.7 0.04 0.04
"""

from __future__ import annotations

import argparse
import sys
import time

import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

# Franka joint names (must match what the bridge expects)
JOINT_NAMES = [
    "panda_joint1", "panda_joint2", "panda_joint3",
    "panda_joint4", "panda_joint5", "panda_joint6",
    "panda_joint7", "panda_finger_joint1", "panda_finger_joint2",
]

# Predefined poses (joint values in radians for arm, meters for gripper)
# Based on Franka default pose: [0.012, -0.568, 0.0, -2.811, 0.0, 3.037, 0.741, 0.04, 0.04]
PREDEFINED = {
    "home": [0.012, -0.568, 0.0, -2.811, 0.0, 3.037, 0.741, 0.04, 0.04],
    # Pick pose: arm forward/down, gripper open
    "pick": [0.0, -0.3, 0.0, -1.8, 0.0, 1.5, 0.5, 0.04, 0.04],
    # Place pose: arm raised slightly, gripper closed
    "place": [0.3, -0.2, 0.0, -1.5, 0.0, 1.3, 0.6, 0.0, 0.0],
    # Ready pose: neutral arm, gripper open
    "ready": [0.0, -0.4, 0.0, -2.0, 0.0, 2.0, 0.7, 0.04, 0.04],
}


class ActionPublisher(Node):
    """Publishes JointTrajectory commands to control the Franka robot."""

    def __init__(self) -> None:
        super().__init__("action_publisher")
        self._publisher = self.create_publisher(
            JointTrajectory, "/joint_trajectory", 10
        )
        self.get_logger().info("ActionPublisher ready. Publishing to /joint_trajectory")

    def send_trajectory(self, positions: list[float],
                        duration_sec: float = 2.0,
                        joint_names: list[str] | None = None) -> None:
        """Send a single-point trajectory (move to target positions).

        Args:
            positions: Target joint positions [j1..j7, finger1, finger2].
            duration_sec: Time to reach the target.
            joint_names: Joint names (defaults to Franka joint names).
        """
        if joint_names is None:
            joint_names = JOINT_NAMES

        msg = JointTrajectory()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "world"
        msg.joint_names = joint_names[:len(positions)]

        point = JointTrajectoryPoint()
        point.positions = positions
        point.velocities = [0.0] * len(positions)
        point.time_from_start.sec = int(duration_sec)
        point.time_from_start.nanosec = int((duration_sec % 1) * 1e9)
        msg.points = [point]

        self._publisher.publish(msg)
        self.get_logger().info(
            f"Sent trajectory: {len(positions)} joints, "
            f"duration={duration_sec:.1f}s, "
            f"positions={[f'{p:.3f}' for p in positions]}"
        )

    def send_multi_point_trajectory(self,
                                     waypoints: list[tuple[list[float], float]],
                                     joint_names: list[str] | None = None) -> None:
        """Send a multi-point trajectory (move through waypoints).

        Args:
            waypoints: List of (positions, time_from_start_sec) tuples.
            joint_names: Joint names (defaults to Franka joint names).
        """
        if joint_names is None:
            joint_names = JOINT_NAMES

        msg = JointTrajectory()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "world"
        msg.joint_names = joint_names

        for positions, t in waypoints:
            point = JointTrajectoryPoint()
            point.positions = positions
            point.velocities = [0.0] * len(positions)
            point.time_from_start.sec = int(t)
            point.time_from_start.nanosec = int((t % 1) * 1e9)
            msg.points.append(point)

        self._publisher.publish(msg)
        self.get_logger().info(
            f"Sent multi-point trajectory: {len(waypoints)} waypoints"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Publish joint trajectory commands to Franka in Isaac Sim"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--home", action="store_true",
                       help="Send robot to home position")
    group.add_argument("--pick", action="store_true",
                       help="Send robot to pick pose")
    group.add_argument("--place", action="store_true",
                       help="Send robot to place pose")
    group.add_argument("--ready", action="store_true",
                       help="Send robot to ready pose")
    group.add_argument("--gripper", choices=["open", "close"],
                       help="Open or close gripper")
    group.add_argument("--positions", nargs="+", type=float, metavar="POS",
                       help="Custom joint positions (9 values)")

    parser.add_argument("--duration", type=float, default=2.0,
                        help="Movement duration in seconds (default: 2.0)")
    parser.add_argument("--multi-step", action="store_true",
                        help="Send multi-step pick-and-place sequence")

    args = parser.parse_args()

    rclpy.init(args=sys.argv)
    node = ActionPublisher()

    try:
        if args.home:
            node.send_trajectory(PREDEFINED["home"], duration_sec=args.duration)

        elif args.pick:
            node.send_trajectory(PREDEFINED["pick"], duration_sec=args.duration)

        elif args.place:
            node.send_trajectory(PREDEFINED["place"], duration_sec=args.duration)

        elif args.ready:
            node.send_trajectory(PREDEFINED["ready"], duration_sec=args.duration)

        elif args.gripper:
            home_pos = PREDEFINED["home"]
            if args.gripper == "open":
                home_pos = home_pos[:7] + [0.04, 0.04]
            else:
                home_pos = home_pos[:7] + [0.0, 0.0]
            node.send_trajectory(home_pos, duration_sec=0.5)

        elif args.positions:
            if len(args.positions) not in (7, 9):
                node.get_logger().error(
                    f"Expected 7 or 9 joint positions, got {len(args.positions)}"
                )
                sys.exit(1)
            # Pad to 9 if only 7 (arm only)
            positions = list(args.positions)
            if len(positions) == 7:
                positions += [0.04, 0.04]  # default gripper open
            node.send_trajectory(positions, duration_sec=args.duration)

        elif args.multi_step:
            # Multi-step pick-and-place sequence
            node.get_logger().info("Running multi-step pick-and-place...")
            waypoints = [
                (PREDEFINED["ready"], 1.0),            # move to ready
                (PREDEFINED["pick"], 3.0),             # move to pick
                ([0.0, -0.3, 0.0, -1.8, 0.0, 1.5, 0.5, 0.0, 0.0], 4.0),  # close gripper
                (PREDEFINED["ready"], 6.0),            # lift object
                (PREDEFINED["place"], 8.0),            # move to place
                ([0.3, -0.2, 0.0, -1.5, 0.0, 1.3, 0.6, 0.04, 0.04], 9.0),  # open gripper
                (PREDEFINED["home"], 11.0),            # return home
            ]
            node.send_multi_point_trajectory(waypoints)

        else:
            # Default: send home
            node.get_logger().info("No action specified, sending home position.")
            node.send_trajectory(PREDEFINED["home"], duration_sec=args.duration)

        # Give time for message to be sent
        rclpy.spin_once(node, timeout_sec=0.5)

    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
