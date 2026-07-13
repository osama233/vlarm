#!/usr/bin/env python3
"""ROS2 State Subscriber — Subscribe to robot state from Isaac Sim.

Run this OUTSIDE Isaac Sim, using system ROS2:
    source /opt/ros/jazzy/setup.bash
    python src/ros2_bridge/state_sub.py

Subscribes to:
    /joint_states  (sensor_msgs/JointState)  — robot joint positions
    /ee_pose       (geometry_msgs/PoseStamped) — end-effector pose

Usage:
    # Print state to console
    python src/ros2_bridge/state_sub.py

    # Log state to a file
    python src/ros2_bridge/state_sub.py --log state_log.csv

    # Listen for N messages then exit
    python src/ros2_bridge/state_sub.py --count 100

    # Quiet mode (no console output)
    python src/ros2_bridge/state_sub.py --quiet
"""

from __future__ import annotations

import argparse
import csv
import sys
import time

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import JointState


class StateSubscriber(Node):
    """Subscribes to robot state topics from Isaac Sim."""

    def __init__(self, log_file: str | None = None, quiet: bool = False) -> None:
        super().__init__("state_subscriber")

        self._quiet = quiet
        self._joint_state: JointState | None = None
        self._ee_pose: PoseStamped | None = None
        self._joint_count: int = 0
        self._ee_count: int = 0

        # Subscribers
        self._js_sub = self.create_subscription(
            JointState, "/joint_states", self._joint_state_callback, 10
        )
        self._ee_sub = self.create_subscription(
            PoseStamped, "/ee_pose", self._ee_pose_callback, 10
        )

        # Optional CSV logging
        self._csv_file = None
        self._csv_writer = None
        if log_file:
            self._csv_file = open(log_file, "w", newline="")
            self._csv_writer = csv.writer(self._csv_file)
            self._csv_writer.writerow([
                "timestamp",
                "j1", "j2", "j3", "j4", "j5", "j6", "j7", "finger1", "finger2",
                "ee_x", "ee_y", "ee_z",
                "ee_qw", "ee_qx", "ee_qy", "ee_qz",
            ])
            self.get_logger().info(f"Logging state to {log_file}")

        self.get_logger().info("StateSubscriber ready.")
        self.get_logger().info("  Subscriptions: /joint_states, /ee_pose")

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    def _joint_state_callback(self, msg: JointState) -> None:
        self._joint_state = msg
        self._joint_count += 1
        if not self._quiet:
            pos_str = "  ".join(
                f"{n}: {p:+.3f}" for n, p in zip(msg.name, msg.position)
            )
            self.get_logger().info(f"[JointState #{self._joint_count}]\n  {pos_str}")

        self._maybe_write_csv()

    def _ee_pose_callback(self, msg: PoseStamped) -> None:
        self._ee_pose = msg
        self._ee_count += 1
        if not self._quiet:
            p = msg.pose.position
            o = msg.pose.orientation
            self.get_logger().info(
                f"[EndEffector #{self._ee_count}] "
                f"pos=({p.x:.3f}, {p.y:.3f}, {p.z:.3f}) "
                f"quat=({o.w:.3f}, {o.x:.3f}, {o.y:.3f}, {o.z:.3f})"
            )

        self._maybe_write_csv()

    def _maybe_write_csv(self) -> None:
        """Write current state to CSV if both joint state and EE pose are available."""
        if self._csv_writer is None:
            return
        if self._joint_state is None or self._ee_pose is None:
            return

        js = self._joint_state
        ee = self._ee_pose
        self._csv_writer.writerow([
            time.time(),
            *js.position[:9],  # 7 arm + 2 gripper
            ee.pose.position.x, ee.pose.position.y, ee.pose.position.z,
            ee.pose.orientation.w, ee.pose.orientation.x,
            ee.pose.orientation.y, ee.pose.orientation.z,
        ])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def joint_state(self) -> JointState | None:
        """Most recent JointState message, or None."""
        return self._joint_state

    @property
    def ee_pose(self) -> PoseStamped | None:
        """Most recent PoseStamped for end-effector, or None."""
        return self._ee_pose

    @property
    def joint_count(self) -> int:
        """Number of JointState messages received."""
        return self._joint_count

    @property
    def ee_count(self) -> int:
        """Number of end-effector pose messages received."""
        return self._ee_count

    def close(self) -> None:
        """Close the CSV log file if open."""
        if self._csv_file:
            self._csv_file.close()
            self.get_logger().info("Log file closed.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Subscribe to Franka robot state from Isaac Sim"
    )
    parser.add_argument("--log", type=str, default=None, metavar="FILE",
                        help="Log joint states + EE pose to CSV file")
    parser.add_argument("--count", type=int, default=0, metavar="N",
                        help="Exit after receiving N joint state messages (0 = unlimited)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress console output")
    args = parser.parse_args()

    rclpy.init(args=sys.argv)
    node = StateSubscriber(log_file=args.log, quiet=args.quiet)

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            if args.count > 0 and node.joint_count >= args.count:
                node.get_logger().info(
                    f"Received {node.joint_count} joint state messages. Exiting."
                )
                break
    except KeyboardInterrupt:
        node.get_logger().info("Interrupted by user.")
    finally:
        node.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
