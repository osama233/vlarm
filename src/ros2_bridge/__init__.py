"""VLARM ROS2 Bridge Package.

Provides ROS2 communication with Isaac Sim's Franka Panda robot.

Modules:
    action_pub.py  — Publish JointTrajectory commands to control the robot.
    state_sub.py   — Subscribe to /joint_states and /ee_pose feedback.
    camera_sub.py  — Subscribe to /rgb, /depth, and /camera_info from Isaac Sim.

Usage (external nodes, use system ROS2):
    source /opt/ros/jazzy/setup.bash
    python -m ros2_bridge.action_pub --home
    python -m ros2_bridge.state_sub
    python -m ros2_bridge.camera_sub --save-dir frames/
"""

from ros2_bridge.action_pub import ActionPublisher, PREDEFINED
from ros2_bridge.camera_sub import CameraSubscriber
from ros2_bridge.state_sub import StateSubscriber

__all__ = [
    "ActionPublisher",
    "PREDEFINED",
    "StateSubscriber",
    "CameraSubscriber",
]
