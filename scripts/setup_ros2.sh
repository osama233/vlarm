#!/bin/bash
# VLARM — ROS2 Environment Setup Helper
#
# Usage:
#   source scripts/setup_ros2.sh              # activate ROS2 env
#   source scripts/setup_ros2.sh action_pub   # run action_pub directly
#   source scripts/setup_ros2.sh state_sub    # run state_sub directly
#
# After sourcing, python and ros2 CLI work correctly.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Source ROS2 Jazzy
if [ -f /opt/ros/jazzy/setup.bash ]; then
    source /opt/ros/jazzy/setup.bash
else
    echo "ERROR: /opt/ros/jazzy/setup.bash not found"
    return 1
fi

# Force system Python 3.12 (conda Python 3.14 can't load rclpy)
# Prepend /usr/bin to PATH so python/python3 resolve to system Python
export PATH="/usr/bin:$PATH"
export PYTHON=/usr/bin/python3.12

# Add VLARM src to PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"

echo "[VLARM ROS2] ROS_DISTRO=$ROS_DISTRO, Python=$($PYTHON --version 2>&1)"

# If a script name was given, run it with any extra args
SCRIPT_NAME="$1"
if [ -n "$SCRIPT_NAME" ]; then
    shift
    case "$SCRIPT_NAME" in
        action_pub)
            $PYTHON "$PROJECT_ROOT/src/ros2_bridge/action_pub.py" "$@"
            ;;
        state_sub)
            $PYTHON "$PROJECT_ROOT/src/ros2_bridge/state_sub.py" "$@"
            ;;
        camera_sub)
            $PYTHON "$PROJECT_ROOT/src/ros2_bridge/camera_sub.py" "$@"
            ;;
        *)
            echo "Unknown script: $SCRIPT_NAME"
            echo "Available: action_pub, state_sub, camera_sub"
            ;;
    esac
fi
