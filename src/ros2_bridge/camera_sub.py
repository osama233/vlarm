#!/usr/bin/env python3
"""ROS2 Camera Subscriber — Subscribe to camera images from Isaac Sim.

Run this OUTSIDE Isaac Sim, using system ROS2:
    source /opt/ros/jazzy/setup.bash
    python src/ros2_bridge/camera_sub.py

Subscribes to:
    /rgb          (sensor_msgs/Image)      — RGB camera images
    /depth        (sensor_msgs/Image)      — Depth images
    /camera_info  (sensor_msgs/CameraInfo) — Camera intrinsics

Usage:
    # View camera feed (requires cv2)
    python src/ros2_bridge/camera_sub.py --view

    # Save frames to disk
    python src/ros2_bridge/camera_sub.py --save-dir captured_frames/

    # Subscribe to depth only
    python src/ros2_bridge/camera_sub.py --depth-only

    # Save N frames then exit
    python src/ros2_bridge/camera_sub.py --save-dir frames/ --count 50

    # Display with OpenCV preview window
    python src/ros2_bridge/camera_sub.py --display
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image

# Optional: OpenCV for display/save
try:
    import cv2
    import numpy as np
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# Optional: ROS2 CV bridge for image conversion
try:
    from cv_bridge import CvBridge
    HAS_CV_BRIDGE = True
except ImportError:
    HAS_CV_BRIDGE = False


class CameraSubscriber(Node):
    """Subscribes to RGB, depth, and camera info topics from Isaac Sim."""

    def __init__(self, save_dir: str | None = None,
                 display: bool = False,
                 depth_only: bool = False) -> None:
        super().__init__("camera_subscriber")

        self._save_dir = Path(save_dir) if save_dir else None
        self._display = display
        self._depth_only = depth_only
        self._rgb_count = 0
        self._depth_count = 0

        # Latest messages
        self._rgb_image: Image | None = None
        self._depth_image: Image | None = None
        self._camera_info: CameraInfo | None = None

        # CV Bridge for image conversion
        if HAS_CV_BRIDGE:
            self._bridge = CvBridge()
        else:
            self._bridge = None

        # Setup save directory
        if self._save_dir:
            rgb_dir = self._save_dir / "rgb"
            depth_dir = self._save_dir / "depth"
            rgb_dir.mkdir(parents=True, exist_ok=True)
            depth_dir.mkdir(parents=True, exist_ok=True)
            self.get_logger().info(f"Saving frames to {self._save_dir}")

        # Subscribers
        if not depth_only:
            self._rgb_sub = self.create_subscription(
                Image, "/rgb", self._rgb_callback, 10
            )
        self._depth_sub = self.create_subscription(
            Image, "/depth", self._depth_callback, 10
        )
        self._info_sub = self.create_subscription(
            CameraInfo, "/camera_info", self._camera_info_callback, 10
        )

        self.get_logger().info("CameraSubscriber ready.")
        topic_list = ["/depth", "/camera_info"]
        if not depth_only:
            topic_list.insert(0, "/rgb")
        self.get_logger().info(f"  Subscriptions: {', '.join(topic_list)}")

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    def _rgb_callback(self, msg: Image) -> None:
        self._rgb_image = msg
        self._rgb_count += 1
        self.get_logger().info(
            f"[RGB #{self._rgb_count}] {msg.width}x{msg.height}, "
            f"encoding={msg.encoding}"
        )

        if self._display and HAS_CV2:
            self._show_image(msg, f"RGB ({msg.width}x{msg.height})")

        if self._save_dir:
            self._save_image(msg, self._save_dir / "rgb", self._rgb_count, "rgb")

    def _depth_callback(self, msg: Image) -> None:
        self._depth_image = msg
        self._depth_count += 1
        self.get_logger().info(
            f"[Depth #{self._depth_count}] {msg.width}x{msg.height}, "
            f"encoding={msg.encoding}"
        )

        if self._display and HAS_CV2:
            self._show_image(msg, f"Depth ({msg.width}x{msg.height})", is_depth=True)

        if self._save_dir:
            self._save_image(msg, self._save_dir / "depth", self._depth_count, "depth")

    def _camera_info_callback(self, msg: CameraInfo) -> None:
        if self._camera_info is None:
            self._camera_info = msg
            self.get_logger().info(
                f"[CameraInfo] {msg.width}x{msg.height}, "
                f"fx={msg.k[0]:.1f}, fy={msg.k[4]:.1f}, "
                f"cx={msg.k[2]:.1f}, cy={msg.k[5]:.1f}"
            )

    # ------------------------------------------------------------------
    # Image handling
    # ------------------------------------------------------------------
    def _image_to_numpy(self, msg: Image) -> "np.ndarray | None":
        """Convert a ROS Image message to a numpy array.

        Uses cv_bridge if available, otherwise manual conversion.
        """
        if not HAS_CV2:
            return None

        if self._bridge:
            return self._bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        else:
            # Manual conversion for common encodings
            data = np.frombuffer(msg.data, dtype=np.uint8)
            if msg.encoding in ("rgb8", "bgr8"):
                return data.reshape((msg.height, msg.width, 3))
            elif msg.encoding in ("mono8", "8UC1"):
                return data.reshape((msg.height, msg.width))
            elif msg.encoding in ("16UC1", "mono16"):
                return np.frombuffer(msg.data, dtype=np.uint16).reshape(
                    (msg.height, msg.width)
                )
            elif msg.encoding == "32FC1":
                return np.frombuffer(msg.data, dtype=np.float32).reshape(
                    (msg.height, msg.width)
                )
            else:
                # Generic: try to reshape
                self.get_logger().warn(f"Unsupported encoding: {msg.encoding}")
                return None

    def _show_image(self, msg: Image, title: str,
                    is_depth: bool = False) -> None:
        """Display image in an OpenCV window."""
        if not HAS_CV2:
            return

        img = self._image_to_numpy(msg)
        if img is None:
            return

        if is_depth:
            # Normalize depth for display
            if img.dtype == np.float32:
                img_disp = np.clip(img, 0.0, 5.0) / 5.0
            else:
                img_disp = img.astype(np.float32) / np.max(img)
            img_disp = (img_disp * 255).astype(np.uint8)
            cv2.imshow(title, img_disp)
        else:
            # RGB: convert to BGR for OpenCV
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imshow(title, img)

        cv2.waitKey(1)

    def _save_image(self, msg: Image, directory: Path,
                    count: int, prefix: str) -> None:
        """Save image to disk as PNG."""
        if not HAS_CV2:
            self.get_logger().warn("OpenCV not available, cannot save images.")
            return

        img = self._image_to_numpy(msg)
        if img is None:
            return

        filename = directory / f"{prefix}_{count:06d}_{int(time.time() * 1000)}.png"

        if prefix == "depth":
            # Save depth as 16-bit PNG or 32-bit TIFF
            if img.dtype == np.float32:
                np.save(str(filename.with_suffix(".npy")), img)
            else:
                cv2.imwrite(str(filename), img)
        else:
            # RGB: convert to BGR for OpenCV saving
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(filename), img)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def rgb_image(self) -> Image | None:
        """Most recent RGB Image, or None."""
        return self._rgb_image

    @property
    def depth_image(self) -> Image | None:
        """Most recent depth Image, or None."""
        return self._depth_image

    @property
    def camera_info(self) -> CameraInfo | None:
        """CameraInfo (received once), or None."""
        return self._camera_info

    @property
    def rgb_count(self) -> int:
        """Number of RGB frames received."""
        return self._rgb_count

    @property
    def depth_count(self) -> int:
        """Number of depth frames received."""
        return self._depth_count

    def close(self) -> None:
        """Clean up OpenCV windows."""
        if HAS_CV2:
            cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Subscribe to camera images from Isaac Sim"
    )
    parser.add_argument("--save-dir", type=str, default=None, metavar="DIR",
                        help="Save frames to this directory")
    parser.add_argument("--display", action="store_true",
                        help="Display images in OpenCV window")
    parser.add_argument("--depth-only", action="store_true",
                        help="Subscribe to depth only")
    parser.add_argument("--count", type=int, default=0, metavar="N",
                        help="Exit after receiving N frames (0=unlimited)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-frame log output")

    args = parser.parse_args()

    if args.display and not HAS_CV2:
        print("ERROR: --display requires OpenCV (pip install opencv-python)")
        sys.exit(1)
    if args.save_dir and not HAS_CV2:
        print("ERROR: --save-dir requires OpenCV (pip install opencv-python)")
        sys.exit(1)

    rclpy.init(args=sys.argv)
    node = CameraSubscriber(
        save_dir=args.save_dir,
        display=args.display,
        depth_only=args.depth_only,
    )

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            frames_received = (node.rgb_count + node.depth_count
                               if not args.depth_only else node.depth_count)
            if args.count > 0 and frames_received >= args.count:
                node.get_logger().info(
                    f"Received {frames_received} frames. Exiting."
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
