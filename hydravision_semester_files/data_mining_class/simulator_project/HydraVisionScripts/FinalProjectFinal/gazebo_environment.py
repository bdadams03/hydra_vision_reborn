#!/usr/bin/env python3
"""
gazebo_environment.py
======================
Utilities for:
  1. Controlling Gazebo lighting (where supported).
  2. Spawning a target object in front of the TurtleBot4 camera.
  3. Collecting ground-truth bounding boxes from the camera stream.

Compatible with ROS2 Jazzy + Gazebo Harmonic (gz-sim).

Usage:
    python3 gazebo_environment.py --preset standard --spawn
    python3 gazebo_environment.py --collect_gt --frames 100
"""

import rclpy
from rclpy.node import Node
import argparse
import json
import time
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple

from geometry_msgs.msg import Pose
from sensor_msgs.msg import CameraInfo, Image
from cv_bridge import CvBridge

# ── gz-sim / ros_gz_interfaces (ROS2 Jazzy) ──────────────────────────────────
try:
    from ros_gz_interfaces.srv import SpawnEntity, DeleteEntity
    ROS_GZ_AVAILABLE = True
    print("[INFO] ros_gz_interfaces found — spawn/delete enabled.")
except ImportError:
    ROS_GZ_AVAILABLE = False
    print("[WARN] ros_gz_interfaces not available — spawn/delete disabled.")
    print("       Install with: sudo apt install ros-jazzy-ros-gz-interfaces")

# ── Lighting presets ──────────────────────────────────────────────────────────

LIGHT_PRESETS: Dict[str, Dict] = {
    "standard": {
        "diffuse":              (0.8, 0.8, 0.8, 1.0),
        "ambient":              (0.3, 0.3, 0.3, 1.0),
        "attenuation_constant": 1.0,
        "attenuation_linear":   0.0,
        "attenuation_quadratic":0.0,
    },
    "bright": {
        "diffuse":              (1.0, 1.0, 1.0, 1.0),
        "ambient":              (0.6, 0.6, 0.6, 1.0),
        "attenuation_constant": 1.0,
        "attenuation_linear":   0.0,
        "attenuation_quadratic":0.0,
    },
    "dim": {
        "diffuse":              (0.3, 0.3, 0.3, 1.0),
        "ambient":              (0.1, 0.1, 0.1, 1.0),
        "attenuation_constant": 1.0,
        "attenuation_linear":   0.01,
        "attenuation_quadratic":0.001,
    },
}

# ── SDF for a simple red box target object ────────────────────────────────────

_TARGET_SDF = """
<?xml version="1.0" ?>
<sdf version="1.6">
  <model name="benchmark_target">
    <static>true</static>
    <link name="link">
      <visual name="visual">
        <geometry>
          <box><size>0.3 0.3 0.5</size></box>
        </geometry>
        <material>
          <ambient>1 0 0 1</ambient>
          <diffuse>1 0 0 1</diffuse>
        </material>
      </visual>
      <collision name="collision">
        <geometry>
          <box><size>0.3 0.3 0.5</size></box>
        </geometry>
      </collision>
    </link>
  </model>
</sdf>
"""


# ── Pinhole projector ─────────────────────────────────────────────────────────

class PinholeProjector:
    """Projects a 3-D camera-frame point to pixel coordinates."""

    def __init__(self):
        self.K: Optional[np.ndarray] = None

    def update_from_msg(self, msg: CameraInfo):
        self.K = np.array(msg.k).reshape(3, 3)

    def project(self, xyz: Tuple[float, float, float]) -> Optional[Tuple[int, int]]:
        if self.K is None or xyz[2] <= 0:
            return None
        p  = np.array(xyz)
        uv = self.K @ p
        return int(uv[0] / uv[2]), int(uv[1] / uv[2])


# ── Main environment manager node ─────────────────────────────────────────────

class GazeboEnvironmentManager(Node):

    def __init__(self, args):
        super().__init__("gazebo_environment_manager")
        self.args = args
        self.bridge = CvBridge()
        self.projector = PinholeProjector()

        # Service clients — only created if ros_gz_interfaces is available
        if ROS_GZ_AVAILABLE:
            self._spawn_cli  = self.create_client(SpawnEntity,  "/world/default/create")
            self._delete_cli = self.create_client(DeleteEntity, "/world/default/remove")
        else:
            self._spawn_cli  = None
            self._delete_cli = None

        # gz-sim does not expose set/get entity state over ROS2 directly
        self._set_state = None
        self._get_state = None

        # Ground-truth state
        self._gt_annotations: Dict[int, List[Dict]] = {}
        self._frame_count = 0
        self._cam_info_received = False

        # Always subscribe to camera info for intrinsics
        self._cam_info_sub = self.create_subscription(
            CameraInfo,
            "/oakd/rgb/preview/camera_info",
            self._cam_info_callback,
            10,
        )

        # Image subscription — only needed for GT collection
        if args.collect_gt:
            print("Creating image subscription...", flush=True)
            self._image_sub = self.create_subscription(
                Image,
                "/oakd/rgb/preview/image_raw",
                self._gt_frame_callback,
                10,
            )
            print("Image subscription created.", flush=True)

    # ── Camera info ───────────────────────────────────────────────────────────

    def _cam_info_callback(self, msg: CameraInfo):
        if not self._cam_info_received:
            self.projector.update_from_msg(msg)
            self._cam_info_received = True
            if self.projector.K is not None:
                print(
                    f"[INFO] Camera intrinsics received — "
                    f"fx={self.projector.K[0,0]:.1f} "
                    f"fy={self.projector.K[1,1]:.1f} "
                    f"cx={self.projector.K[0,2]:.1f} "
                    f"cy={self.projector.K[1,2]:.1f}",
                    flush=True,
                )

    # ── Lighting ──────────────────────────────────────────────────────────────

    def apply_lighting_preset(self, preset_name: str):
        """
        Attempts to set light properties via the Gazebo ROS bridge.
        If the service is unavailable (common in gz-sim), prints instructions
        for setting lighting manually in the world file instead.
        """
        preset = LIGHT_PRESETS.get(preset_name)
        if not preset:
            self.get_logger().error(f"Unknown light preset: {preset_name}")
            return

        self.get_logger().info(f"Applying light preset '{preset_name}'...")

        # Try classic Gazebo service first
        try:
            from gazebo_msgs.srv import SetLightProperties
            from std_msgs.msg import ColorRGBA

            cli = self.create_client(SetLightProperties, "/gazebo/set_light_properties")
            if cli.wait_for_service(timeout_sec=3.0):
                req = SetLightProperties.Request()
                req.light_name = "sun"
                d = preset["diffuse"]
                a = preset["ambient"]
                req.diffuse  = ColorRGBA(r=d[0], g=d[1], b=d[2], a=d[3])
                req.ambient  = ColorRGBA(r=a[0], g=a[1], b=a[2], a=a[3])
                req.attenuation_constant  = preset["attenuation_constant"]
                req.attenuation_linear    = preset["attenuation_linear"]
                req.attenuation_quadratic = preset["attenuation_quadratic"]
                future = cli.call_async(req)
                rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
                self.get_logger().info("Light properties updated via service.")
                return
        except (ImportError, Exception):
            pass

        # Fallback — print manual instructions
        d = preset["diffuse"]
        a = preset["ambient"]
        print(
            f"\n[LIGHTING] Service unavailable for gz-sim.\n"
            f"To apply the '{preset_name}' preset manually,\n"
            f"edit the <scene> and <light> blocks in your .world file:\n"
            f"\n"
            f"  <scene>\n"
            f"    <ambient>{a[0]} {a[1]} {a[2]} {a[3]}</ambient>\n"
            f"  </scene>\n"
            f"  <light name='sun' type='directional'>\n"
            f"    <diffuse>{d[0]} {d[1]} {d[2]} {d[3]}</diffuse>\n"
            f"  </light>\n",
            flush=True,
        )

    # ── Object spawning ───────────────────────────────────────────────────────

    def spawn_target_object(self, x: float = 1.5, y: float = 0.0, z: float = 0.25):
        """Spawn a red box 1.5 m in front of the robot."""
        if self._spawn_cli is None:
            print(
                "\n[WARN] Spawn service unavailable (ros_gz_interfaces missing).\n"
                "       Place a target object manually in the Gazebo GUI instead:\n"
                "         1. Click Insert in the Gazebo GUI\n"
                "         2. Add any model ~1.5 m in front of the robot\n"
                "         3. Make sure it is visible in the camera view\n",
                flush=True,
            )
            return

        if not self._spawn_cli.wait_for_service(timeout_sec=5.0):
            print(
                "[WARN] /world/default/create service not responding.\n"
                "       Try placing the object manually in the Gazebo GUI.",
                flush=True,
            )
            return

        req = SpawnEntity.Request()
        req.name = "benchmark_target"
        req.xml  = _TARGET_SDF
        req.initial_pose = Pose()
        req.initial_pose.position.x = x
        req.initial_pose.position.y = y
        req.initial_pose.position.z = z

        future = self._spawn_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)

        if future.result():
            self.get_logger().info(f"Target object spawned at ({x}, {y}, {z})")
        else:
            self.get_logger().warn(
                "Spawn call returned no result — object may already exist."
            )

    def delete_target_object(self):
        if self._delete_cli is None:
            return
        if not self._delete_cli.wait_for_service(timeout_sec=3.0):
            return
        req = DeleteEntity.Request()
        req.name = "benchmark_target"
        future = self._delete_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        self.get_logger().info("Target object deleted.")

    # ── Ground-truth collection ───────────────────────────────────────────────

    def _gt_frame_callback(self, msg: Image):
        if self._frame_count >= self.args.frames:
            return

        print(
            f"[GT] Image received — {msg.width}x{msg.height} "
            f"encoding={msg.encoding}",
            flush=True,
        )

        w, h = msg.width, msg.height

        # Use camera intrinsics + known spawn position for bbox projection.
        # gz-sim does not expose entity state over ROS2, so we use the known
        # spawn position (1.5 m ahead, centred) directly.
        if self._cam_info_received and self.projector.K is not None:
            depth  = 1.5   # metres — matches spawn_target_object x=1.5
            cx_obj = 0.0   # centred horizontally
            cz_obj = 0.0   # centred vertically

            fx  = self.projector.K[0, 0]
            fy  = self.projector.K[1, 1]
            ppx = self.projector.K[0, 2]
            ppy = self.projector.K[1, 2]

            # Box half-dimensions: 0.15 m wide, 0.25 m tall
            hw, hh = 0.15, 0.25
            u_c = int(ppx + fx * cx_obj / depth)
            v_c = int(ppy - fy * cz_obj / depth)
            du  = int(fx * hw / depth)
            dv  = int(fy * hh / depth)

            # Clamp to image bounds
            x1 = max(0, u_c - du)
            y1 = max(0, v_c - dv)
            x2 = min(w, u_c + du)
            y2 = min(h, v_c + dv)
            bbox = [x1, y1, x2, y2]
            print(f"[GT] Projected bbox (intrinsics): {bbox}", flush=True)
        else:
            # Fallback: centre quarter of frame
            bbox = [w // 4, h // 4, 3 * w // 4, 3 * h // 4]
            print(f"[GT] Fallback bbox (no intrinsics yet): {bbox}", flush=True)

        self._gt_annotations[self._frame_count] = [
            {"bbox": bbox, "class_id": 0}
        ]
        self._frame_count += 1
        print(f"[GT] Frame {self._frame_count}/{self.args.frames} saved.", flush=True)

    def _save_gt(self):
        out = Path("benchmark_results") / "ground_truth.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(self._gt_annotations, f, indent=2)
        print(
            f"\n[GT] Annotations saved → {out}\n"
            f"     Total frames annotated: {len(self._gt_annotations)}",
            flush=True,
        )


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Gazebo environment setup for TurtleBot4 benchmark"
    )
    parser.add_argument(
        "--preset", default="standard",
        choices=list(LIGHT_PRESETS.keys()),
        help="Lighting preset to apply (standard / bright / dim)",
    )
    parser.add_argument(
        "--spawn", action="store_true",
        help="Spawn the red box target object in front of the robot",
    )
    parser.add_argument(
        "--collect_gt", action="store_true",
        help="Collect ground-truth bounding boxes for N frames",
    )
    parser.add_argument(
        "--frames", type=int, default=100,
        help="Number of GT frames to collect (default: 100)",
    )
    args = parser.parse_args()

    rclpy.init()
    node = GazeboEnvironmentManager(args)

    # Apply lighting preset
    node.apply_lighting_preset(args.preset)

    # Spawn object if requested
    if args.spawn or args.collect_gt:
        node.spawn_target_object()
        print("[INFO] Waiting 2 s for object to appear in simulation...", flush=True)
        time.sleep(2.0)

    # Collect ground truth
    if args.collect_gt:
        print(
            f"\n[GT] Starting ground-truth collection ({args.frames} frames).\n"
            f"     Make sure the simulator is running and the target object\n"
            f"     is visible in the camera view.\n",
            flush=True,
        )

        last_count  = -1
        no_progress = 0

        while rclpy.ok() and node._frame_count < args.frames:
            rclpy.spin_once(node, timeout_sec=0.5)

            if node._frame_count != last_count:
                last_count  = node._frame_count
                no_progress = 0
            else:
                no_progress += 1

            # Warn every 5 seconds if no frames arrive
            if no_progress >= 10:
                print(
                    "\n[WARN] No frames received in the last 5 s.\n"
                    "       Verify the simulator is running:\n"
                    "         ros2 topic hz /oakd/rgb/preview/image_raw\n",
                    flush=True,
                )
                no_progress = 0

        node._save_gt()
        print(
            "\n[GT] Collection complete.\n"
            "     Set in config.yaml:\n"
            "       ground_truth_path: \"benchmark_results/ground_truth.json\"\n"
            "     Then run: python3 main.py\n",
            flush=True,
        )

    else:
        print(
            "\n[INFO] Environment ready.\n"
            "       Run 'python3 main.py' to start the benchmark.\n",
            flush=True,
        )
        for _ in range(20):
            rclpy.spin_once(node, timeout_sec=0.1)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()