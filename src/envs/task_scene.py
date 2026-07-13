#!/usr/bin/env python3
"""VLARM Task Scene Builder.

Creates a desktop manipulation scene in Isaac Sim:
  - Table surface with legs
  - Target objects (red, green, blue cubes, 3 cm each)
  - Basket/container for placing objects
  - RGB + Depth cameras pointed at the workspace

All objects are USD prims with rigid-body physics so they interact with
the robot gripper realistically.

Usage (inside Isaac Sim's Python environment):
    from isaacsim import SimulationApp
    simulation_app = SimulationApp({"headless": False})

    # Enable required extensions first
    import isaacsim.core.experimental.utils.app as app_utils
    app_utils.enable_extension("isaacsim.ros2.bridge")
    app_utils.enable_extension("isaacsim.robot.experimental.manipulators.examples")

    from envs.task_scene import build_task_scene
    scene = build_task_scene()

    # Then create Franka robot and start simulation...
"""

from __future__ import annotations

import math
from typing import Any, Optional

# pxr (USD Python bindings) are imported inside each function body using
# plain ``from pxr import X`` (no ``as`` aliasing), because pxr requires
# SimulationApp to be created first.  Each function imports only the
# submodules it needs; CPython caches the imports in sys.modules.

# ---------------------------------------------------------------------------
# Default scene parameters
# ---------------------------------------------------------------------------
# Table top centre at (0.55, 0, 0.25).  Legs extend from floor to underside.
# Top surface is at z ≈ 0.26 (0.25 + 0.02/2).
DEFAULT_TABLE_POSITION = (0.55, 0.0, 0.25)
DEFAULT_TABLE_SIZE = (0.70, 0.50, 0.02)          # width, depth, thickness (m)
DEFAULT_TABLE_COLOR = (0.55, 0.35, 0.15)         # wood-like brown

# Cube objects — sit ON the table surface
DEFAULT_CUBE_SIZE = 0.03                          # 3 cm edge
DEFAULT_CUBE_POSITIONS = [
    (0.50, -0.12, 0.275),  # front-right
    (0.55,  0.00, 0.275),  # centre
    (0.60,  0.10, 0.275),  # back-left
]
DEFAULT_CUBE_COLORS = [
    (1.0, 0.15, 0.15),     # red
    (0.15, 0.50, 1.0),     # blue
    (0.15, 0.85, 0.15),    # green
]

# Target pad — a flat disc on the table surface (placement target)
DEFAULT_TARGET_POSITION = (0.70, -0.18, 0.263)   # centre
DEFAULT_TARGET_RADIUS = 0.10                      # disc radius
DEFAULT_TARGET_THICKNESS = 0.005                  # thin disc
DEFAULT_TARGET_COLOR = (0.90, 0.85, 0.50)         # pale yellow/gold

# Camera — behind and above robot, looking at workspace
DEFAULT_CAMERA_POSITION = (0.30, 0.0, 1.10)
DEFAULT_CAMERA_LOOK_AT = (0.58, 0.0, 0.26)
DEFAULT_CAMERA_RESOLUTION = (640, 480)

# Stage paths
TABLE_PATH = "/World/Table"
TARGET_PATH = "/World/TargetPad"
CAMERA_RGB_PATH = "/World/CameraRGB"
CAMERA_DEPTH_PATH = "/World/CameraDepth"
CUBE_PREFIX = "/World/Cube"


# ===================================================================
# Helpers
# ===================================================================
def _get_stage():
    """Return the current USD stage."""
    import omni
    return omni.usd.get_context().get_stage()


def _create_material(stage, path, color):
    """Create a simple diffuse material with the given RGB colour."""
    from pxr import Gf, Sdf, UsdShade
    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


def _bind_material(prim, material):
    """Bind a material to a prim for rendering."""
    from pxr import UsdShade
    UsdShade.MaterialBindingAPI(prim).Bind(material)


def _add_rigid_body(prim, mass=1.0, is_kinematic=False):
    """Add rigid-body physics to a USD prim."""
    from pxr import UsdPhysics
    rb_api = UsdPhysics.RigidBodyAPI.Apply(prim)
    if is_kinematic:
        rb_api.CreateKinematicEnabledAttr().Set(True)
    else:
        rb_api.CreateKinematicEnabledAttr().Set(False)
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr().Set(mass)
    UsdPhysics.CollisionAPI.Apply(prim)


# ===================================================================
# Scene Elements
# ===================================================================
def build_table(
    position=DEFAULT_TABLE_POSITION,
    size=DEFAULT_TABLE_SIZE,
    color=DEFAULT_TABLE_COLOR,
    stage_path=TABLE_PATH,
):
    """Create a static table surface as a thin cuboid (no legs)."""
    from pxr import Gf, UsdGeom

    stage = _get_stage()
    w, d, t = size
    px, py, pz = position

    table_prim = UsdGeom.Cube.Define(stage, stage_path)
    xform = UsdGeom.XformCommonAPI(table_prim)
    xform.SetScale(Gf.Vec3f(w / 2, d / 2, t / 2))
    xform.SetTranslate(Gf.Vec3d(px, py, pz))
    _add_rigid_body(table_prim.GetPrim(), mass=0.0, is_kinematic=True)
    _bind_material(table_prim.GetPrim(), _create_material(stage, f"{stage_path}/Material", color))
    return table_prim


def spawn_cubes(positions, colors, size=DEFAULT_CUBE_SIZE, stage_prefix=CUBE_PREFIX):
    """Spawn coloured cubes at given world positions (dynamic rigid bodies)."""
    from pxr import Gf, UsdGeom

    stage = _get_stage()
    cubes = []
    half = size / 2

    for i, ((px, py, pz), color) in enumerate(zip(positions, colors)):
        cube = UsdGeom.Cube.Define(stage, f"{stage_prefix}{i}")
        xform = UsdGeom.XformCommonAPI(cube)
        xform.SetScale(Gf.Vec3f(half, half, half))
        xform.SetTranslate(Gf.Vec3d(px, py, pz))
        _add_rigid_body(cube.GetPrim(), mass=0.027, is_kinematic=False)
        _bind_material(cube.GetPrim(), _create_material(stage, f"{stage_prefix}{i}/Material", color))
        cubes.append(cube)

    return cubes


def spawn_target_pad(
    position=DEFAULT_TARGET_POSITION,
    radius=DEFAULT_TARGET_RADIUS,
    thickness=DEFAULT_TARGET_THICKNESS,
    color=DEFAULT_TARGET_COLOR,
    stage_path=TARGET_PATH,
):
    """Create a flat disc on the table surface as the placement target.

    This replaces a basket — cubes placed on the pad count as success.
    A flat disc is simpler, avoids occlusion, and matches standard
    manipulation benchmarks (Ravens, RLBench).
    """
    from pxr import Gf, UsdGeom

    stage = _get_stage()
    px, py, pz = position

    # Root Xform
    root = UsdGeom.Xform.Define(stage, stage_path)
    _add_rigid_body(root.GetPrim(), mass=0.0, is_kinematic=True)

    # Disc (flattened cylinder, rotated to lie flat on XY table surface)
    disc = UsdGeom.Cylinder.Define(stage, f"{stage_path}/Disc")
    dxform = UsdGeom.XformCommonAPI(disc)
    dxform.SetRotate(Gf.Vec3f(90, 0, 0), UsdGeom.XformCommonAPI.RotationOrderXYZ)
    dxform.SetScale(Gf.Vec3f(radius, thickness / 2, radius))
    dxform.SetTranslate(Gf.Vec3d(px, py, pz))
    _bind_material(disc.GetPrim(), _create_material(stage, f"{stage_path}/Material", color))

    return root.GetPrim()


def setup_cameras(
    rgb_position=DEFAULT_CAMERA_POSITION,
    look_at=DEFAULT_CAMERA_LOOK_AT,
    resolution=DEFAULT_CAMERA_RESOLUTION,
    rgb_path=CAMERA_RGB_PATH,
    depth_path=CAMERA_DEPTH_PATH,
):
    """Create RGB and Depth camera prims looking at the workspace.

    Orientation is computed via ``Gf.Matrix4d.SetLookAt``, decomposed into
    translate + rotate components for ``XformCommonAPI``.
    """
    from pxr import Gf, UsdGeom

    stage = _get_stage()

    def _make_camera(path):
        cam = UsdGeom.Camera.Define(stage, path)
        xform = UsdGeom.XformCommonAPI(cam)

        # Build look-at matrix
        mat = Gf.Matrix4d()
        mat.SetLookAt(
            Gf.Vec3d(*rgb_position),
            Gf.Vec3d(*look_at),
            Gf.Vec3d(0.0, 0.0, 1.0),
        )

        # Set translation
        xform.SetTranslate(mat.ExtractTranslation())

        # Extract rotation as Euler angles (XYZ order, degrees)
        rot = mat.ExtractRotation()
        euler_rad = rot.Decompose(
            Gf.Vec3d(1, 0, 0),  # X axis
            Gf.Vec3d(0, 1, 0),  # Y axis
            Gf.Vec3d(0, 0, 1),  # Z axis
        )
        euler_deg = (math.degrees(euler_rad[0]),
                     math.degrees(euler_rad[1]),
                     math.degrees(euler_rad[2]))
        xform.SetRotate(Gf.Vec3f(*euler_deg), UsdGeom.XformCommonAPI.RotationOrderXYZ)

        # Intrinsics
        cam.GetHorizontalApertureAttr().Set(21.0)
        cam.GetVerticalApertureAttr().Set(15.75)
        cam.GetFocalLengthAttr().Set(18.0)
        cam.GetFocusDistanceAttr().Set(400.0)
        cam.GetProjectionAttr().Set("perspective")
        cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 100.0))
        return cam

    return {
        "rgb_prim": _make_camera(rgb_path),
        "depth_prim": _make_camera(depth_path),
        "resolution": resolution,
    }




# ===================================================================
# Main entry point
# ===================================================================
def build_task_scene(
    table_position=DEFAULT_TABLE_POSITION,
    table_size=DEFAULT_TABLE_SIZE,
    cube_positions=None,
    cube_colors=None,
    cube_size=DEFAULT_CUBE_SIZE,
    target_position=DEFAULT_TARGET_POSITION,
    target_radius=DEFAULT_TARGET_RADIUS,
    target_thickness=DEFAULT_TARGET_THICKNESS,
    camera_position=DEFAULT_CAMERA_POSITION,
    camera_look_at=DEFAULT_CAMERA_LOOK_AT,
    camera_resolution=DEFAULT_CAMERA_RESOLUTION,
):
    """Build the full VLARM task scene.

    Returns a dict with keys: table, cubes, basket, cameras, config.
    The ``config`` sub-dict stores all parameters so consumers (e.g.
    ``IsaacEnv``) can read positions without hardcoding constants.
    """
    if cube_positions is None:
        cube_positions = list(DEFAULT_CUBE_POSITIONS)
    if cube_colors is None:
        cube_colors = list(DEFAULT_CUBE_COLORS)

    # Build elements
    table = build_table(position=table_position, size=table_size)
    cubes = spawn_cubes(positions=cube_positions, colors=cube_colors, size=cube_size)
    target = spawn_target_pad(position=target_position, radius=target_radius,
                             thickness=target_thickness)
    cameras = setup_cameras(rgb_position=camera_position, look_at=camera_look_at,
                            resolution=camera_resolution)

    return {
        "table": table,
        "cubes": cubes,
        "target": target,
        "cameras": cameras,
        "config": {
            "table_position": table_position,
            "table_size": table_size,
            "cube_positions": cube_positions,
            "cube_colors": cube_colors,
            "cube_size": cube_size,
            "target_position": target_position,
            "target_radius": target_radius,
            "target_thickness": target_thickness,
            "camera_position": camera_position,
            "camera_look_at": camera_look_at,
            "camera_resolution": camera_resolution,
        },
    }


# ===================================================================
# Standalone preview
# ===================================================================
def _standalone_preview():
    """Launch Isaac Sim, build the scene, and keep running for visual check.

    Usage:
        ~/isaac-sim-standalone-6.0.1-linux-x86_64/python.sh src/envs/task_scene.py
    """
    import argparse
    import sys

    parser_ = argparse.ArgumentParser()
    parser_.add_argument("--headless", action="store_true")
    parser_.add_argument("--test", action="store_true",
                         help="Build scene then exit after 2 s")
    args_, unknown_ = parser_.parse_known_args()

    from isaacsim import SimulationApp

    sim_app_ = SimulationApp({"renderer": "RayTracedLighting",
                              "headless": args_.headless})

    import carb
    import isaacsim.core.experimental.utils.app as app_utils
    from isaacsim.core.experimental.objects import DistantLight, GroundPlane
    import isaacsim.core.experimental.utils.stage as stage_utils

    # World setup
    stage_utils.set_stage_units(meters_per_unit=1.0)
    GroundPlane("/World/GroundPlane")
    DistantLight("/World/DistantLight").set_intensities(intensities=[3000])

    app_utils.enable_extension("isaacsim.ros2.bridge")
    app_utils.enable_extension("isaacsim.robot.experimental.manipulators.examples")
    sim_app_.update()

    scene = build_task_scene()
    carb.log_info("[task_scene] Scene built successfully.")
    carb.log_info(f"  Table     : {scene['table'].GetPath()}")
    carb.log_info(f"  Cubes     : {[c.GetPath() for c in scene['cubes']]}")
    carb.log_info(f"  Target    : {scene['target'].GetPath()}")
    carb.log_info(f"  RGB Cam   : {scene['cameras']['rgb_prim'].GetPath()}")
    carb.log_info(f"  Depth Cam : {scene['cameras']['depth_prim'].GetPath()}")

    if args_.test:
        import time
        sim_app_.update()
        time.sleep(2)
        carb.log_info("[task_scene] Test mode — exiting.")
    else:
        carb.log_info("[task_scene] Running. Close the window or Ctrl+C to exit.")
        while sim_app_.is_running():
            sim_app_.update()

    sim_app_.close()


if __name__ == "__main__":
    _standalone_preview()
