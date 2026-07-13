#!/usr/bin/env python3
"""Day 3 Integration Test — Task Scene + Environment.

Verifies:
  1. Task scene builds correctly (table, cubes, basket, cameras)
  2. IsaacEnv can be instantiated with Franka robot
  3. reset() returns valid observations
  4. step() advances simulation and returns (obs, reward, done, info)
  5. Multiple episodes work correctly
  6. Scene objects exist in USD stage
  7. Robot responds to position commands

Usage:
    # Quick headless smoke test (2 episodes, 10 steps each)
    source /opt/ros/jazzy/setup.bash
    ~/isaac-sim-standalone-6.0.1-linux-x86_64/python.sh scripts/03_test_task_scene.py --headless

    # GUI mode for visual inspection
    source /opt/ros/jazzy/setup.bash
    ~/isaac-sim-standalone-6.0.1-linux-x86_64/python.sh scripts/03_test_task_scene.py

Exit code: 0 = all checks passed, 1 = issues found.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

# ---------------------------------------------------------------------------
# Parse args BEFORE importing SimulationApp
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Day 3 — Task Scene + Env Test")
parser.add_argument("--headless", action="store_true",
                    help="Run headless (no GUI)")
parser.add_argument("--episodes", type=int, default=2,
                    help="Number of episodes to test (default: 2)")
parser.add_argument("--steps", type=int, default=15,
                    help="Steps per episode (default: 15)")
args, unknown = parser.parse_known_args()

# ---------------------------------------------------------------------------
# Start Isaac Sim
# ---------------------------------------------------------------------------
from isaacsim import SimulationApp

simulation_app = SimulationApp({
    "renderer": "RayTracedLighting",
    "headless": args.headless,
})

import carb
import isaacsim.core.experimental.utils.app as app_utils
import numpy as np

# Enable extensions
app_utils.enable_extension("isaacsim.ros2.bridge")
app_utils.enable_extension("isaacsim.robot.experimental.manipulators.examples")
simulation_app.update()

# Now import our modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from envs.task_scene import build_task_scene, DEFAULT_CUBE_POSITIONS
from envs.isaac_env import IsaacEnv


# ===================================================================
# Helpers
# ===================================================================
CHECK_PASSED = 0
CHECK_FAILED = 0


def ok(msg: str) -> None:
    global CHECK_PASSED
    CHECK_PASSED += 1
    print(f"  ✅  {msg}")


def fail(msg: str) -> None:
    global CHECK_FAILED
    CHECK_FAILED += 1
    print(f"  ❌  {msg}")


def section(title: str) -> None:
    print(f"\n{'=' * 55}")
    print(f"  {title}")
    print(f"{'=' * 55}")


# ===================================================================
# Tests
# ===================================================================
def check_scene_objects() -> None:
    """Verify all scene prims exist in the USD stage."""
    section("1. Scene Objects in USD Stage")

    import omni
    stage = omni.usd.get_context().get_stage()

    expected_prims = [
        "/World/GroundPlane",
        "/World/DistantLight",
        "/World/Table",
        "/World/TargetPad",
        "/World/CameraRGB",
        "/World/CameraDepth",
        "/World/Cube0",
        "/World/Cube1",
        "/World/Cube2",
    ]

    for prim_path in expected_prims:
        prim = stage.GetPrimAtPath(prim_path)
        if prim.IsValid():
            ok(f"Prim exists: {prim_path}")
        else:
            fail(f"Prim MISSING: {prim_path}")


def check_env_instantiation() -> IsaacEnv:
    """Create the env and verify it initializes."""
    section("2. IsaacEnv Instantiation")

    try:
        env = IsaacEnv(
            headless=args.headless,
            enable_ros2=False,
            simulation_app=simulation_app,
        )
        ok("IsaacEnv created successfully")
        return env
    except Exception as e:
        fail(f"IsaacEnv creation failed: {e}")
        traceback.print_exc()
        raise


def check_reset(env: IsaacEnv) -> dict:
    """Verify reset() returns valid observations."""
    section("3. Environment Reset")

    try:
        obs = env.reset(seed=42)
    except Exception as e:
        fail(f"reset() failed: {e}")
        traceback.print_exc()
        raise

    # Check observation keys
    required_keys = [
        "joint_positions", "joint_velocities",
        "ee_position", "ee_orientation", "gripper_width",
        "rgb", "depth",
    ]
    for key in required_keys:
        if key in obs:
            ok(f"obs['{key}'] present — shape {obs[key].shape}, dtype {obs[key].dtype}")
        else:
            fail(f"obs['{key}'] MISSING")

    # Check joint positions shape and range
    jp = obs["joint_positions"]
    if jp.shape == (9,) and jp.dtype == np.float32:
        ok(f"joint_positions shape={jp.shape} dtype={jp.dtype}")
    else:
        fail(f"joint_positions wrong shape/dtype: {jp.shape}/{jp.dtype}")

    # Check EE position
    ee = obs["ee_position"]
    if ee.shape == (3,) and np.all(np.isfinite(ee)):
        ok(f"ee_position = ({ee[0]:.3f}, {ee[1]:.3f}, {ee[2]:.3f})")
    else:
        fail(f"ee_position invalid: {ee}")

    # Check home position is ~reached (allow some tolerance)
    home = env.HOME_POSITION
    joint_diff = np.abs(jp - home)
    max_diff = float(np.max(joint_diff))
    if max_diff < 0.5:  # within ~30 deg
        ok(f"Robot near home position (max joint diff={max_diff:.3f} rad)")
    else:
        fail(f"Robot far from home (max joint diff={max_diff:.3f} rad)")

    return obs


def check_step(env: IsaacEnv, initial_obs: dict) -> None:
    """Verify step() returns correct tuple format."""
    section("4. Environment Step")

    # Simple action: move arm slightly from current position
    action = initial_obs["joint_positions"][:7].copy()
    action[0] += 0.1  # rotate joint 1 by ~6 deg
    action[4] -= 0.1  # rotate joint 5

    try:
        result = env.step(action)
    except Exception as e:
        fail(f"step() failed: {e}")
        traceback.print_exc()
        return

    # Check return format: (obs, reward, terminated, truncated, info)
    if not isinstance(result, tuple) or len(result) != 5:
        fail(f"step() should return 5-tuple, got {type(result)} len={len(result)}")
        return
    ok("step() returns 5-tuple (obs, reward, terminated, truncated, info)")

    obs, reward, terminated, truncated, info = result

    # Check types
    if isinstance(reward, float):
        ok(f"reward = {reward:.4f} (float)")
    else:
        fail(f"reward should be float, got {type(reward)}")

    if isinstance(terminated, bool) and isinstance(truncated, bool):
        ok(f"terminated={terminated}, truncated={truncated}")
    else:
        fail(f"terminated/truncated should be bool")

    if isinstance(info, dict) and "step" in info:
        ok(f"info step={info['step']}, episode={info.get('episode', '?')}")
    else:
        fail("info missing 'step' key")

    # Check that joint positions have changed from initial
    jp = obs["joint_positions"]
    initial_jp = initial_obs["joint_positions"]
    diff = np.linalg.norm(jp[:7] - initial_jp[:7])
    if diff > 0.001:  # should have moved
        ok(f"Robot moved (joint delta norm={diff:.4f})")
    else:
        fail(f"Robot did NOT move (joint delta norm={diff:.6f})")


def check_multiple_episodes(env: IsaacEnv) -> None:
    """Run multiple reset/step cycles."""
    section("5. Multiple Episodes")

    total_steps = 0
    for ep in range(args.episodes):
        obs = env.reset(seed=100 + ep)
        ep_steps = 0
        for st in range(args.steps):
            # Simple action: gentle noise around home
            action = env.HOME_POSITION[:7].copy()
            action += np.random.normal(0, 0.05, size=7).astype(np.float32)

            obs, reward, terminated, truncated, info = env.step(action)
            ep_steps = info["step"]

            if terminated or truncated:
                break

        total_steps += ep_steps
        status = "✅" if info.get("success") else ("⏰" if truncated else "  ")
        print(f"  {status}  Episode {ep}: {ep_steps} steps, "
              f"final reward={reward:.3f}, success={info['success']}")

    ok(f"Completed {args.episodes} episodes, {total_steps} total steps")


def check_robot_properties(env: IsaacEnv) -> None:
    """Verify robot property accessors."""
    section("6. Robot Properties")

    try:
        jp = env.joint_positions
        if jp.shape == (9,):
            ok(f"env.joint_positions shape={jp.shape}")
        else:
            fail(f"joint_positions shape={jp.shape}")

        ee = env.ee_position
        if ee.shape == (3,):
            ok(f"env.ee_position = ({ee[0]:.3f}, {ee[1]:.3f}, {ee[2]:.3f})")
        else:
            fail(f"ee_position shape={ee.shape}")

        eeq = env.ee_orientation
        if eeq.shape == (4,):
            ok(f"env.ee_orientation = (w={eeq[0]:.3f}, x={eeq[1]:.3f}, "
               f"y={eeq[2]:.3f}, z={eeq[3]:.3f})")
        else:
            fail(f"ee_orientation shape={eeq.shape}")

        franka = env.franka
        if franka is not None:
            ok(f"Franka robot accessible: {type(franka).__name__}")
        else:
            fail("Franka robot is None")

        scene = env.scene
        if isinstance(scene, dict) and "table" in scene and "target" in scene:
            ok(f"Scene dict accessible with {len(scene)} keys")
        else:
            fail(f"Scene dict invalid: {type(scene)}")
    except Exception as e:
        fail(f"Property access failed: {e}")
        traceback.print_exc()


def check_cleanup(env: IsaacEnv) -> None:
    """Verify close() works without errors."""
    section("7. Cleanup")
    try:
        env.close()
        ok("env.close() completed without errors")
    except Exception as e:
        fail(f"env.close() failed: {e}")


# ===================================================================
# Main
# ===================================================================
def main() -> None:
    global CHECK_PASSED, CHECK_FAILED

    print("=" * 55)
    print("  VLARM — Day 3: Task Scene + Environment Test")
    print("=" * 55)
    print(f"  Mode:      {'Headless' if args.headless else 'GUI'}")
    print(f"  Episodes:  {args.episodes}")
    print(f"  Steps:     {args.steps}")
    print()

    # --- Create env first (this builds the scene internally) ---
    carb.log_info("[Day 3 Test] Creating environment (builds scene + loads Franka)...")

    env = None
    try:
        env = check_env_instantiation()
        simulation_app.update()

        check_scene_objects()

        obs = check_reset(env)
        simulation_app.update()

        check_step(env, obs)
        simulation_app.update()

        check_robot_properties(env)
        check_multiple_episodes(env)
        check_cleanup(env)

    except Exception as e:
        fail(f"FATAL: {e}")
        traceback.print_exc()

    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass

        app_utils.stop()
        simulation_app.close()

    # --- Summary ---
    section("Summary")
    total = CHECK_PASSED + CHECK_FAILED
    print(f"  Checks: {CHECK_PASSED} passed / {total} total")
    if CHECK_FAILED == 0:
        print("  ✅  ALL CHECKS PASSED")
        print()
        print("  Day 3 task scene and environment are ready.")
        print("  Next: Day 4 — Data collection pipeline")
        sys.exit(0)
    else:
        print(f"  ❌  {CHECK_FAILED} CHECK(S) FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
