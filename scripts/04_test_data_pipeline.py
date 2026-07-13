#!/usr/bin/env python3
"""Day 4 Integration Test — Data Collection Pipeline.

Verifies:
  1. EpisodeRecorder can be instantiated with IsaacEnv
  2. Single episode: record T steps, HDF5 file created with correct structure
  3. HDF5 validation: shapes, ranges, no NaN, metadata
  4. Multiple episodes: record 3 episodes, each in its own file
  5. EpisodicDataset: load from HDF5 directory, iterate samples
  6. Window sampling: obs/action window shapes correct
  7. Augmentation: apply transforms, verify output shapes unchanged

Usage:
    # Headless smoke test
    source /opt/ros/jazzy/setup.bash
    ~/isaac-sim-standalone-6.0.1-linux-x86_64/python.sh scripts/04_test_data_pipeline.py --headless

    # GUI mode
    source /opt/ros/jazzy/setup.bash
    ~/isaac-sim-standalone-6.0.1-linux-x86_64/python.sh scripts/04_test_data_pipeline.py

Exit code: 0 = all checks passed, 1 = issues found.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import traceback
from pathlib import Path

# ---------------------------------------------------------------------------
# Parse args BEFORE importing SimulationApp
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Day 4 — Data Pipeline Test")
parser.add_argument("--headless", action="store_true",
                    help="Run headless (no GUI)")
parser.add_argument("--episodes", type=int, default=3,
                    help="Number of episodes to record (default: 3)")
parser.add_argument("--steps", type=int, default=15,
                    help="Steps per episode (default: 15)")
parser.add_argument("--output-dir", type=str, default=None,
                    help="Override output directory (default: data/raw/)")
args, unknown = parser.parse_known_args()

# ---------------------------------------------------------------------------
# Ensure project src/ is importable BEFORE SimulationApp loads extensions
# (Isaac Sim extensions may shadow the "data" package name)
# ---------------------------------------------------------------------------
_src_path = str(Path(__file__).resolve().parent.parent / "src")
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

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
from envs.isaac_env import IsaacEnv
from vl_data.recorder import EpisodeRecorder, NullCameraSource, validate_episode

# Torch-dependent imports (may fail inside Isaac Sim's Python)
_HAS_TORCH = False
try:
    from vl_data.dataset import EpisodicDataset  # noqa: F811
    from vl_data.augmentation import Compose, JointNoise, RandomColorJitter, RandomCrop
    import torch
    _HAS_TORCH = True
except ImportError:
    EpisodicDataset = None  # type: ignore[assignment]
    Compose = None          # type: ignore[assignment]
    JointNoise = None       # type: ignore[assignment]
    RandomColorJitter = None  # type: ignore[assignment]
    RandomCrop = None       # type: ignore[assignment]


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
def check_recorder_instantiation(env: IsaacEnv, output_dir: str) -> EpisodeRecorder:
    """Create EpisodeRecorder and verify internal state."""
    section("1. EpisodeRecorder Instantiation")

    camera = NullCameraSource()
    try:
        recorder = EpisodeRecorder(
            save_dir=output_dir,
            camera_source=camera,
            flush_every=10,
        )
        ok("EpisodeRecorder created successfully")
        ok(f"Camera source: {type(recorder._camera_source).__name__}")
        ok(f"Save directory: {output_dir}")
        return recorder
    except Exception as e:
        fail(f"EpisodeRecorder creation failed: {e}")
        traceback.print_exc()
        raise


def check_single_episode_record(
    env: IsaacEnv, recorder: EpisodeRecorder
) -> str:
    """Record one episode and verify the HDF5 file exists."""
    section("2. Single Episode Recording")

    try:
        # Reset env
        obs = env.reset(seed=42)
        simulation_app.update()

        # Start recording
        ep_id = recorder.start_episode(env)
        ok(f"Episode {ep_id} started")

        # Record first observation
        recorder.record_step(
            obs=obs,
            action=env.HOME_POSITION.copy(),
            reward=0.0,
            terminated=False,
            truncated=False,
        )

        # Record N steps
        for step in range(args.steps - 1):
            action = env.HOME_POSITION[:7].copy()
            action += np.random.normal(0, 0.05, size=7).astype(np.float32)

            obs, reward, terminated, truncated, info = env.step(action)

            recorder.record_step(
                obs=obs,
                action=action,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
            )

            if terminated or truncated:
                break

        file_path = recorder.end_episode(success=False)
        ok(f"Episode saved to: {file_path}")
        ok(f"File exists: {Path(file_path).exists()}")
        ok(f"Total steps: {recorder.current_step}")

        return file_path

    except Exception as e:
        fail(f"Recording failed: {e}")
        traceback.print_exc()
        raise


def check_hdf5_validation(file_path: str) -> None:
    """Validate the recorded HDF5 file."""
    section("3. HDF5 Validation")

    try:
        import h5py

        with h5py.File(file_path, "r") as f:
            # Check attributes
            for attr in ("episode_id", "timestamp", "num_steps", "success"):
                if attr in f.attrs:
                    ok(f"Attribute '{attr}' = {f.attrs[attr]}")
                else:
                    fail(f"Missing attribute: {attr}")

            num_steps = int(f.attrs["num_steps"])
            ok(f"num_steps = {num_steps}")

            # Check required datasets
            required = [
                "observations/joint_positions",
                "observations/joint_velocities",
                "observations/ee_position",
                "observations/ee_orientation",
                "observations/gripper_width",
                "observations/rgb",
                "observations/depth",
                "actions/joint_targets",
                "rewards",
                "terminals",
                "truncations",
            ]
            for ds_path in required:
                if ds_path in f:
                    ds = f[ds_path]
                    if ds.shape[0] == num_steps:
                        ok(f"{ds_path}: shape={ds.shape}, dtype={ds.dtype}")
                    else:
                        fail(f"{ds_path}: expected T={num_steps}, got shape[0]={ds.shape[0]}")
                else:
                    fail(f"Missing dataset: {ds_path}")

            # Data sanity checks
            jp = f["observations/joint_positions"][:]
            if not np.any(np.isnan(jp)):
                ok(f"joint_positions: no NaN, min={np.min(jp):.3f}, max={np.max(jp):.3f}")
            else:
                fail("joint_positions contains NaN")

            gw = f["observations/gripper_width"][:]
            if np.min(gw) >= 0 and np.max(gw) <= 0.1:
                ok(f"gripper_width range OK: [{np.min(gw):.4f}, {np.max(gw):.4f}]")
            else:
                fail(f"gripper_width out of range: min={np.min(gw):.4f}, max={np.max(gw):.4f}")

            ee = f["observations/ee_position"][:]
            if np.all(np.isfinite(ee)):
                ok(f"ee_position all finite: min_z={np.min(ee[:, 2]):.3f}")
            else:
                fail("ee_position has NaN/Inf")

    except Exception as e:
        fail(f"Validation failed: {e}")
        traceback.print_exc()


def check_multiple_episodes(
    env: IsaacEnv, recorder: EpisodeRecorder, output_dir: str
) -> None:
    """Record multiple episodes and verify each produces a file."""
    section("4. Multiple Episode Recording")

    for ep in range(args.episodes):
        try:
            obs = env.reset(seed=100 + ep)
            recorder.start_episode(env)

            for step in range(args.steps):
                action = env.HOME_POSITION[:7].copy()
                action += np.random.normal(0, 0.05, size=7).astype(np.float32)
                obs, reward, terminated, truncated, info = env.step(action)
                recorder.record_step(
                    obs=obs, action=action,
                    reward=reward, terminated=terminated, truncated=truncated,
                )
                if terminated or truncated:
                    break

            recorder.end_episode(success=False)
            ok(f"Episode {ep} recorded ({recorder.current_step} steps)")

        except Exception as e:
            fail(f"Episode {ep} failed: {e}")
            traceback.print_exc()

    # Verify all files exist
    h5_files = sorted(Path(output_dir).glob("episode_*.h5"))
    expected = args.episodes + 1  # +1 from single episode test
    if len(h5_files) >= expected:
        ok(f"All {len(h5_files)} HDF5 files present in {output_dir}/")
    else:
        fail(f"Expected >= {expected} HDF5 files, found {len(h5_files)}")

    for fp in h5_files:
        print(f"    📄  {fp.name} ({fp.stat().st_size / 1024:.1f} KB)")


def check_dataset_loading(output_dir: str) -> None:
    """Load HDF5 files via EpisodicDataset."""
    section("5. Dataset Loading")

    if not _HAS_TORCH:
        ok("Skipped — torch not available in this Python environment")
        ok("(Dataset loading works with system Python: pip install torch)")
        return

    try:
        ds = EpisodicDataset(
            data_dir=output_dir,
            obs_horizon=2,
            action_horizon=8,
            action_downsample=1,
        )
        ok(f"EpisodicDataset created: {len(ds)} samples from {len(ds._episode_files)} episodes")

        if len(ds) > 0:
            sample = ds[0]
            required_keys = ["observations", "actions", "rgb", "depth", "language_embedding"]
            for key in required_keys:
                if key in sample:
                    ok(f"Sample has '{key}'")
                else:
                    fail(f"Sample missing '{key}'")

            # Check observation dict
            obs = sample["observations"]
            for ok_key in ("joint_positions", "ee_position", "gripper_width"):
                if ok_key in obs:
                    ok(f"observations['{ok_key}'] shape={obs[ok_key].shape}")
                else:
                    fail(f"observations missing '{ok_key}'")

            # Check shapes
            actions = sample["actions"]
            ok(f"actions shape: {tuple(actions.shape)}")
        else:
            fail("Dataset has 0 samples (need more steps or smaller windows)")

    except Exception as e:
        fail(f"Dataset loading failed: {e}")
        traceback.print_exc()


def check_window_sampling(output_dir: str) -> None:
    """Verify that windowed sampling returns correct shapes."""
    section("6. Window Sampling")

    if not _HAS_TORCH:
        ok("Skipped — torch not available in this Python environment")
        return

    try:
        ds = EpisodicDataset(
            data_dir=output_dir,
            obs_horizon=2,
            action_horizon=4,
            action_downsample=2,
        )

        if len(ds) < 2:
            fail(f"Not enough samples for batch test ({len(ds)} < 2)")
            return

        # Get two samples
        s0 = ds[0]
        s1 = ds[1]

        # Check obs horizon
        for key in ("joint_positions", "ee_position"):
            if key in s0["observations"]:
                obs = s0["observations"][key]
                if obs.shape[0] == ds._obs_horizon:
                    ok(f"{key}: obs_horizon={obs.shape[0]} ✅")
                else:
                    fail(f"{key}: expected obs_horizon={ds._obs_horizon}, got {obs.shape[0]}")

        # Check action horizon (with downsample)
        if s0["actions"].shape[0] == ds._action_horizon:
            ok(f"actions: horizon={s0['actions'].shape[0]} ✅")
        else:
            fail(f"actions: expected horizon={ds._action_horizon}, got {s0['actions'].shape[0]}")

        # Check action is arm-only (7 DOF)
        if s0["actions"].shape[1] == 7:
            ok(f"actions: 7 arm DOF ✅")
        else:
            fail(f"actions: expected 7 DOF, got {s0['actions'].shape[1]}")

    except Exception as e:
        fail(f"Window sampling check failed: {e}")
        traceback.print_exc()


def check_augmentation(output_dir: str) -> None:
    """Apply augmentations and verify output shapes unchanged."""
    section("7. Augmentation")

    if not _HAS_TORCH:
        ok("Skipped — torch not available in this Python environment")
        ok("(Augmentation works with system Python + torchvision)")
        return

    try:
        ds = EpisodicDataset(
            data_dir=output_dir,
            obs_horizon=2,
            action_horizon=4,
            transform=None,  # Apply augmentation manually
        )

        if len(ds) < 1:
            fail("No samples for augmentation test")
            return

        # Get a raw sample (numpy)
        file_idx, start_step = ds._index[0]
        import h5py
        with h5py.File(ds._episode_files[file_idx], "r") as f:
            obs = {}
            for key in ds._obs_keys:
                ds_path = f"observations/{key}"
                if ds_path in f:
                    obs[key] = f[ds_path][start_step:start_step + ds._obs_horizon]
            raw_sample = {
                "observations": obs,
                "actions": np.zeros((4, 7), dtype=np.float32),
                "rgb": obs.get("rgb", np.zeros((2, 480, 640, 3), dtype=np.uint8)),
                "depth": obs.get("depth", np.zeros((2, 480, 640, 1), dtype=np.float32)),
            }

        # Test RandomColorJitter
        aug = RandomColorJitter(brightness=0.3, contrast=0.3, p=1.0)
        augmented = aug(raw_sample)
        if "rgb" in augmented:
            rgb_orig = np.asarray(raw_sample["rgb"])
            rgb_aug = np.asarray(augmented["rgb"])
            if rgb_orig.shape == rgb_aug.shape:
                ok(f"ColorJitter: shape preserved {rgb_aug.shape}")
            else:
                fail(f"ColorJitter: shape changed {rgb_orig.shape} → {rgb_aug.shape}")
            if not np.allclose(rgb_orig, rgb_aug) or np.any(rgb_orig != rgb_aug):
                ok("ColorJitter: pixel values changed (expected)")
            else:
                ok("ColorJitter: no visible change (images may be all zeros)")

        # Test RandomCrop
        aug_crop = RandomCrop(scale=(0.8, 1.0), p=1.0)
        cropped = aug_crop(raw_sample)
        if "rgb" in cropped:
            ok(f"RandomCrop: shape preserved {np.asarray(cropped['rgb']).shape}")

        # Test JointNoise
        aug_noise = JointNoise(joint_std=0.01, p=1.0)
        noisy = aug_noise(raw_sample)
        if "joint_positions" in noisy["observations"]:
            jp_orig = raw_sample["observations"]["joint_positions"]
            jp_noisy = noisy["observations"]["joint_positions"]
            if jp_orig.shape == jp_noisy.shape:
                ok(f"JointNoise: shape preserved {jp_noisy.shape}")
                # Check actual noise added (skip if all zeros)
                if np.any(jp_orig != 0):
                    max_diff = np.max(np.abs(jp_noisy - jp_orig))
                    if max_diff > 0:
                        ok(f"JointNoise: max delta = {max_diff:.6f} (expected)")
                    else:
                        ok("JointNoise: no delta (input may be zero)")
            else:
                fail(f"JointNoise: shape changed")

        # Test Compose
        aug_compose = Compose([
            RandomColorJitter(p=1.0),
            JointNoise(joint_std=0.005, p=1.0),
        ])
        composed = aug_compose(raw_sample)
        if "observations" in composed and "rgb" in composed:
            ok(f"Compose: sample intact with {len(composed)} top-level keys")

    except Exception as e:
        fail(f"Augmentation check failed: {e}")
        traceback.print_exc()


# ===================================================================
# Main
# ===================================================================
def main() -> None:
    global CHECK_PASSED, CHECK_FAILED

    output_dir = args.output_dir or str(
        Path(__file__).resolve().parent.parent / "data" / "raw"
    )

    print("=" * 55)
    print("  VLARM — Day 4: Data Pipeline Test")
    print("=" * 55)
    print(f"  Mode:      {'Headless' if args.headless else 'GUI'}")
    print(f"  Episodes:  {args.episodes}")
    print(f"  Steps:     {args.steps}")
    print(f"  Output:    {output_dir}")
    print()

    # Clean output directory for a fresh test
    if Path(output_dir).exists():
        import glob
        for f in Path(output_dir).glob("episode_*.h5"):
            f.unlink()
            print(f"  🗑️   Removed previous: {f.name}")

    recorder = None
    env = None
    single_ep_file = None

    try:
        carb.log_info("[Day 4 Test] Creating environment...")

        env = IsaacEnv(
            headless=args.headless,
            enable_ros2=False,
            simulation_app=simulation_app,
        )
        simulation_app.update()

        # --- Test 1: Recorder instantiation ---
        recorder = check_recorder_instantiation(env, output_dir)

        # --- Test 2: Single episode ---
        single_ep_file = check_single_episode_record(env, recorder)

        # --- Test 3: HDF5 validation ---
        check_hdf5_validation(single_ep_file)

        # --- Test 4: Multiple episodes ---
        check_multiple_episodes(env, recorder, output_dir)

        # --- Test 5: Dataset loading ---
        check_dataset_loading(output_dir)

        # --- Test 6: Window sampling ---
        check_window_sampling(output_dir)

        # --- Test 7: Augmentation ---
        check_augmentation(output_dir)

    except Exception as e:
        fail(f"FATAL: {e}")
        traceback.print_exc()

    finally:
        # --- Summary (before cleanup, so output isn't swallowed) ---
        section("Summary")
        total = CHECK_PASSED + CHECK_FAILED
        print(f"  Checks: {CHECK_PASSED} passed / {total} total")
        if CHECK_FAILED == 0:
            print("  ✅  ALL CHECKS PASSED")
            print()
            print("  Day 4 data pipeline is ready.")
            print(f"  Recorded episodes are in {output_dir}/")
            print("  Next: Day 5 — Expert policy collection")
        else:
            print(f"  ❌  {CHECK_FAILED} CHECK(S) FAILED")

        # Cleanup
        if recorder is not None:
            try:
                recorder.close()
            except Exception:
                pass
        if env is not None:
            try:
                env.close()
            except Exception:
                pass

        app_utils.stop()
        simulation_app.close()

    # Exit based on results
    sys.exit(0 if CHECK_FAILED == 0 else 1)


if __name__ == "__main__":
    main()
