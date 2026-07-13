#!/usr/bin/env python3
"""Day 5 — Expert Data Collection Script.

Runs the PickPlaceExpert policy in Isaac Sim and records episodes as HDF5
files for Diffusion Policy training.

The expert directly controls the Franka via ``set_end_effector_pose()``.
Physics stepping, observation collection, and reward computation are handled
manually to avoid ``IsaacEnv.step()`` overwriting the expert's IK targets.

Usage:
    # Quick test (5 episodes, headless)
    source /opt/ros/jazzy/setup.bash
    ~/isaac-sim-standalone-6.0.1-linux-x86_64/python.sh scripts/05_collect_expert_data.py --headless --episodes 5

    # Full collection (50 episodes, headless)
    source /opt/ros/jazzy/setup.bash
    ~/isaac-sim-standalone-6.0.1-linux-x86_64/python.sh scripts/05_collect_expert_data.py --headless --episodes 50

    # Clean old data first
    ~/isaac-sim-standalone-6.0.1-linux-x86_64/python.sh scripts/05_collect_expert_data.py --headless --episodes 50 --clean
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

# ---------------------------------------------------------------------------
# Parse args BEFORE importing SimulationApp
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Day 5 — Expert Data Collection")
parser.add_argument("--headless", action="store_true",
                    help="Run headless (no GUI)")
parser.add_argument("--episodes", type=int, default=50,
                    help="Number of episodes to collect (default: 50)")
parser.add_argument("--max-steps", type=int, default=300,
                    help="Maximum steps per episode (default: 300)")
parser.add_argument("--output-dir", type=str, default=None,
                    help="Override output directory (default: data/raw/)")
parser.add_argument("--clean", action="store_true",
                    help="Remove existing HDF5 files before starting")
parser.add_argument("--skip-validation", action="store_true",
                    help="Skip per-episode HDF5 validation (faster)")
args, unknown = parser.parse_known_args()

# ---------------------------------------------------------------------------
# Ensure project src/ is importable
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

app_utils.enable_extension("isaacsim.ros2.bridge")
app_utils.enable_extension("isaacsim.robot.experimental.manipulators.examples")
simulation_app.update()

import numpy as np

from envs.expert_policy import PickPlaceExpert
from envs.isaac_env import IsaacEnv
from vl_data.recorder import EpisodeRecorder, NullCameraSource, validate_episode


# ===================================================================
# Main
# ===================================================================
def main() -> None:
    output_dir = args.output_dir or str(
        Path(__file__).resolve().parent.parent / "data" / "raw"
    )

    print("=" * 60)
    print("  VLARM — Day 5: Expert Data Collection")
    print("=" * 60)
    print(f"  Mode:       {'Headless' if args.headless else 'GUI'}")
    print(f"  Episodes:   {args.episodes}")
    print(f"  Max steps:  {args.max_steps}")
    print(f"  Output:     {output_dir}")
    print()

    # --- Clean old data ---
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    if args.clean:
        removed = 0
        for f in out_path.glob("episode_*.h5"):
            f.unlink()
            removed += 1
        print(f"  Cleaned {removed} existing episode files.\n")

    # --- Set up env + recorder ---
    env = None
    recorder = None

    try:
        carb.log_info("[Day 5] Creating environment...")
        env = IsaacEnv(
            headless=args.headless,
            enable_ros2=False,
            simulation_app=simulation_app,
        )
        simulation_app.update()

        camera = NullCameraSource()
        recorder = EpisodeRecorder(
            save_dir=output_dir,
            camera_source=camera,
            flush_every=50,
        )

        # Statistics
        attempted = 0
        collected = 0
        successful = 0
        skipped = 0
        total_steps = 0
        step_counts: list[int] = []

        episode_id = 0
        seed = 100
        t_start = time.monotonic()

        franka = env.franka

        while collected < args.episodes:
            attempted += 1
            ep_start = time.monotonic()

            # --- Reset environment ---
            obs = env.reset(seed=seed)
            simulation_app.update()

            # --- Skip if any cube already on target ---
            cube_positions = env._cube_positions_np
            target_center = np.array(
                env._scene["config"]["target_position"], dtype=np.float32
            )
            target_radius = env._scene["config"]["target_radius"]

            cube_to_target = np.linalg.norm(
                cube_positions - target_center.reshape(1, 3), axis=1
            )
            if np.any(cube_to_target < target_radius):
                carb.log_info(
                    f"[Day 5] Episode {episode_id} skipped — "
                    f"cube already on target ({seed=})"
                )
                skipped += 1
                seed += 1
                continue

            # --- Start recording ---
            ep_num = recorder.start_episode(env)

            # --- Create expert & inform it about the scene ---
            expert = PickPlaceExpert()
            expert.set_scene_info(cube_positions, target_center)

            ep_steps = 0
            ep_success = False

            for step in range(args.max_steps):
                # 1. Expert sets EE + gripper targets on the Franka
                expert.act(obs, franka)

                # 2. Read the joint position targets that were set
                try:
                    action = franka.get_dof_position_targets().numpy().flatten().astype(
                        np.float32
                    )
                except AssertionError:
                    # Physics tensor invalid — fall back to obs joint positions
                    action = obs["joint_positions"].copy()

                # 3. Step physics (bypass env.step to avoid overwriting targets)
                for _ in range(env._substeps):
                    env._physics_step()

                # 4. Get new observation
                obs = env._get_obs()
                env._step_count += 1

                # 5. Compute reward and termination
                reward, terminated = env._compute_reward_and_done()
                truncated = step + 1 >= args.max_steps

                # 6. Record the step
                recorder.record_step(
                    obs=obs, action=action,
                    reward=reward, terminated=terminated, truncated=truncated,
                )

                ep_steps = step + 1

                # 7. Check stop conditions
                if expert.is_done:
                    # Expert completed all phases successfully
                    ep_success = True
                    carb.log_info(
                        f"[Day 5] Episode {ep_num} COMPLETE at step {step} "
                        f"(expert finished all phases)"
                    )
                    break

                if terminated:
                    ep_success = True
                    carb.log_info(
                        f"[Day 5] Episode {ep_num} SUCCESS at step {step} "
                        f"(cube on target, reward={reward:.1f})"
                    )
                    break

                if truncated:
                    carb.log_info(
                        f"[Day 5] Episode {ep_num} TRUNCATED at step {step}"
                    )
                    break

            # --- Finalise episode ---
            file_path = recorder.end_episode(success=ep_success)
            ep_elapsed = time.monotonic() - ep_start

            # --- Validate ---
            validation_ok = True
            if not args.skip_validation:
                result = validate_episode(file_path)
                if not result["valid"]:
                    carb.log_warn(
                        f"[Day 5] Validation issues in episode {ep_num}: "
                        f"{result['issues']}"
                    )
                    validation_ok = False

            if validation_ok:
                collected += 1
                step_counts.append(ep_steps)
                total_steps += ep_steps
                if ep_success:
                    successful += 1

            carb.log_info(
                f"[Day 5] Episode {ep_num}: {ep_steps} steps, "
                f"success={ep_success}, valid={validation_ok}, "
                f"time={ep_elapsed:.1f}s "
                f"({collected}/{args.episodes} collected)"
            )

            episode_id += 1
            seed += 1

        # --- Summary ---
        t_total = time.monotonic() - t_start
        print()
        print("=" * 60)
        print("  Collection Complete")
        print("=" * 60)
        print(f"  Episodes attempted:  {attempted}")
        print(f"  Episodes collected:  {collected}")
        print(f"  Episodes skipped:    {skipped}")
        print(f"  Successful pick-places: {successful}/{collected}")
        if collected > 0:
            rate = successful / collected * 100
            print(f"  Success rate:        {rate:.1f}%")
            print(f"  Avg steps/episode:   {total_steps / collected:.1f}")
            if step_counts:
                print(f"  Min steps:           {min(step_counts)}")
                print(f"  Max steps:           {max(step_counts)}")
        print(f"  Total time:          {t_total:.1f}s"
              + (f" ({t_total / collected:.1f}s/ep)" if collected > 0 else ""))
        print(f"  Output directory:    {output_dir}")

        # List files
        h5_files = sorted(out_path.glob("episode_*.h5"))
        total_size = sum(f.stat().st_size for f in h5_files)
        print(f"  HDF5 files:          {len(h5_files)} "
              f"({total_size / 1024**2:.1f} MB)")
        print()
        print("  Day 5 data collection complete!")
        print(f"  Next: Day 6 — Diffusion Policy model training")

    except Exception as e:
        carb.log_error(f"[Day 5] Fatal error: {e}")
        traceback.print_exc()
        sys.exit(1)

    finally:
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


if __name__ == "__main__":
    main()
