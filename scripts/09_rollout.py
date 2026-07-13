#!/usr/bin/env python3
"""Day 9 — Model-Driven Rollout in Isaac Sim (file-IPC with model server).

Runs pick-and-place episodes using a trained Diffusion Policy.  Model
inference is delegated to ``scripts/09_model_server.py`` (runs in conda
vlarm) via temp-file IPC — no PyTorch needed in Isaac Sim's Python.

Start the model server FIRST (in another terminal)::

    conda activate vlarm
    PYTHONPATH=src python scripts/09_model_server.py

Then run the rollout (in Isaac Sim's Python)::

    ~/isaac-sim-standalone-6.0.1-linux-x86_64/python.sh scripts/09_rollout.py

Usage::

    ~/isaac-sim-standalone-6.0.1-linux-x86_64/python.sh scripts/09_rollout.py \\
        --episodes 10 --seed 100 --headless
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

# Parse args BEFORE importing SimulationApp
parser = argparse.ArgumentParser(description="VLARM — Day 9: Model Rollout")
parser.add_argument("--headless", action="store_true", help="Run headless (no GUI)")
parser.add_argument("--episodes", type=int, default=10,
                    help="Number of rollout episodes")
parser.add_argument("--max-steps", type=int, default=200,
                    help="Max steps per episode")
parser.add_argument("--seed", type=int, default=42, help="Base random seed")
parser.add_argument("--exec-horizon", type=int, default=8,
                    help="Actions to execute before re-predicting")
parser.add_argument("--work-dir", type=str, default="/tmp/vlarm_server",
                    help="IPC directory (must match model server)")
parser.add_argument("--no-gui-keep", action="store_true",
                    help="Do not keep GUI open after completion")
parser.add_argument("--no-model", action="store_true",
                    help="Run WITHOUT model (use zero actions) — for testing the sim side")
args, unknown = parser.parse_known_args()

_src = str(Path(__file__).resolve().parent.parent / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

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

from envs.isaac_env import IsaacEnv


# =============================================================================
# Rollout result
# =============================================================================


@dataclass
class RolloutResult:
    episode: int = 0
    seed: int = 0
    success: bool = False
    steps: int = 0
    final_reward: float = 0.0
    final_ee_pos: tuple = (0.0, 0.0, 0.0)
    cube_pos: tuple = (0.0, 0.0, 0.0)
    target_pos: tuple = (0.0, 0.0, 0.0)
    failure_reason: str = ""
    n_predictions: int = 0
    avg_inf_time_ms: float = 0.0


# =============================================================================
# Model client (file-IPC)
# =============================================================================


class ModelClient:
    """Communicate with the model server via temp files."""

    def __init__(self, work_dir: str = "/tmp/vlarm_server"):
        self._work_dir = Path(work_dir)
        self._request_file = self._work_dir / "request.npz"
        self._response_file = self._work_dir / "response.npy"
        self._ready_file = self._work_dir / "ready"
        self._info_file = self._work_dir / "info.json"
        self._timeout = 60.0  # seconds
        self._total_time = 0.0
        self._n_calls = 0

        # Check server is running
        if not self._info_file.exists():
            raise RuntimeError(
                f"Model server not found! Start it first:\n"
                f"  conda activate vlarm\n"
                f"  PYTHONPATH=src python scripts/09_model_server.py"
            )
        with open(self._info_file) as f:
            info = json.load(f)
        print(f"Model client connected: epoch={info['epoch']}, "
              f"val_loss={info['val_loss']:.4f}")

    def predict(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        """Send observation to model server, return predicted action trajectory.

        Parameters
        ----------
        obs : dict
            Keys: joint_positions (2, 9), ee_position (2, 3), gripper_width (2, 1)

        Returns
        -------
        action_traj : np.ndarray shape (1, 16, 7)
            Predicted arm joint trajectory.
        """
        # Clean up any stale files
        self._request_file.unlink(missing_ok=True)
        self._response_file.unlink(missing_ok=True)
        self._ready_file.unlink(missing_ok=True)

        # Write request
        np.savez_compressed(str(self._request_file), **obs)

        # Wait for response
        t_start = time.monotonic()
        while not self._ready_file.exists():
            if time.monotonic() - t_start > self._timeout:
                raise TimeoutError("Model server did not respond within timeout")
            time.sleep(0.1)

        # Read response
        action = np.load(str(self._response_file))

        # Clean up
        self._ready_file.unlink(missing_ok=True)
        self._response_file.unlink(missing_ok=True)

        elapsed = (time.monotonic() - t_start) * 1000
        self._total_time += elapsed
        self._n_calls += 1

        return action

    def no_model_predict(self) -> np.ndarray:
        """Return a zero-action trajectory (for testing without model)."""
        return np.zeros((1, 16, 7), dtype=np.float32)

    @property
    def avg_inference_time_ms(self) -> float:
        return self._total_time / max(1, self._n_calls)

    @property
    def n_calls(self) -> int:
        return self._n_calls


# =============================================================================
# Gripper heuristic
# =============================================================================


class GripperHeuristic:
    """Distance-based gripper control.

    - Close gripper when EE is near cube
    - Open when near target pad (after having been closed)
    """

    def __init__(self) -> None:
        self._closed = False

    def decide(self, ee_pos: np.ndarray, cube_pos: np.ndarray,
               target_pos: np.ndarray) -> bool:
        ee = ee_pos.flatten()
        cube = cube_pos.flatten()
        target = target_pos.flatten()

        close_xy = np.linalg.norm(ee[:2] - cube[:2])
        close_z = abs(ee[2] - cube[2])
        open_xy = np.linalg.norm(ee[:2] - target[:2])
        open_z = abs(ee[2] - target[2])

        if close_xy < 0.08 and close_z < 0.05:
            self._closed = True
        elif open_xy < 0.08 and open_z < 0.03 and self._closed:
            self._closed = False

        return self._closed

    def reset(self) -> None:
        self._closed = False


# =============================================================================
# Rollout runner
# =============================================================================


class ModelRollout:
    """Run closed-loop pick-and-place with a trained Diffusion Policy."""

    def __init__(self, env: IsaacEnv, client: ModelClient,
                 exec_horizon: int = 8, no_model: bool = False):
        self._env = env
        self._client = client
        self._exec_horizon = exec_horizon
        self._no_model = no_model
        self._gripper = GripperHeuristic()
        self._obs_buffer: list[dict[str, np.ndarray]] = []

    def run_episode(self, seed: int, episode_idx: int = 0) -> RolloutResult:
        result = RolloutResult(episode=episode_idx, seed=seed)
        self._gripper.reset()
        self._obs_buffer = []
        n_predictions = 0

        obs = self._env.reset(seed=seed)
        simulation_app.update()

        cube_pos = self._env._cube_positions_np[0]
        target_pos = np.array(
            self._env._scene["config"]["target_position"], dtype=np.float32
        )
        result.cube_pos = tuple(cube_pos.tolist())
        result.target_pos = tuple(target_pos.tolist())

        franka = self._env.franka
        self._obs_buffer.append(self._extract_state(obs))

        action_buffer: list[np.ndarray] = []

        for step in range(args.max_steps):
            # --- Get action ---
            if len(action_buffer) == 0:
                # Fill observation buffer
                while len(self._obs_buffer) < 2:
                    self._obs_buffer.insert(0, self._obs_buffer[0])

                # Build model input
                model_obs = self._prepare_model_obs()

                # Predict (via model server or zero)
                if self._no_model:
                    pred = self._client.no_model_predict()
                else:
                    pred = self._client.predict(model_obs)

                n_predictions += 1
                action_buffer = [pred[0, i] for i in range(pred.shape[1])]

            arm_action = action_buffer.pop(0)

            # --- Gripper ---
            ee_pos = obs["ee_position"].flatten()
            close = self._gripper.decide(ee_pos, cube_pos, target_pos)

            # --- Build & execute 9-DOF action ---
            full_action = np.zeros(9, dtype=np.float32)
            full_action[:7] = arm_action
            full_action[7:] = 0.0 if close else 0.04

            lo = IsaacEnv.JOINT_LIMITS_LOW[:7]
            hi = IsaacEnv.JOINT_LIMITS_HIGH[:7]
            full_action[:7] = np.clip(full_action[:7], lo, hi)

            try:
                franka.set_dof_position_targets(
                    full_action.reshape(1, -1).astype(np.float32),
                    dof_indices=list(range(9)),
                )
            except Exception:
                pass

            for _ in range(self._env._substeps):
                self._env._physics_step()

            obs = self._env._get_obs()
            self._env._step_count += 1

            self._obs_buffer.append(self._extract_state(obs))
            if len(self._obs_buffer) > 2:
                self._obs_buffer.pop(0)

            reward, terminated = self._env._compute_reward_and_done()
            result.steps = step + 1
            result.final_reward = float(reward)

            if terminated:
                result.success = True
                result.final_ee_pos = tuple(obs["ee_position"].flatten().tolist())
                break

        if not result.success:
            result.final_ee_pos = tuple(obs["ee_position"].flatten().tolist())
            ee = obs["ee_position"].flatten()
            dist_cube = np.linalg.norm(ee - cube_pos)
            dist_target = np.linalg.norm(ee - target_pos)
            dist_ct = np.linalg.norm(cube_pos - target_pos)
            if dist_ct < 0.05:
                result.success = True
            else:
                result.failure_reason = (
                    f"ee→cube={dist_cube:.2f}m, ee→target={dist_target:.2f}m, "
                    f"cube→target={dist_ct:.2f}m"
                )

        result.n_predictions = n_predictions
        result.avg_inf_time_ms = self._client.avg_inference_time_ms
        return result

    @staticmethod
    def _extract_state(obs: dict) -> dict:
        jp = np.asarray(obs["joint_positions"], dtype=np.float32).flatten()
        ee = np.asarray(obs["ee_position"], dtype=np.float32).flatten()
        gw = np.asarray(obs["gripper_width"], dtype=np.float32).flatten()
        # Pad or truncate joint_positions to 9 DOF
        if len(jp) < 9:
            jp = np.pad(jp, (0, 9 - len(jp)))
        elif len(jp) > 9:
            jp = jp[:9]
        # Ensure gripper_width is 1D
        if len(gw) == 0:
            gw = np.array([0.04], dtype=np.float32)
        elif len(gw) > 1:
            gw = gw[:1]
        return {
            "joint_positions": jp,
            "ee_position": ee,
            "gripper_width": gw,
        }

    def _prepare_model_obs(self) -> dict[str, np.ndarray]:
        result = {}
        for key in ["joint_positions", "ee_position", "gripper_width"]:
            frames = [f[key] for f in self._obs_buffer[-2:]]
            stacked = np.stack(frames, axis=0)  # (2, D)
            result[key] = stacked.astype(np.float32)
        return result


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    print("=" * 55)
    print("  VLARM — Day 9: Model Rollout in Isaac Sim")
    print("=" * 55)
    print(f"  Episodes:   {args.episodes}")
    print(f"  Exec horizon: {args.exec_horizon}")
    print(f"  Headless:   {args.headless}")
    print(f"  No model:   {args.no_model}")
    print()

    # --- Init environment ---
    print("Initializing Isaac Sim environment...")
    env = IsaacEnv(
        headless=args.headless,
        enable_ros2=False,
        simulation_app=simulation_app,
    )
    simulation_app.update()
    print("Environment ready.\n")

    # --- Init model client ---
    client = ModelClient(work_dir=args.work_dir)

    # --- Run episodes ---
    roller = ModelRollout(
        env=env, client=client,
        exec_horizon=args.exec_horizon,
        no_model=args.no_model,
    )

    results: list[RolloutResult] = []
    t_start = time.monotonic()

    for ep in range(args.episodes):
        seed = args.seed + ep
        print(f"\n{'─' * 45}")
        print(f"  Episode {ep + 1}/{args.episodes} (seed={seed})")
        print(f"{'─' * 45}")

        result = roller.run_episode(seed=seed, episode_idx=ep)
        results.append(result)

        status = "SUCCESS" if result.success else "FAILED"
        print(f"  {status} in {result.steps} steps, "
              f"{result.n_predictions} predictions")
        print(f"  Cube: ({result.cube_pos[0]:.3f}, "
              f"{result.cube_pos[1]:.3f}, {result.cube_pos[2]:.3f})")
        print(f"  Target: ({result.target_pos[0]:.3f}, "
              f"{result.target_pos[1]:.3f}, {result.target_pos[2]:.3f})")
        print(f"  Final EE: ({result.final_ee_pos[0]:.3f}, "
              f"{result.final_ee_pos[1]:.3f}, {result.final_ee_pos[2]:.3f})")
        if not result.success:
            print(f"  Reason: {result.failure_reason}")

    # --- Summary ---
    t_total = time.monotonic() - t_start
    success_count = sum(1 for r in results if r.success)
    success_rate = success_count / len(results) * 100 if results else 0
    avg_steps = np.mean([r.steps for r in results])

    print(f"\n{'=' * 55}")
    print(f"  Rollout Summary")
    print(f"{'=' * 55}")
    print(f"  Episodes:       {len(results)}")
    print(f"  Success rate:   {success_count}/{len(results)} ({success_rate:.0f}%)")
    print(f"  Avg steps:      {avg_steps:.1f}")
    print(f"  Avg inf time:   {client.avg_inference_time_ms:.0f} ms")
    print(f"  Total time:     {t_total:.0f}s")
    print(f"{'=' * 55}")

    if not args.headless and not args.no_gui_keep:
        print("\n  GUI will stay open for 10 seconds...")
        for _ in range(600):
            simulation_app.update()

    env.close()
    app_utils.stop()
    simulation_app.close()
    print("  Done.")


if __name__ == "__main__":
    main()
