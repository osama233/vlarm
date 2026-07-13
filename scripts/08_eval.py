#!/usr/bin/env python3
"""Day 8 — Model Evaluation & Rollout Script.

Evaluates a trained Diffusion Policy checkpoint on held-out validation data:
  1. DDIM inference: predict action trajectories from observations
  2. Compare predicted actions vs ground-truth expert actions
  3. Per-joint MSE breakdown
  4. Trajectory smoothness analysis
  5. Optional: live Isaac Sim rollout (Day 9 integration point)

Usage::

    conda activate vlarm
    # Quick evaluation (no Isaac Sim needed)
    PYTHONPATH=src python scripts/08_eval.py --checkpoint checkpoints/best.pt

    # Detailed evaluation with more samples
    PYTHONPATH=src python scripts/08_eval.py --checkpoint checkpoints/best.pt --num-samples 100

    # Compare multiple checkpoints
    PYTHONPATH=src python scripts/08_eval.py --checkpoint checkpoints/epoch_0100.pt \
                                              --checkpoint checkpoints/best.pt
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Ensure project src/ is importable
_src = str(Path(__file__).resolve().parent.parent / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from models.diffusion_policy import DiffusionPolicy
from models.noise_scheduler import DDPMScheduler
from utils.config import load_config, TrainConfig
from vl_data.dataset import EpisodicDataset


# =============================================================================
# Metrics dataclass
# =============================================================================


@dataclass
class EvalMetrics:
    """Aggregated evaluation metrics for one checkpoint."""
    checkpoint: str = ""
    num_samples: int = 0

    # Overall
    mse: float = 0.0          # Mean squared error over all actions
    rmse: float = 0.0         # Root MSE
    mae: float = 0.0          # Mean absolute error

    # Per-joint breakdown
    joint_mse: list[float] = field(default_factory=lambda: [0.0] * 7)
    joint_mae: list[float] = field(default_factory=lambda: [0.0] * 7)

    # Trajectory-level
    traj_smoothness: float = 0.0  # Mean ||a_t - a_{t-1}||₂
    traj_range_ratio: float = 0.0 # (pred range) / (gt range) — 1.0 is perfect

    # Timing
    inference_time_ms: float = 0.0  # Per-sample inference time

    # Per-step prediction accuracy (how well each horizon step is predicted)
    step_wise_mse: list[float] = field(default_factory=lambda: [0.0] * 16)

    @property
    def joint_names(self) -> list[str]:
        return [f"joint_{i + 1}" for i in range(7)]


# =============================================================================
# Evaluation
# =============================================================================


@torch.no_grad()
def evaluate_checkpoint(
    checkpoint_path: str | Path,
    cfg: TrainConfig,
    val_dataset,
    collate_fn,
    num_samples: int = 50,
    num_inference_steps: int = 16,
) -> EvalMetrics:
    """Evaluate a single checkpoint on validation data.

    Parameters
    ----------
    checkpoint_path : str or Path
        Path to a ``.pt`` checkpoint.
    cfg : TrainConfig
        Training configuration (used to recreate the model architecture).
    val_dataset : EpisodicDataset or Subset
        Validation dataset.
    collate_fn : callable
        Collate function from the dataset.
    num_samples : int
        Max number of validation samples to evaluate.
    num_inference_steps : int
        Number of DDIM inference steps.

    Returns
    -------
    EvalMetrics
        Aggregated evaluation metrics.
    """
    device = torch.device(cfg.training.device if cfg.training.device != "auto" else "cpu")
    checkpoint_path = Path(checkpoint_path)

    print(f"\n{'─' * 55}")
    print(f"  Evaluating: {checkpoint_path.name}")
    print(f"{'─' * 55}")

    # --- Load model ---
    model = DiffusionPolicy(
        action_dim=cfg.model.action_dim,
        action_horizon=cfg.model.action_horizon,
        obs_horizon=cfg.model.obs_horizon,
        state_dim=cfg.model.state_dim,
        vision_output_dim=cfg.model.vision_output_dim,
        state_output_dim=cfg.model.state_output_dim,
        time_dim=cfg.model.time_dim,
        unet_base_channels=cfg.model.unet_base_channels,
        use_vision=cfg.model.use_vision,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Load normalization stats (may not exist for older checkpoints)
    act_mean = ckpt.get("act_mean", None)
    act_std = ckpt.get("act_std", None)
    if act_mean is not None:
        act_mean = act_mean.to(device).view(1, 1, -1)
        act_std = act_std.to(device).view(1, 1, -1)
        print(f"  Normalization: mean={act_mean.flatten()[:3].tolist()}..., std={act_std.flatten()[:3].tolist()}...")
    else:
        print(f"  ⚠️  No normalization stats in checkpoint — assuming raw radian data")

    train_epoch = ckpt.get("epoch", "?")
    print(f"  Loaded epoch {train_epoch}, val_loss={ckpt.get('best_val_loss', '?'):.4f}")

    # --- Scheduler ---
    scheduler = DDPMScheduler(
        num_train_steps=cfg.diffusion.num_train_steps,
        schedule=cfg.diffusion.schedule,
        device=device,
    )

    # --- Create DataLoader for evaluation ---
    loader = DataLoader(
        val_dataset,
        batch_size=1,  # One sample at a time for fair timing
        shuffle=False,
        collate_fn=collate_fn,
    )

    # --- Run evaluation ---
    all_mse = []
    all_mae = []
    all_joint_mse = [[] for _ in range(cfg.model.action_dim)]
    all_joint_mae = [[] for _ in range(cfg.model.action_dim)]
    all_smoothness = []
    all_range_ratios = []
    all_step_mse = [[] for _ in range(cfg.model.action_horizon)]
    inference_times = []

    n_evaluated = 0
    for batch_idx, batch in enumerate(loader):
        if n_evaluated >= num_samples:
            break

        actions_gt = batch["actions"].to(device)  # (1, T, D)
        if actions_gt.shape[1] != cfg.model.action_horizon:
            continue

        # Normalize ground truth if stats available
        if act_mean is not None:
            actions_gt_norm = (actions_gt - act_mean) / (act_std + 1e-8)
        else:
            actions_gt_norm = actions_gt

        # Prepare observations
        obs = {}
        for key, tensor in batch["observations"].items():
            if tensor.numel() > 0:
                obs[key] = tensor.to(device)

        # Time the inference (model outputs normalized actions)
        t0 = time.monotonic()
        actions_pred_norm = model.predict_action(
            obs, scheduler,
            device=device,
            use_ddpm=True,  # DDPM: numerically stable for cosine schedule
        )  # (1, T, D)
        t1 = time.monotonic()

        # Denormalize predictions
        if act_mean is not None:
            actions_pred = actions_pred_norm * act_std + act_mean
        else:
            actions_pred = actions_pred_norm

        inference_times.append((t1 - t0) * 1000)

        # --- Compute metrics (in original radian space) ---
        gt = actions_gt  # (1, T, D)
        # Ensure both have same batch dim
        if gt.ndim == 2:
            gt = gt.unsqueeze(0)
        pred = actions_pred  # (1, T, D)

        # Overall MSE / MAE
        mse = F.mse_loss(pred, gt).item()
        mae = F.l1_loss(pred, gt).item()
        all_mse.append(mse)
        all_mae.append(mae)

        # Per-joint
        for j in range(cfg.model.action_dim):
            jmse = F.mse_loss(pred[:, :, j], gt[:, :, j]).item()
            jmae = F.l1_loss(pred[:, :, j], gt[:, :, j]).item()
            all_joint_mse[j].append(jmse)
            all_joint_mae[j].append(jmae)

        # Per-step (horizon)
        for t in range(cfg.model.action_horizon):
            smse = F.mse_loss(pred[:, t, :], gt[:, t, :]).item()
            all_step_mse[t].append(smse)

        # Trajectory smoothness: mean ||a_t - a_{t-1}||
        diffs = torch.norm(pred[0, 1:] - pred[0, :-1], dim=1)
        smoothness = diffs.mean().item()
        all_smoothness.append(smoothness)

        # Range ratio
        pred_range = pred.max() - pred.min()
        gt_range = gt.max() - gt.min()
        if gt_range > 1e-8:
            all_range_ratios.append((pred_range / gt_range).item())

        n_evaluated += 1

    # --- Aggregate ---
    metrics = EvalMetrics(
        checkpoint=checkpoint_path.name,
        num_samples=n_evaluated,
        mse=float(np.mean(all_mse)),
        rmse=float(np.sqrt(np.mean(all_mse))),
        mae=float(np.mean(all_mae)),
        joint_mse=[float(np.mean(ms)) for ms in all_joint_mse],
        joint_mae=[float(np.mean(ms)) for ms in all_joint_mae],
        traj_smoothness=float(np.mean(all_smoothness)),
        traj_range_ratio=float(np.mean(all_range_ratios)) if all_range_ratios else 0.0,
        inference_time_ms=float(np.mean(inference_times)),
        step_wise_mse=[float(np.mean(ms)) for ms in all_step_mse],
    )

    return metrics


# =============================================================================
# Reporting
# =============================================================================


def print_metrics(metrics: EvalMetrics) -> None:
    """Pretty-print evaluation metrics."""
    print(f"\n  Checkpoint: {metrics.checkpoint}")
    print(f"  Samples evaluated: {metrics.num_samples}")
    print()
    print(f"  {'Metric':<25s} {'Value':>10s}")
    print(f"  {'─' * 25} {'─' * 10}")
    print(f"  {'MSE':<25s} {metrics.mse:>10.6f}")
    print(f"  {'RMSE':<25s} {metrics.rmse:>10.6f}")
    print(f"  {'MAE':<25s} {metrics.mae:>10.6f}")
    print(f"  {'Traj Smoothness':<25s} {metrics.traj_smoothness:>10.6f}")
    print(f"  {'Range Ratio (pred/gt)':<25s} {metrics.traj_range_ratio:>10.4f}")
    print(f"  {'Inference Time':<25s} {metrics.inference_time_ms:>9.1f} ms")

    # Per-joint breakdown
    print(f"\n  Per-Joint MSE:")
    print(f"  {'Joint':<12s}", end="")
    for name in metrics.joint_names:
        print(f"{name:>10s}", end="")
    print()
    print(f"  {'─' * 12}", end="")
    print(f"{'─' * 10}" * 7)
    print(f"  {'MSE':<12s}", end="")
    for v in metrics.joint_mse:
        print(f"{v:>10.4f}", end="")
    print()
    print(f"  {'MAE':<12s}", end="")
    for v in metrics.joint_mae:
        print(f"{v:>10.4f}", end="")
    print()

    # Per-step MSE (horizon)
    print(f"\n  Prediction Horizon Quality (MSE by step):")
    # Find max for scaling
    max_mse = max(metrics.step_wise_mse) if max(metrics.step_wise_mse) > 0 else 1.0
    print(f"  {'Step':>5s}  {'MSE':>10s}  {'Bar'}")
    print(f"  {'─' * 5}  {'─' * 10}  {'─' * 30}")
    for t, mse in enumerate(metrics.step_wise_mse):
        bar_len = int(mse / max_mse * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        print(f"  {t:>5d}  {mse:>10.4f}  {bar}")


def compare_checkpoints(all_metrics: list[EvalMetrics]) -> None:
    """Print side-by-side comparison of multiple checkpoints."""
    if len(all_metrics) < 2:
        return

    print(f"\n{'=' * 55}")
    print(f"  Checkpoint Comparison")
    print(f"{'=' * 55}")
    print(f"  {'Checkpoint':<25s} {'MSE':>8s}  {'RMSE':>8s}  {'MAE':>8s}  {'Range':>8s}")
    print(f"  {'─' * 25} {'─' * 8}  {'─' * 8}  {'─' * 8}  {'─' * 8}")
    for m in all_metrics:
        print(f"  {m.checkpoint:<25s} {m.mse:>8.4f}  {m.rmse:>8.4f}  {m.mae:>8.4f}  {m.traj_range_ratio:>8.3f}")
    print()

    # Find best
    best = min(all_metrics, key=lambda m: m.mse)
    print(f"  ✅ Best checkpoint: {best.checkpoint} (MSE={best.mse:.4f})")


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="VLARM — Model Evaluation (Day 8)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", type=str, action="append", dest="checkpoints",
        help="Path to checkpoint .pt file (repeatable for comparison). "
             "Default: checkpoints/best.pt",
    )
    parser.add_argument(
        "--config", type=str, default="configs/train_config.yaml",
        help="Path to training config (must match checkpoint architecture)",
    )
    parser.add_argument(
        "--num-samples", type=int, default=50,
        help="Number of validation samples to evaluate (default: 50)",
    )
    parser.add_argument(
        "--num-inference-steps", type=int, default=16,
        help="Number of DDIM inference steps (default: 16)",
    )
    parser.add_argument(
        "--data-dir", type=str, default="data/raw",
        help="Path to HDF5 episode directory",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device for inference (cpu/cuda)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    # Default checkpoint
    if not args.checkpoints:
        args.checkpoints = ["checkpoints/best.pt"]

    print("=" * 55)
    print("  VLARM — Day 8: Model Evaluation")
    print("=" * 55)
    print(f"  Checkpoints: {args.checkpoints}")
    print(f"  Num samples: {args.num_samples}")
    print(f"  DDIM steps:  {args.num_inference_steps}")
    print(f"  Device:      {args.device}")

    # --- Load config ---
    cfg = load_config(args.config)
    cfg.training.device = args.device

    # --- Load validation data ---
    print(f"\n  Loading validation data from {args.data_dir}...")

    # Identify bad episodes
    import h5py
    bad_ids = set(cfg.data.exclude_episodes or [])
    data_path = Path(args.data_dir)
    all_ep_files = sorted(data_path.glob("episode_*.h5"))
    good_files = []
    for fp in all_ep_files:
        ep_id = int(fp.stem.split("_")[1])
        if ep_id in bad_ids:
            continue
        with h5py.File(fp, "r") as f:
            jp = f["observations/joint_positions"][:, :7]
            jmax = float(np.abs(jp).max())
            jmin = float(jp.min())
            success = bool(f.attrs.get("success", True))
        if jmax <= 6.28 and jmin >= -6.28 and success:
            good_files.append(fp)

    # For evaluation, use a subset of good episodes as "held-out"
    # Take the last 20% of episodes as validation
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    val_size = max(1, len(good_files) // 5)
    # Use deterministic split: last val_size files
    val_files = good_files[-val_size:]

    # Create a temporary directory with only validation files for evaluation
    import tempfile, shutil
    tmp_dir = Path(tempfile.mkdtemp(prefix="vlarm_eval_"))
    for fp in val_files:
        shutil.copy2(fp, tmp_dir / fp.name)

    val_dataset = EpisodicDataset(
        str(tmp_dir),
        obs_horizon=cfg.data.obs_horizon,
        action_horizon=cfg.data.action_horizon,
        action_downsample=cfg.data.action_downsample,
    )
    collate_fn = val_dataset.collate_fn

    print(f"  Validation episodes: {len(val_files)}")
    print(f"  Validation samples:  {len(val_dataset)}")

    # --- Evaluate each checkpoint ---
    all_metrics: list[EvalMetrics] = []
    for ckpt_path in args.checkpoints:
        if not Path(ckpt_path).exists():
            print(f"  ⚠️  Checkpoint not found: {ckpt_path} — skipping")
            continue
        metrics = evaluate_checkpoint(
            ckpt_path, cfg, val_dataset, collate_fn,
            num_samples=args.num_samples,
            num_inference_steps=args.num_inference_steps,
        )
        print_metrics(metrics)
        all_metrics.append(metrics)

    # --- Compare ---
    compare_checkpoints(all_metrics)

    # Cleanup
    val_dataset.close()
    shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"\n{'=' * 55}")
    print(f"  ✅ Evaluation complete!")
    print(f"{'=' * 55}")


if __name__ == "__main__":
    main()
