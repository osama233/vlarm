#!/usr/bin/env python3
"""Day 7 — VLARM Diffusion Policy Training Pipeline.

Full training loop with:
  - YAML configuration (override via command-line)
  - Auto episode filtering (skip episodes with joint overflows)
  - Train/validation split
  - Cosine LR schedule with warmup
  - TensorBoard logging (loss, lr, grad_norm)
  - Checkpoint saving (best + periodic)
  - Gradient clipping

Usage::

    conda activate vlarm
    PYTHONPATH=src python scripts/07_train.py                           # default config
    PYTHONPATH=src python scripts/07_train.py --batch-size 32 --epochs 500  # override
    PYTHONPATH=src python scripts/07_train.py --device cpu --epochs 5       # quick test
    PYTHONPATH=src python scripts/07_train.py --resume checkpoints/last.pt   # resume

Outputs:
    logs/                      TensorBoard event files
    checkpoints/last.pt        Most recent checkpoint (resume)
    checkpoints/best.pt        Best validation-loss checkpoint
    checkpoints/epoch_*.pt     Periodic checkpoints
    checkpoints/config.yaml    Frozen config snapshot for reproducibility
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split
from torch.utils.tensorboard import SummaryWriter

# Ensure project src/ is importable
_src = str(Path(__file__).resolve().parent.parent / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from models.diffusion_policy import DiffusionPolicy
from models.noise_scheduler import DDPMScheduler
from utils.config import load_config, save_config, TrainConfig
from vl_data.dataset import EpisodicDataset


# =============================================================================
# Episode filtering
# =============================================================================


def _identify_bad_episodes(
    data_dir: str, exclude_ids: list[int] | None = None
) -> tuple[list[int], list[int]]:
    """Scan HDF5 files and flag episodes with invalid joint values.

    An episode is considered "bad" if:
    - Any arm joint (first 7 DOF) exceeds ±6.28 rad (one full revolution)
    - The episode attribute ``success`` is False

    Parameters
    ----------
    data_dir : str
        Path to the HDF5 episode directory.
    exclude_ids : list[int] or None
        Additional episode IDs to exclude (e.g. from config).

    Returns
    -------
    good_ids : list[int]
        Episode IDs that pass validation.
    bad_ids : list[int]
        Episode IDs to exclude from training.
    """
    import h5py

    data_path = Path(data_dir)
    if not data_path.exists():
        return [], []

    bad = set(exclude_ids or [])
    all_ids = []
    for fp in sorted(data_path.glob("episode_*.h5")):
        ep_id = int(fp.stem.split("_")[1])
        all_ids.append(ep_id)
        with h5py.File(fp, "r") as f:
            jp = f["observations/joint_positions"][:, :7]  # arm joints only
            jmax = float(np.abs(jp).max())
            jmin = float(jp.min())
            success = bool(f.attrs.get("success", True))
        if jmax > 6.28 or jmin < -6.28 or not success:
            bad.add(ep_id)

    good = sorted(set(all_ids) - bad)
    bad = sorted(bad)
    return good, bad


def _filter_dataset(
    dataset: EpisodicDataset,
    bad_ids: list[int],
    data_dir: str,
) -> EpisodicDataset:
    """Return a new dataset that excludes bad episodes.

    This works by re-initialising the dataset with only the episode files
    whose IDs are not in ``bad_ids``.  Since ``EpisodicDataset`` scans for
    ``episode_*.h5``, we use ``Subset`` as a simpler approach: filter the
    global index entries that belong to bad episodes.
    """
    # Map file basename → file index
    bad_files = {f"episode_{i:05d}.h5" for i in bad_ids}

    good_indices = []
    for idx in range(len(dataset)):
        file_idx, _ = dataset._index[idx]
        fp = dataset._episode_files[file_idx]
        if fp.name not in bad_files:
            good_indices.append(idx)

    if len(good_indices) == len(dataset):
        return dataset  # nothing to filter

    subset = Subset(dataset, good_indices)
    # Preserve key attributes for downstream use
    subset._filtered_episode_files = [
        fp for fp in dataset._episode_files
        if fp.name not in bad_files
    ]
    subset._bad_ids = bad_ids
    return subset


# =============================================================================
# Learning rate utilities
# =============================================================================


class _WarmupCosineScheduler:
    """Linear warmup followed by cosine annealing.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
    warmup_steps : int
        Number of warmup update steps.
    total_steps : int
        Total number of update steps (warmup + decay).
    min_lr_ratio : float
        Final LR as a fraction of initial LR.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.0,
    ) -> None:
        self._optimizer = optimizer
        self._warmup = max(1, warmup_steps)
        self._total = max(1, total_steps)
        self._min_ratio = min_lr_ratio
        self._base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self._step = 0

    def step(self) -> None:
        self._step += 1
        lr_scale = self._compute_scale()
        for group, base in zip(self._optimizer.param_groups, self._base_lrs):
            group["lr"] = base * lr_scale

    def _compute_scale(self) -> float:
        step = self._step
        if step <= self._warmup:
            return step / self._warmup
        # Cosine decay
        progress = (step - self._warmup) / max(1, self._total - self._warmup)
        progress = min(progress, 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self._min_ratio + (1.0 - self._min_ratio) * cosine

    def get_last_lr(self) -> list[float]:
        return [pg["lr"] for pg in self._optimizer.param_groups]


# =============================================================================
# Training loop helpers
# =============================================================================


def _normalize_actions(actions: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Normalize actions to zero-mean unit-variance: (a - mean) / (std + eps)."""
    eps = 1e-8
    return (actions - mean) / (std + eps)


def _denormalize_actions(actions_norm: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Reverse normalization: a_norm * std + mean."""
    return actions_norm * std + mean


def _prepare_obs_for_model(batch: dict, device: torch.device) -> dict[str, torch.Tensor]:
    """Extract and move observation tensors to device.

    The dataset returns observations with shape ``(B, obs_horizon, D)``.
    The model expects the full horizon (it uses ``[:, -1]`` internally for
    the last frame, but needs all frames for future temporal conditioning).
    """
    obs = {}
    for key, tensor in batch["observations"].items():
        if tensor.numel() > 0:
            obs[key] = tensor.to(device)
    return obs


@torch.no_grad()
def _validate(
    model: DiffusionPolicy,
    scheduler: DDPMScheduler,
    val_loader: DataLoader,
    device: torch.device,
    act_mean: torch.Tensor,
    act_std: torch.Tensor,
    max_batches: int = 0,
) -> dict[str, float]:
    """Run a validation pass and return aggregate metrics.

    Parameters
    ----------
    max_batches : int
        If > 0, only evaluate this many batches (for faster validation).
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for batch_idx, batch in enumerate(val_loader):
        if max_batches > 0 and batch_idx >= max_batches:
            break

        actions = _normalize_actions(
            batch["actions"].to(device), act_mean, act_std
        )
        B = actions.shape[0]

        noise = torch.randn_like(actions)
        timesteps = torch.randint(
            0, scheduler.num_train_steps, (B,), device=device, dtype=torch.long
        )
        noisy = scheduler.add_noise(actions, noise, timesteps)
        obs = _prepare_obs_for_model(batch, device)
        pred_noise = model(noisy, timesteps, obs)
        loss = F.mse_loss(pred_noise, noise)

        total_loss += loss.item()
        n_batches += 1

    model.train()
    return {"val_loss": total_loss / max(1, n_batches), "val_batches": float(n_batches)}


# =============================================================================
# Main training loop
# =============================================================================


def train(cfg: TrainConfig, resume_from: str | None = None) -> None:
    """Run the full Diffusion Policy training loop.

    Parameters
    ----------
    cfg : TrainConfig
        Complete training configuration.
    resume_from : str or None
        Path to a checkpoint to resume from.
    """
    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    device = torch.device(cfg.training.device)
    seed = cfg.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    out_dir = Path(cfg.logging.checkpoint_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(cfg.logging.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    run_log_dir = log_dir / run_id
    writer = SummaryWriter(log_dir=str(run_log_dir))

    # Save frozen config for reproducibility
    config_snapshot = out_dir / "config.yaml"
    save_config(cfg, config_snapshot)
    print(f"Checkpoint dir: {out_dir}")
    print(f"Config snapshot saved to {config_snapshot}")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    print(f"\n{'=' * 55}")
    print("  Loading data...")
    print(f"{'=' * 55}")

    data_dir = cfg.data.data_dir

    # Identify bad episodes
    good_ids, bad_ids = _identify_bad_episodes(
        data_dir, exclude_ids=cfg.data.exclude_episodes
    )
    if bad_ids:
        print(f"  ⚠️  Excluding {len(bad_ids)} bad episodes: {bad_ids}")
    print(f"  ✅ {len(good_ids)} good episodes for training")

    # Create dataset
    full_dataset = EpisodicDataset(
        data_dir=data_dir,
        obs_horizon=cfg.data.obs_horizon,
        action_horizon=cfg.data.action_horizon,
        action_downsample=cfg.data.action_downsample,
    )

    # Apply filtering
    if bad_ids:
        dataset = _filter_dataset(full_dataset, bad_ids, data_dir)
    else:
        dataset = full_dataset

    print(f"  Total samples: {len(dataset)}")

    # Train/val split
    val_size = max(1, int(len(dataset) * cfg.data.val_ratio))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )
    print(f"  Train: {train_size}, Val: {val_size}")

    # Collate function (must be on dataset, not subset)
    collate_fn = full_dataset.collate_fn

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        collate_fn=collate_fn,
        drop_last=True,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        collate_fn=collate_fn,
        drop_last=False,
        pin_memory=(device.type == "cuda"),
    )

    # ------------------------------------------------------------------
    # Normalization statistics (computed on full training data)
    # ------------------------------------------------------------------
    act_stats = full_dataset.compute_action_stats()
    act_mean = torch.from_numpy(act_stats["mean"]).to(device).view(1, 1, -1)  # (1, 1, D)
    act_std = torch.from_numpy(act_stats["std"]).to(device).view(1, 1, -1)

    print(f"\n  Action normalization (arm joints, radians):")
    print(f"    mean: {act_stats['mean']}")
    print(f"    std:  {act_stats['std']}")
    # Warn if any joint has very low variance
    for j, s in enumerate(act_stats["std"]):
        if s < 0.01:
            print(f"    ⚠️  joint_{j+1} std={s:.4f} (very low variance — check data!)")
    print()

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    print(f"\n{'=' * 55}")
    print("  Building model...")
    print(f"{'=' * 55}")

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
        pretrained_vision=cfg.model.pretrained_vision,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,} total, {n_trainable:,} trainable")
    print(f"  Use vision: {cfg.model.use_vision}")

    scheduler = DDPMScheduler(
        num_train_steps=cfg.diffusion.num_train_steps,
        schedule=cfg.diffusion.schedule,
        device=device,
    )

    # ------------------------------------------------------------------
    # Optimizer & LR schedule
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )

    total_steps = len(train_loader) * cfg.training.epochs
    warmup_steps = cfg.lr_schedule.warmup_epochs * len(train_loader)

    if cfg.lr_schedule.name == "cosine":
        lr_scheduler = _WarmupCosineScheduler(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr_ratio=1e-3,
        )
    else:
        lr_scheduler = None

    # AMP scaler (CPU-safe)
    scaler = None
    if cfg.training.use_amp and device.type == "cuda":
        scaler = torch.amp.GradScaler("cuda")

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------
    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")

    if resume_from:
        print(f"\n  Resuming from {resume_from}")
        ckpt = torch.load(resume_from, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0)
        global_step = ckpt.get("global_step", 0)
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"  Resumed at epoch {start_epoch}, step {global_step}")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    print(f"\n{'=' * 55}")
    print(f"  Training — {cfg.training.epochs} epochs, {len(train_loader)} batches/epoch")
    print(f"  Device: {device}, LR: {cfg.training.lr}")
    print(f"  Log dir:  {run_log_dir}")
    print(f"  Checkpoint dir: {out_dir}")
    print(f"{'=' * 55}\n")

    model.train()
    t_train_start = time.monotonic()

    for epoch in range(start_epoch, cfg.training.epochs):
        epoch_loss = 0.0
        epoch_steps = 0
        t_epoch_start = time.monotonic()

        for batch_idx, batch in enumerate(train_loader):
            # --- Prepare ---
            actions = _normalize_actions(
                batch["actions"].to(device), act_mean, act_std
            )  # (B, T, D)
            B = actions.shape[0]

            # Sample noise + timesteps
            noise = torch.randn_like(actions)
            timesteps = torch.randint(
                0, scheduler.num_train_steps, (B,), device=device, dtype=torch.long
            )

            # Forward diffusion
            noisy_actions = scheduler.add_noise(actions, noise, timesteps)

            # Observation conditioning
            obs = _prepare_obs_for_model(batch, device)

            # --- Forward / backward ---
            optimizer.zero_grad()

            if scaler is not None:
                with torch.amp.autocast("cuda"):
                    pred_noise = model(noisy_actions, timesteps, obs)
                    loss = F.mse_loss(pred_noise, noise)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                pred_noise = model(noisy_actions, timesteps, obs)
                loss = F.mse_loss(pred_noise, noise)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip_norm)
                optimizer.step()

            # LR schedule (per-step)
            if lr_scheduler is not None:
                lr_scheduler.step()

            # --- Logging ---
            loss_val = loss.item()
            epoch_loss += loss_val
            epoch_steps += 1
            global_step += 1

            if global_step % cfg.logging.log_every_steps == 0:
                current_lr = optimizer.param_groups[0]["lr"]
                grad_norm = _compute_grad_norm(model)
                writer.add_scalar("train/loss", loss_val, global_step)
                writer.add_scalar("train/lr", current_lr, global_step)
                writer.add_scalar("train/grad_norm", grad_norm, global_step)

                elapsed = time.monotonic() - t_train_start
                print(
                    f"  [Epoch {epoch + 1:3d}/{cfg.training.epochs}] "
                    f"step {batch_idx + 1:4d}/{len(train_loader)} | "
                    f"loss={loss_val:.4f} | lr={current_lr:.2e} | "
                    f"grad={grad_norm:.3f} | {elapsed:.0f}s"
                )

        # --- Epoch summary ---
        avg_loss = epoch_loss / max(1, epoch_steps)
        epoch_time = time.monotonic() - t_epoch_start
        writer.add_scalar("train/epoch_loss", avg_loss, epoch)
        print(
            f"  ── Epoch {epoch + 1:3d} done | "
            f"avg_loss={avg_loss:.4f} | {epoch_time:.1f}s ──"
        )

        # --- Validation ---
        if (epoch + 1) % cfg.logging.eval_every_epochs == 0 or epoch == 0:
            val_metrics = _validate(
                model, scheduler, val_loader, device, act_mean, act_std, max_batches=20
            )
            val_loss = val_metrics["val_loss"]
            writer.add_scalar("val/loss", val_loss, epoch)
            print(f"         Validation | val_loss={val_loss:.4f}")

            # Save best
            if cfg.logging.save_best and val_loss < best_val_loss:
                best_val_loss = val_loss
                _save_checkpoint(
                    out_dir / "best.pt",
                    model, optimizer, epoch + 1, global_step, best_val_loss,
                    act_mean=act_mean, act_std=act_std,
                )
                print(f"         🏆 New best! saved to best.pt")

        # --- Periodic checkpoint ---
        if (epoch + 1) % cfg.logging.save_every_epochs == 0:
            ckpt_path = out_dir / f"epoch_{epoch + 1:04d}.pt"
            _save_checkpoint(
                ckpt_path,
                model, optimizer, epoch + 1, global_step, best_val_loss,
                act_mean=act_mean, act_std=act_std,
            )

        # --- Always save latest ---
        _save_checkpoint(
            out_dir / "last.pt",
            model, optimizer, epoch + 1, global_step, best_val_loss,
            act_mean=act_mean, act_std=act_std,
        )

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    total_time = time.monotonic() - t_train_start
    print(f"\n{'=' * 55}")
    print(f"  ✅ Training complete!")
    print(f"  Total time:   {total_time / 60:.1f} min")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Final model:   {out_dir / 'last.pt'}")
    print(f"  Best model:    {out_dir / 'best.pt'}")
    print(f"  TensorBoard:   {run_log_dir}")
    print(f"{'=' * 55}")

    writer.close()
    # Clean up dataset file handles
    if hasattr(full_dataset, "close"):
        full_dataset.close()


# =============================================================================
# Checkpoint helpers
# =============================================================================


def _save_checkpoint(
    path: Path,
    model: DiffusionPolicy,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    best_val_loss: float,
    act_mean: torch.Tensor | None = None,
    act_std: torch.Tensor | None = None,
) -> None:
    """Save a training checkpoint."""
    ckpt = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_val_loss": best_val_loss,
    }
    if act_mean is not None:
        ckpt["act_mean"] = act_mean.cpu()
    if act_std is not None:
        ckpt["act_std"] = act_std.cpu()
    torch.save(ckpt, path)


def _compute_grad_norm(model: torch.nn.Module) -> float:
    """Compute the total gradient norm across all parameters."""
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.data.norm(2).item() ** 2
    return math.sqrt(total)


# =============================================================================
# CLI
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="VLARM — Diffusion Policy Training (Day 7)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", type=str, default="configs/train_config.yaml",
        help="Path to YAML config file",
    )

    # Overrides group
    group = parser.add_argument_group("Config overrides")
    group.add_argument("--batch-size", type=int, default=None)
    group.add_argument("--epochs", type=int, default=None)
    group.add_argument("--lr", type=float, default=None)
    group.add_argument("--device", type=str, default=None)
    group.add_argument("--data-dir", type=str, default=None)
    group.add_argument("--log-dir", type=str, default=None)
    group.add_argument("--checkpoint-dir", type=str, default=None)
    group.add_argument("--no-vision", action="store_true", default=None)
    group.add_argument("--num-workers", type=int, default=None)
    group.add_argument("--resume", type=str, default=None,
                       help="Resume from a checkpoint file")
    group.add_argument("--seed", type=int, default=None)

    args = parser.parse_args()

    # Build overrides dict from CLI args (skip None and the config path)
    overrides = {}
    override_map = {
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "device": args.device,
        "data_dir": args.data_dir,
        "log_dir": args.log_dir,
        "checkpoint_dir": args.checkpoint_dir,
        "num_workers": args.num_workers,
        "seed": args.seed,
    }
    for top_key, val in override_map.items():
        if val is not None:
            overrides[f"training.{top_key}"] = val

    if args.no_vision:
        overrides["model.use_vision"] = False

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        # Try relative to script dir
        alt = Path(__file__).resolve().parent.parent / args.config
        if alt.exists():
            config_path = alt

    cfg = load_config(config_path, overrides=overrides)

    print("=" * 55)
    print("  VLARM — Day 7: Diffusion Policy Training")
    print("=" * 55)
    print(f"  Config:    {config_path}")
    print(f"  Start:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  PyTorch:   {torch.__version__}")
    print(f"  Device:    {cfg.training.device}")

    train(cfg, resume_from=args.resume)


if __name__ == "__main__":
    main()
