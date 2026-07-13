#!/usr/bin/env python3
"""Day 6 — Model Unit Tests.

Validates the three core modules (noise scheduler, vision encoder,
diffusion policy) and runs a mini training loop with real collected data.

Usage (outside Isaac Sim)::

    conda activate vlarm
    PYTHONPATH=src python scripts/06_test_models.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project src/ is importable
_src = str(Path(__file__).resolve().parent.parent / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


# ===========================================================================
# Test 1: Noise Scheduler
# ===========================================================================


def test_noise_scheduler() -> int:
    """Verify DDPM/DDIM properties."""
    print("─" * 50)
    print("Test 1: Noise Scheduler")
    print("─" * 50)

    from models.noise_scheduler import DDPMScheduler

    errors = 0

    for schedule in ("linear", "cosine", "squared_cosine"):
        sch = DDPMScheduler(num_train_steps=100, schedule=schedule)

        # 1a. Forward: more noise at higher t
        B, T, D = 8, 16, 7
        x0 = torch.randn(B, T, D)
        noise = torch.randn(B, T, D)

        xt_early = sch.add_noise(x0, noise, torch.full((B,), 10, dtype=torch.long))
        xt_late = sch.add_noise(x0, noise, torch.full((B,), 90, dtype=torch.long))

        # Early timestep: should be closer to original
        diff_early = (xt_early - x0).norm(dim=(1, 2)).mean()
        diff_late = (xt_late - x0).norm(dim=(1, 2)).mean()
        if not (diff_late > diff_early):
            print(f"  ❌ {schedule}: t=10 should be closer to x0 than t=90")
            errors += 1
        else:
            print(f"  ✅ {schedule}: diff(t=10)={diff_early:.2f} < diff(t=90)={diff_late:.2f}")

        # 1b. DDIM steps are descending
        steps = sch.sample_steps(16)
        assert steps == sorted(steps, reverse=True), f"Steps not descending: {steps[:5]}"
        assert steps[0] == 99

        # 1c. DDPM step produces valid output
        eps = torch.randn(B, T, D)
        x = torch.randn(B, T, D)
        x_prev = sch.ddpm_step(eps, timestep=50, current_x=x)
        assert x_prev.shape == x.shape

        # 1d. DDIM step produces valid output
        x_ddim = sch.ddim_step(eps, timestep=99, next_timestep=93, current_x=x)
        assert x_ddim.shape == x.shape

    print(f"  [{errors} errors]\n")
    return errors


# ===========================================================================
# Test 2: Vision Encoder
# ===========================================================================


def test_vision_encoder() -> int:
    """Verify ResNet-18 encoder shapes and modes."""
    print("─" * 50)
    print("Test 2: Vision Encoder")
    print("─" * 50)

    from models.vision_encoder import ResNet18Encoder

    errors = 0
    B, H, W = 4, 480, 640

    # 2a. Basic forward
    enc = ResNet18Encoder(output_dim=512, use_film=False)
    rgb = torch.randn(B, H, W, 3)
    out = enc(rgb)
    if out.shape != (B, 512):
        print(f"  ❌ Expected (4, 512), got {out.shape}")
        errors += 1
    else:
        print(f"  ✅ Basic forward: {list(rgb.shape)} → {list(out.shape)}")

    # 2b. FiLM conditioning
    enc_film = ResNet18Encoder(output_dim=512, use_film=True, time_dim=128)
    t_embed = torch.randn(B, 128)
    out_film = enc_film(rgb, t_embed)
    if out_film.shape != (B, 512):
        print(f"  ❌ FiLM forward failed: {out_film.shape}")
        errors += 1
    else:
        print(f"  ✅ FiLM conditioning: output shape {list(out_film.shape)}")

    # 2c. Custom output dim
    enc_256 = ResNet18Encoder(output_dim=256)
    out_256 = enc_256(rgb)
    if out_256.shape != (B, 256):
        print(f"  ❌ Custom dim failed: {out_256.shape}")
        errors += 1
    else:
        print(f"  ✅ Custom output dim: {list(out_256.shape)}")

    # 2d. Frozen backbone
    enc_frozen = ResNet18Encoder(freeze_backbone=True)
    frozen = sum(1 for p in enc_frozen._backbone.parameters() if p.requires_grad)
    if frozen != 0:
        print(f"  ❌ Expected 0 frozen, got {frozen}")
        errors += 1
    else:
        print(f"  ✅ Frozen backbone: {frozen} trainable backbone params")

    print(f"  [{errors} errors]\n")
    return errors


# ===========================================================================
# Test 3: Diffusion Policy shapes & gradient flow
# ===========================================================================


def test_diffusion_policy() -> int:
    """Verify model shapes, conditioning paths, and gradient flow."""
    print("─" * 50)
    print("Test 3: Diffusion Policy")
    print("─" * 50)

    from models.diffusion_policy import DiffusionPolicy
    from models.noise_scheduler import DDPMScheduler

    errors = 0
    B, T, D = 4, 16, 7
    device = "cpu"

    # 3a. With vision
    model = DiffusionPolicy(action_dim=D, action_horizon=T, use_vision=True)
    model.train()

    noisy = torch.randn(B, T, D)
    t = torch.randint(0, 100, (B,), dtype=torch.long)
    obs = {
        "rgb": torch.rand(B, 2, 480, 640, 3),
        "joint_positions": torch.randn(B, 2, 9),
        "ee_position": torch.randn(B, 2, 3),
        "gripper_width": torch.rand(B, 2, 1),
    }
    pred = model(noisy, t, obs)
    if pred.shape != (B, T, D):
        print(f"  ❌ Vision mode: expected ({B},{T},{D}), got {pred.shape}")
        errors += 1
    else:
        print(f"  ✅ Vision mode: shape {list(pred.shape)}")

    # 3b. No vision
    model_novis = DiffusionPolicy(action_dim=D, action_horizon=T, use_vision=False)
    obs_novis = {k: v for k, v in obs.items() if k != "rgb"}
    pred_novis = model_novis(noisy, t, obs_novis)
    if pred_novis.shape != (B, T, D):
        print(f"  ❌ No-vision mode: expected ({B},{T},{D}), got {pred_novis.shape}")
        errors += 1
    else:
        print(f"  ✅ No-vision mode: shape {list(pred_novis.shape)}")

    # 3c. Gradient flow
    loss = F.mse_loss(pred, torch.randn_like(pred))
    loss.backward()
    params_with_grad = sum(1 for p in model.parameters()
                           if p.grad is not None and p.grad.norm().item() > 1e-12)
    if params_with_grad < 50:
        print(f"  ❌ Only {params_with_grad} params with grad (expected 50+)")
        errors += 1
    else:
        print(f"  ✅ Gradient flow: {params_with_grad} params with grad ≠ 0")

    # 3d. DDIM inference
    scheduler = DDPMScheduler(num_train_steps=100, schedule="cosine")
    model.eval()
    single_obs = {k: v[:1] for k, v in obs.items()}
    with torch.no_grad():
        action = model.predict_action(single_obs, scheduler, num_inference_steps=10, device=device)
    if action.shape != (1, T, D):
        print(f"  ❌ Inference shape: expected (1,{T},{D}), got {list(action.shape)}")
        errors += 1
    else:
        print(f"  ✅ DDIM inference (10 steps): output {list(action.shape)}")
        print(f"     action range: [{action.min().item():.3f}, {action.max().item():.3f}]")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"     model size: {n_params:,} params ({n_params * 4 / 1024**2:.1f} MB)")

    print(f"  [{errors} errors]\n")
    return errors


# ===========================================================================
# Test 4: Mini training loop with real data
# ===========================================================================


def test_mini_training(data_dir: str = "data/raw") -> int:
    """Run 5 training steps with real collected data to verify pipeline."""
    print("─" * 50)
    print("Test 4: Mini Training Loop (real data)")
    print("─" * 50)

    from models.diffusion_policy import DiffusionPolicy
    from models.noise_scheduler import DDPMScheduler
    from vl_data.dataset import EpisodicDataset

    errors = 0
    device = "cpu"
    batch_size = 8

    # --- Load data ---
    data_path = Path(data_dir)
    if not data_path.exists() or not list(data_path.glob("episode_*.h5")):
        print(f"  ⚠️  No HDF5 files in {data_dir} — skipping real data test")
        print(f"  [0 errors, skipped]\n")
        return 0

    ds = EpisodicDataset(
        data_dir,
        obs_horizon=2,
        action_horizon=16,
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=ds.collate_fn,
        drop_last=True,
    )
    print(f"  Dataset: {len(ds)} samples from {len(ds._episode_files)} episodes")

    # --- Create model + scheduler ---
    model = DiffusionPolicy(
        action_dim=7,
        action_horizon=16,
        use_vision=False,  # RGB is all zeros — skip vision for now
    ).to(device)
    model.train()

    scheduler = DDPMScheduler(num_train_steps=100, schedule="cosine", device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-6)

    # --- Training steps ---
    n_steps = 5
    losses = []
    t0 = time.monotonic()

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= n_steps:
            break

        # Prepare
        actions = batch["actions"].to(device)  # (B, 16, 7)
        B = actions.shape[0]

        # Remove the last observation frame dimension
        # Dataset returns obs with shape (B, obs_horizon, D)
        # We need flat dict for the model
        obs_flat = {}
        for key in batch["observations"]:
            obs_flat[key] = batch["observations"][key].to(device)

        # Sample noise and timestep
        noise = torch.randn_like(actions)
        timesteps = torch.randint(0, scheduler.num_train_steps, (B,),
                                   device=device, dtype=torch.long)

        # Forward diffusion
        noisy_actions = scheduler.add_noise(actions, noise, timesteps)

        # Predict noise
        pred_noise = model(noisy_actions, timesteps, obs_flat)

        # Loss
        loss = F.mse_loss(pred_noise, noise)
        losses.append(loss.item())

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if batch_idx == 0:
            print(f"  Step {batch_idx + 1}: loss={loss.item():.4f}")
        elif batch_idx == n_steps - 1:
            print(f"  Step {batch_idx + 1}: loss={loss.item():.4f}")

    t1 = time.monotonic()

    # --- Validate ---
    if len(losses) < 2:
        print("  ❌ Not enough training steps")
        errors += 1
    else:
        # Check loss is finite
        if not all(np.isfinite(losses)):
            print(f"  ❌ Loss contains NaN/Inf: {losses}")
            errors += 1
        else:
            print(f"  ✅ All losses finite: {[f'{l:.4f}' for l in losses]}")

        # Check loss decreases or is reasonable
        if losses[0] < 0.01 or losses[0] > 10.0:
            print(f"  ⚠️  Initial loss {losses[0]:.4f} is outside expected range [0.01, 10.0]")
        else:
            print(f"  ✅ Initial loss in expected range: {losses[0]:.4f}")

    print(f"  Time: {t1 - t0:.1f}s for {n_steps} steps "
          f"({(t1 - t0) / n_steps * 1000:.0f} ms/step)")

    print(f"  [{errors} errors]\n")
    return errors


# ===========================================================================
# Test 5: Dataset statistics
# ===========================================================================


def test_dataset_stats(data_dir: str = "data/raw") -> int:
    """Print summary statistics of the collected expert data."""
    print("─" * 50)
    print("Test 5: Dataset Statistics")
    print("─" * 50)

    from vl_data.dataset import EpisodicDataset

    errors = 0
    data_path = Path(data_dir)

    if not data_path.exists() or not list(data_path.glob("episode_*.h5")):
        print(f"  ⚠️  No data in {data_dir} — skipping\n")
        return 0

    ds = EpisodicDataset(data_dir, obs_horizon=2, action_horizon=16)
    stats = ds.compute_statistics()

    print(f"  Episodes: {len(ds._episode_files)}")
    print(f"  Training samples: {len(ds)}")
    print()
    print(f"  {'Field':<22s} {'Min':>10s} {'Max':>10s} {'Mean':>10s} {'Std':>10s}")
    print(f"  {'─' * 22} {'─' * 10} {'─' * 10} {'─' * 10} {'─' * 10}")

    for key, s in stats.items():
        print(f"  {key:<22s} {s['min']:>10.4f} {s['max']:>10.4f} "
              f"{s['mean']:>10.4f} {s['std']:>10.4f}")

    ds.close()
    print(f"  [0 errors]\n")
    return errors


# ===========================================================================
# Main
# ===========================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="Day 6 — Model Unit Tests")
    parser.add_argument("--data-dir", type=str, default="data/raw",
                        help="Path to collected HDF5 data")
    args = parser.parse_args()

    print("=" * 50)
    print("  VLARM — Day 6: Model Unit Tests")
    print("=" * 50)
    print(f"  Device:     {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"  PyTorch:    {torch.__version__}")
    print()

    total_errors = 0
    total_errors += test_noise_scheduler()
    total_errors += test_vision_encoder()
    total_errors += test_diffusion_policy()
    total_errors += test_mini_training(args.data_dir)
    total_errors += test_dataset_stats(args.data_dir)

    print("=" * 50)
    if total_errors == 0:
        print("  ✅ All tests passed!")
    else:
        print(f"  ❌ {total_errors} test(s) failed")
    print("=" * 50)


if __name__ == "__main__":
    main()
