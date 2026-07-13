#!/usr/bin/env python3
"""VLARM Diffusion Policy — 1D Conv U-Net with observation conditioning.

Implements the CNN-based Diffusion Policy from Chi et al. (2023):
a 1-D convolutional U-Net that denoises action trajectories conditioned
on visual features and robot state.

Architecture overview::

    rgb (B, H, W, 3) ──→ VisionEncoder ──→ (B, 512)
    state (B, S)      ──→ StateEncoder  ──→ (B, 256)
                                              │
                                   Concat → obs_embed (B, 768)
                                              │
    noisy_action (B, T, D) ──→ 1D Conv U-Net ←── FiLM(obs_embed + time_embed)
                                              │
                                   predicted_noise (B, T, D)

Training::

    model = DiffusionPolicy(action_dim=7, action_horizon=16)
    noisy = scheduler.add_noise(actions, noise, t)
    pred_noise = model(noisy, t, obs)
    loss = F.mse_loss(pred_noise, noise)

Inference (DDIM, 16 steps)::

    x = torch.randn(B, T, D)
    for t in scheduler.sample_steps(16):
        eps = model(x, t_full, obs)
        x = scheduler.ddim_step(eps, t, t_next, x)
    return x   # predicted action trajectory
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.vision_encoder import ResNet18Encoder


# ===========================================================================
# Sinusoidal timestep embedding
# ===========================================================================


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal position embedding for diffusion timesteps.

    Maps integer timesteps t ∈ [0, T-1] to continuous vectors via
    sin/cos at different frequencies, as in "Attention Is All You Need".

    Parameters
    ----------
    dim : int
        Output embedding dimension (must be even).
    max_period : int
        Maximum period (default 10_000 as in transformers).
    """

    def __init__(self, dim: int = 128, max_period: int = 10_000) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"dim must be even, got {dim}")
        self._dim = dim
        self._max_period = max_period

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Embed integer timesteps.

        Parameters
        ----------
        timesteps : (B,) int64
            Diffusion timesteps.

        Returns
        -------
        embed : (B, dim) float32
            Sinusoidal embedding.
        """
        half = self._dim // 2
        freq = torch.exp(
            -math.log(self._max_period) * torch.arange(half, dtype=torch.float32,
                                                         device=timesteps.device) / half
        )
        args = timesteps.float().unsqueeze(1) * freq.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=1)


# ===========================================================================
# FiLM-conditioned 1-D Convolution block
# ===========================================================================


class _FiLMConv1dBlock(nn.Module):
    """Conv1d → GroupNorm → FiLM → ReLU (repeated twice, as in U-Net)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        kernel_size: int = 3,
        num_groups: int = 8,
    ) -> None:
        super().__init__()
        self._conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                                 padding=kernel_size // 2)
        self._gn1 = nn.GroupNorm(min(num_groups, out_channels), out_channels)
        self._film1 = nn.Linear(cond_dim, out_channels * 2)

        self._conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                                 padding=kernel_size // 2)
        self._gn2 = nn.GroupNorm(min(num_groups, out_channels), out_channels)
        self._film2 = nn.Linear(cond_dim, out_channels * 2)

        # Zero-init FiLM layers for identity at start of training
        nn.init.zeros_(self._film1.weight)
        nn.init.zeros_(self._film1.bias)
        nn.init.zeros_(self._film2.weight)
        nn.init.zeros_(self._film2.bias)

        # Residual projection if channel count changes
        if in_channels != out_channels:
            self._residual = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self._residual = nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Apply two FiLM-conditioned Conv1d layers with residual connection.

        Parameters
        ----------
        x : (B, C_in, L) float32
        cond : (B, cond_dim) float32

        Returns
        -------
        out : (B, C_out, L) float32
        """
        residual = self._residual(x)

        # First conv
        h = self._conv1(x)
        h = self._gn1(h)
        h = self._apply_film(h, self._film1(cond))
        h = F.relu(h)

        # Second conv
        h = self._conv2(h)
        h = self._gn2(h)
        h = self._apply_film(h, self._film2(cond))
        h = F.relu(h + residual)

        return h

    @staticmethod
    def _apply_film(x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """Apply FiLM: out = (1 + γ)·x + β."""
        gamma, beta = params.chunk(2, dim=1)  # (B, C) each
        gamma = gamma.unsqueeze(-1)  # (B, C, 1)
        beta = beta.unsqueeze(-1)
        return (1.0 + gamma) * x + beta


# ===========================================================================
# 1-D Convolutional U-Net for action denoising
# ===========================================================================


class _ActionUNet(nn.Module):
    """1-D Conv U-Net that denoises action trajectories.

    Operates with ``Conv1d`` over the temporal dimension.  The action
    dimensionality is treated as channels:  (B, action_dim, action_horizon).

    Parameters
    ----------
    action_dim : int
        Number of action dimensions (default 7 = Franka arm joints).
    action_horizon : int
        Number of future action steps (default 16).
    cond_dim : int
        Dimension of the combined conditioning vector (obs_embed + time_embed).
    base_channels : int
        Channel count of the first U-Net block.
    """

    def __init__(
        self,
        action_dim: int = 7,
        action_horizon: int = 16,
        cond_dim: int = 768 + 128,
        base_channels: int = 64,
    ) -> None:
        super().__init__()
        self._action_dim = action_dim
        self._action_horizon = action_horizon

        ch = base_channels

        # --- Encoder (down-sampling) ---
        self._down1 = _FiLMConv1dBlock(action_dim, ch, cond_dim)
        self._pool1 = nn.MaxPool1d(2)  # T → T/2

        self._down2 = _FiLMConv1dBlock(ch, ch * 2, cond_dim)
        self._pool2 = nn.MaxPool1d(2)

        self._down3 = _FiLMConv1dBlock(ch * 2, ch * 4, cond_dim)
        self._pool3 = nn.MaxPool1d(2)

        # --- Bottleneck ---
        self._bottleneck = _FiLMConv1dBlock(ch * 4, ch * 4, cond_dim)

        # --- Decoder (up-sampling) ---
        self._up3 = nn.Upsample(scale_factor=2, mode="nearest")
        self._dec3 = _FiLMConv1dBlock(ch * 4 + ch * 4, ch * 2, cond_dim)

        self._up2 = nn.Upsample(scale_factor=2, mode="nearest")
        self._dec2 = _FiLMConv1dBlock(ch * 2 + ch * 2, ch, cond_dim)

        self._up1 = nn.Upsample(scale_factor=2, mode="nearest")
        self._dec1 = _FiLMConv1dBlock(ch + ch, ch, cond_dim)

        # --- Output ---
        self._out = nn.Conv1d(ch, action_dim, kernel_size=1)

        # Near-zero init for stable training start while allowing gradients
        nn.init.normal_(self._out.weight, mean=0.0, std=1e-6)
        nn.init.zeros_(self._out.bias)

    def forward(
        self,
        noisy_action: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """Denoise one step.

        Parameters
        ----------
        noisy_action : (B, T, D) float32
            Noisy action trajectory.
        cond : (B, cond_dim) float32
            Combined observation + time embedding.

        Returns
        -------
        noise : (B, T, D) float32
            Predicted noise ε_θ.
        """
        # (B, T, D) → (B, D, T) for Conv1d over time axis
        x = noisy_action.permute(0, 2, 1).contiguous()

        # Encoder
        d1 = self._down1(x, cond)
        d2 = self._down2(self._pool1(d1), cond)
        d3 = self._down3(self._pool2(d2), cond)

        # Bottleneck
        b = self._bottleneck(self._pool3(d3), cond)

        # Decoder with skip connections
        u3 = self._up3(b)
        u3 = self._dec3(torch.cat([u3, d3], dim=1), cond)

        u2 = self._up2(u3)
        u2 = self._dec2(torch.cat([u2, d2], dim=1), cond)

        u1 = self._up1(u2)
        u1 = self._dec1(torch.cat([u1, d1], dim=1), cond)

        # Output
        out = self._out(u1)  # (B, D, T)
        out = out.permute(0, 2, 1).contiguous()  # (B, T, D)

        return out


# ===========================================================================
# State encoder — small MLP
# ===========================================================================


class _StateEncoder(nn.Module):
    """Encode robot state (joint positions + EE pose + gripper) → fixed vector."""

    def __init__(self, state_dim: int = 13, hidden_dim: int = 256,
                 output_dim: int = 256) -> None:
        super().__init__()
        self._net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._net(x)


# ===========================================================================
# Diffusion Policy — top-level model
# ===========================================================================


class DiffusionPolicy(nn.Module):
    """Diffusion Policy for visuomotor action prediction.

    Combines a vision encoder, robot-state encoder, timestep embedding,
    and a 1-D convolutional U-Net into a single denoising model.

    Parameters
    ----------
    action_dim : int
        Action dimensionality (default 7 = Franka arm joints).
    action_horizon : int
        Number of future action steps to predict (default 16).
    obs_horizon : int
        Number of observation frames used for conditioning (default 2).
    state_dim : int
        Robot state dimension (joint_positions 9 + ee_position 3 +
        gripper_width 1 = 13).
    vision_output_dim : int
        Output dimension of the ResNet-18 encoder.
    state_output_dim : int
        Output dimension of the state encoder MLP.
    time_dim : int
        Dimension of the sinusoidal timestep embedding.
    unet_base_channels : int
        Base channel count for the 1-D Conv U-Net.
    use_vision : bool
        If False, skip vision encoder entirely (for training without cameras).
    pretrained_vision : bool
        If True, use ImageNet-pretrained ResNet-18 weights.
    """

    def __init__(
        self,
        action_dim: int = 7,
        action_horizon: int = 16,
        obs_horizon: int = 2,
        state_dim: int = 13,
        vision_output_dim: int = 512,
        state_output_dim: int = 256,
        time_dim: int = 128,
        unet_base_channels: int = 64,
        use_vision: bool = True,
        pretrained_vision: bool = False,
    ) -> None:
        super().__init__()

        self._action_dim = action_dim
        self._action_horizon = action_horizon
        self._obs_horizon = obs_horizon
        self._use_vision = use_vision
        self._time_dim = time_dim

        # --- Timestep embedding ---
        self._time_embed = SinusoidalTimeEmbedding(dim=time_dim)
        self._time_proj = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.ReLU(),
            nn.Linear(time_dim, time_dim),
        )

        # --- Vision encoder ---
        if use_vision:
            self._vision_encoder = ResNet18Encoder(
                output_dim=vision_output_dim,
                pretrained=pretrained_vision,
                use_film=False,
            )
        else:
            self._vision_encoder = None

        # --- State encoder ---
        self._state_encoder = _StateEncoder(
            state_dim=state_dim,
            hidden_dim=state_output_dim,
            output_dim=state_output_dim,
        )

        # --- Observation projection ---
        obs_total_dim = (vision_output_dim if use_vision else 0) + state_output_dim
        self._obs_proj = nn.Sequential(
            nn.Linear(obs_total_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
        )
        self._obs_total_dim = obs_total_dim

        # --- 1-D Conv U-Net ---
        # The U-Net receives conditioning = projected_obs + time_embed
        unet_cond_dim = 256 + time_dim
        self._unet = _ActionUNet(
            action_dim=action_dim,
            action_horizon=action_horizon,
            cond_dim=unet_cond_dim,
            base_channels=unet_base_channels,
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        noisy_action: torch.Tensor,
        timesteps: torch.Tensor,
        obs: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Predict the noise added to the action.

        Parameters
        ----------
        noisy_action : (B, action_horizon, action_dim) float32
            Noisy action trajectory a_t.
        timesteps : (B,) int64
            Diffusion timestep t for each batch item.
        obs : dict
            Observation dict with keys:
            - ``rgb``: (B, obs_horizon, H, W, 3) float32 (optional)
            - ``joint_positions``: (B, obs_horizon, 9) float32
            - ``ee_position``: (B, obs_horizon, 3) float32
            - ``gripper_width``: (B, obs_horizon, 1) float32

        Returns
        -------
        predicted_noise : (B, action_horizon, action_dim) float32
            Estimated noise ε_θ.
        """
        B = noisy_action.shape[0]

        # --- Time embedding ---
        t_embed = self._time_embed(timesteps)  # (B, time_dim)
        t_embed = self._time_proj(t_embed)     # (B, time_dim)

        # --- Vision features ---
        if self._use_vision and self._vision_encoder is not None:
            rgb = obs.get("rgb")
            if rgb is None:
                vis_feat = torch.zeros(B, 512, device=noisy_action.device)
            else:
                # Take the last observation frame (most recent)
                if rgb.ndim == 5:  # (B, obs_horizon, H, W, 3)
                    rgb = rgb[:, -1]
                # If batch items have different obs_horizon, handle gracefully
                vis_feat = self._vision_encoder(rgb)
        else:
            vis_feat = torch.zeros(B, 0, device=noisy_action.device)

        # --- State features ---
        state_parts = []
        for key in ("joint_positions", "ee_position", "gripper_width"):
            val = obs.get(key)
            if val is not None:
                if val.ndim == 3:  # (B, obs_horizon, D)
                    val = val[:, -1]  # last frame
                state_parts.append(val.reshape(B, -1))
        state_cat = torch.cat(state_parts, dim=1) if state_parts else \
            torch.zeros(B, 13, device=noisy_action.device)
        state_feat = self._state_encoder(state_cat)  # (B, 256)

        # --- Combined observation embedding ---
        if self._use_vision:
            obs_cat = torch.cat([vis_feat, state_feat], dim=1)
        else:
            obs_cat = state_feat
        obs_embed = self._obs_proj(obs_cat)  # (B, 256)

        # --- U-Net conditioning ---
        cond = torch.cat([obs_embed, t_embed], dim=1)  # (B, 256 + time_dim)

        # --- Denoise ---
        noise = self._unet(noisy_action, cond)
        return noise

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_action(
        self,
        obs: dict[str, torch.Tensor],
        scheduler,
        num_inference_steps: int = 100,
        device: str | torch.device = "cpu",
        use_ddpm: bool = True,
    ) -> torch.Tensor:
        """Full denoising loop: random noise → clean action trajectory.

        Uses DDPM sampling by default (numerically stable for cosine schedule
        where ᾱ_T ≈ 0).  Set ``use_ddpm=False`` for faster DDIM sampling
        (only advisable when the model is well-trained and ᾱ_T is not too
        close to zero).

        Parameters
        ----------
        obs : dict
            Observation conditioning (same format as ``forward()``).
        scheduler : DDPMScheduler
            Configured noise scheduler.
        num_inference_steps : int
            Number of denoising steps (100 for DDPM, 16 for DDIM).
        device : str or torch.device
        use_ddpm : bool
            If True, use DDPM (slow but stable).  If False, use DDIM (fast).

        Returns
        -------
        action : (1, action_horizon, action_dim) float32
            Predicted action trajectory a_0.
        """
        self.eval()
        B = 1  # single inference sample
        shape = (B, self._action_horizon, self._action_dim)

        if use_ddpm:
            return scheduler.ddpm_loop(self.forward, shape, obs, device=device)
        else:
            # DDIM path (fast but may diverge if ᾱ_T ≈ 0 and model is undertrained)
            steps = scheduler.sample_steps(num_inference_steps)
            x = torch.randn(shape, device=device, dtype=torch.float32)
            for i, t in enumerate(steps):
                t_tensor = torch.full((B,), t, device=device, dtype=torch.long)
                eps = self.forward(x, t_tensor, obs)
                t_next = steps[i + 1] if i + 1 < len(steps) else -1
                x = scheduler.ddim_step(eps, t, t_next, x, eta=0.0)
            return x

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def action_horizon(self) -> int:
        return self._action_horizon

    @property
    def obs_horizon(self) -> int:
        return self._obs_horizon


# ===========================================================================
# Self-test
# ===========================================================================


def _test() -> None:
    """Quick smoke test with random data and a full denoising loop."""
    print("=== Diffusion Policy Smoke Test ===")

    B, T, D = 4, 16, 7
    device = "cpu"

    # Build model
    model = DiffusionPolicy(
        action_dim=D,
        action_horizon=T,
        use_vision=True,
    ).to(device)
    model.train()

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Params: {n_params:,} total, {n_trainable:,} trainable")

    # Fake batch
    noisy = torch.randn(B, T, D, device=device)
    t = torch.randint(0, 100, (B,), device=device, dtype=torch.long)
    obs = {
        "rgb": torch.rand(B, 2, 480, 640, 3, device=device),
        "joint_positions": torch.randn(B, 2, 9, device=device),
        "ee_position": torch.randn(B, 2, 3, device=device),
        "gripper_width": torch.rand(B, 2, 1, device=device),
    }

    # Forward
    pred = model(noisy, t, obs)
    print(f"  Forward:     noisy={list(noisy.shape)} → pred={list(pred.shape)}")
    assert pred.shape == (B, T, D), f"Expected ({B},{T},{D}), got {pred.shape}"

    # Loss
    loss = F.mse_loss(pred, torch.randn_like(pred))
    print(f"  MSE loss:    {loss.item():.4f}")
    assert torch.isfinite(loss), "Loss should be finite"

    # No-vision mode
    model_novis = DiffusionPolicy(action_dim=D, action_horizon=T, use_vision=False)
    pred_novis = model_novis(noisy, t, {k: v for k, v in obs.items() if k != "rgb"})
    assert pred_novis.shape == (B, T, D)

    # Inference
    from models.noise_scheduler import DDPMScheduler
    scheduler = DDPMScheduler(num_train_steps=100, schedule="cosine")
    single_obs = {k: v[:1] for k, v in obs.items()}
    action = model.predict_action(single_obs, scheduler, num_inference_steps=10, device=device)
    print(f"  Inference:   DDIM 10 steps → action={list(action.shape)}")
    assert action.shape == (1, T, D)

    # Gradient check
    loss.backward()
    grad_norms = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            g = p.grad.norm().item()
            if g > 0:
                grad_norms[name] = g
    print(f"  Gradients:   {len(grad_norms)} params with grad ≠ 0")
    assert len(grad_norms) > 0, "No gradients!"

    print("\n✅ All diffusion policy tests passed!")


if __name__ == "__main__":
    _test()
