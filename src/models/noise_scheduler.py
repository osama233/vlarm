#!/usr/bin/env python3
"""VLARM Noise Scheduler — DDPM / DDIM noise schedules and sampling.

Supports linear, cosine, and squared-cosine beta schedules.  Provides both
DDPM (stochastic) and DDIM (deterministic, fewer steps) reverse processes.

Reference
---------
* DDPM: Ho et al., "Denoising Diffusion Probabilistic Models" (2020)
* DDIM: Song et al., "Denoising Diffusion Implicit Models" (2021)
* Diffusion Policy: Chi et al., "Diffusion Policy" (2023)

Usage (training)::

    scheduler = DDPMScheduler(num_train_steps=100, schedule="cosine")
    noise = torch.randn_like(actions)                  # ε ~ N(0, I)
    t = torch.randint(0, 100, (B,))                     # random timesteps
    noisy = scheduler.add_noise(actions, noise, t)      # a_t
    pred = model(noisy, t, cond)                        # ε_θ
    loss = F.mse_loss(pred, noise)

Usage (inference, DDIM 16 steps)::

    scheduler = DDPMScheduler(num_train_steps=100, schedule="cosine")
    x = torch.randn(B, action_horizon, action_dim)     # a_T ~ N(0, I)
    for t in scheduler.sample_steps(16):
        pred = model(x, t, cond)
        x = scheduler.ddim_step(pred, t, x)
    return x  # a_0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F


# ===========================================================================
# Beta schedules
# ===========================================================================


def _linear_beta_schedule(num_steps: int, beta_start: float = 1e-4,
                          beta_end: float = 0.02) -> torch.Tensor:
    """Linearly spaced betas from ``beta_start`` to ``beta_end``."""
    return torch.linspace(beta_start, beta_end, num_steps, dtype=torch.float32)


def _cosine_beta_schedule(num_steps: int, s: float = 0.008) -> torch.Tensor:
    """Cosine schedule as proposed in "Improved DDPM" (Nichol & Dhariwal, 2021).

    ``s`` is a small offset to prevent β_t from being too small near t=0.
    """
    steps = torch.arange(num_steps + 1, dtype=torch.float32)
    x = (steps / num_steps + s) / (1.0 + s) * 0.5 * np.pi
    alpha_bar = torch.cos(x) ** 2
    # Normalise so ᾱ_0 ≈ 1
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1.0 - alpha_bar[1:] / alpha_bar[:-1]
    return torch.clamp(betas, max=0.999)


def _squared_cosine_beta_schedule(num_steps: int, s: float = 0.008) -> torch.Tensor:
    """Squared-cosine variant — smoother decay than vanilla cosine."""
    steps = torch.arange(num_steps + 1, dtype=torch.float32)
    x = (steps / num_steps + s) / (1.0 + s) * 0.5 * np.pi
    alpha_bar = torch.cos(x) ** 4  # squared cosine → sharper drop-off
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1.0 - alpha_bar[1:] / alpha_bar[:-1]
    return torch.clamp(betas, max=0.999)


_SCHEDULES = {
    "linear": _linear_beta_schedule,
    "cosine": _cosine_beta_schedule,
    "squared_cosine": _squared_cosine_beta_schedule,
}

# ===========================================================================
# Scheduler state (pre-computed values)
# ===========================================================================


@dataclass
class _ScheduleState:
    """Pre-computed diffusion schedule tensors.

    All tensors are 1-D of length ``num_train_steps`` unless noted otherwise.
    """

    num_train_steps: int
    betas: torch.Tensor           # β_t      — noise variance per step
    alphas: torch.Tensor          # α_t = 1 - β_t
    alphas_cumprod: torch.Tensor  # ᾱ_t      — cumulative product of α
    alphas_cumprod_prev: torch.Tensor  # ᾱ_{t-1}  (padded with 1.0 at t=0)
    sqrt_alphas_cumprod: torch.Tensor      # √(ᾱ_t)
    sqrt_one_minus_alphas_cumprod: torch.Tensor  # √(1 - ᾱ_t)
    sqrt_recip_alphas: torch.Tensor        # 1 / √(α_t)
    posterior_variance: torch.Tensor       # σ_t²  — DDPM posterior variance


# ===========================================================================
# DDPMScheduler
# ===========================================================================


class DDPMScheduler:
    """DDPM noise scheduler with DDIM sampling support.

    Parameters
    ----------
    num_train_steps : int
        Number of diffusion steps T (default 100, as in Diffusion Policy).
    schedule : str
        Beta schedule name: ``"linear"``, ``"cosine"``, or ``"squared_cosine"``.
    beta_start : float
        Starting β for linear schedule.
    beta_end : float
        Ending β for linear schedule.
    device : str or torch.device
        Device for pre-computed tensors (default ``"cpu"``).
    """

    def __init__(
        self,
        num_train_steps: int = 100,
        schedule: str = "cosine",
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        device: str | torch.device = "cpu",
    ) -> None:
        if schedule not in _SCHEDULES:
            raise ValueError(
                f"Unknown schedule '{schedule}'.  Choose from {list(_SCHEDULES)}."
            )

        self._num_train_steps = num_train_steps
        self._schedule_name = schedule

        # Build betas
        build_fn = _SCHEDULES[schedule]
        if schedule == "linear":
            betas = build_fn(num_train_steps, beta_start, beta_end).to(device)
        else:
            betas = build_fn(num_train_steps).to(device)

        # Pre-compute all derived values
        state = self._precompute(betas)
        self._state = state
        self._device = device

    @staticmethod
    def _precompute(betas: torch.Tensor) -> _ScheduleState:
        T = betas.shape[0]
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        # ᾱ_{t-1} — padded at front with 1.0
        alphas_cumprod_prev = torch.cat([
            torch.ones(1, dtype=betas.dtype, device=betas.device),
            alphas_cumprod[:-1],
        ])

        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

        # DDPM posterior variance σ_t²
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

        return _ScheduleState(
            num_train_steps=T,
            betas=betas,
            alphas=alphas,
            alphas_cumprod=alphas_cumprod,
            alphas_cumprod_prev=alphas_cumprod_prev,
            sqrt_alphas_cumprod=sqrt_alphas_cumprod,
            sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod,
            sqrt_recip_alphas=sqrt_recip_alphas,
            posterior_variance=posterior_variance,
        )

    # ------------------------------------------------------------------
    # Training: add noise
    # ------------------------------------------------------------------

    def add_noise(
        self,
        original: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Forward diffusion:  a_t = √(ᾱ_t)·a_0 + √(1-ᾱ_t)·ε

        Parameters
        ----------
        original : (B, ...) float32
            Clean action sequence a_0.
        noise : (B, ...) float32
            Gaussian noise ε ~ N(0, I).  Must match ``original`` shape.
        timesteps : (B,) int64
            Diffusion timesteps t ∈ [0, T-1].

        Returns
        -------
        noisy : (B, ...) float32
            Noised action sequence a_t.
        """
        s = self._state
        shape = [original.shape[0]] + [1] * (original.ndim - 1)

        sqrt_alpha = s.sqrt_alphas_cumprod.to(original.device)
        sqrt_one_alpha = s.sqrt_one_minus_alphas_cumprod.to(original.device)

        a = sqrt_alpha[timesteps].view(shape)
        b = sqrt_one_alpha[timesteps].view(shape)
        return a * original + b * noise

    # ------------------------------------------------------------------
    # Inference: DDPM reverse step (stochastic)
    # ------------------------------------------------------------------

    def ddpm_step(
        self,
        predicted_noise: torch.Tensor,
        timestep: int,
        current_x: torch.Tensor,
    ) -> torch.Tensor:
        """Single DDPM reverse step:  a_{t-1} = 1/√(α_t) · (a_t - β_t/√(1-ᾱ_t)·ε) + σ_t·z

        Parameters
        ----------
        predicted_noise : (B, ...)
            Model prediction ε_θ(a_t, t, obs).
        timestep : int
            Current timestep t (scalar).
        current_x : (B, ...)
            Current noisy sample a_t.

        Returns
        -------
        prev_x : (B, ...)
            a_{t-1}.
        """
        s = self._state
        device = current_x.device

        if timestep <= 0:
            return current_x

        t = timestep

        # 1/√(α_t)
        alpha_inv = s.sqrt_recip_alphas[t].to(device)
        # β_t / √(1-ᾱ_t)
        beta_over_sqrt = s.betas[t] / s.sqrt_one_minus_alphas_cumprod[t]
        beta_over_sqrt = beta_over_sqrt.to(device)

        # Predicted x_0
        pred_x0 = alpha_inv * current_x - beta_over_sqrt * predicted_noise

        # Direction to x_{t-1}
        alpha_cumprod_prev = s.alphas_cumprod_prev[t].to(device)
        pred_dir = torch.sqrt(alpha_cumprod_prev) * pred_x0

        # Random noise
        if t > 1:
            var = s.posterior_variance[t].to(device)
            z = torch.randn_like(current_x)
            pred_dir = pred_dir + torch.sqrt(var) * z

        return pred_dir

    # ------------------------------------------------------------------
    # Inference: DDIM reverse step (deterministic, accelerated)
    # ------------------------------------------------------------------

    def ddim_step(
        self,
        predicted_noise: torch.Tensor,
        timestep: int,
        next_timestep: int,
        current_x: torch.Tensor,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """Single DDIM reverse step.

        When ``eta=0`` the step is fully deterministic.  Set ``eta=1`` to
        recover DDPM-like stochasticity.

        Parameters
        ----------
        predicted_noise : (B, ...)
            Model prediction ε_θ(a_t, t, obs).
        timestep : int
            Current timestep t.
        next_timestep : int
            Next timestep t' (< t).
        current_x : (B, ...)
            Current noisy sample a_t.
        eta : float
            Stochasticity parameter (0 = deterministic, 1 = DDPM-like).

        Returns
        -------
        prev_x : (B, ...)
            a_{t'}.
        """
        s = self._state
        device = current_x.device

        t = timestep
        t_next = next_timestep

        # ᾱ_t, ᾱ_{t'}
        alpha_t = s.alphas_cumprod[t].to(device)
        alpha_next = s.alphas_cumprod[t_next].to(device) if t_next >= 0 else torch.tensor(1.0, device=device)

        # --- Numerically stable DDIM ---
        # When ᾱ_t ≈ 0, the naive formula x̂_0 = (x_t - √(1-ᾱ_t)·ε_θ)/√(ᾱ_t)
        # divides by near-zero, amplifying noise prediction errors by >1000×.
        # We stabilise by combining the two DDIM terms into a single update
        # that avoids explicit division by sqrt(ᾱ_t) when it is very small.
        #
        # Standard DDIM (η=0):
        #   x_{t'} = √(ᾱ_{t'})/√(ᾱ_t) · x_t
        #          + ( √(1-ᾱ_{t'}) - √(ᾱ_{t'})·√(1-ᾱ_t)/√(ᾱ_t) ) · ε_θ
        #
        # For numerical stability, clip the ratio √(ᾱ_{t'})/√(ᾱ_t).

        sqrt_alpha_t = torch.sqrt(torch.clamp(alpha_t, min=1e-8))
        sqrt_one_alpha_t = torch.sqrt(1.0 - alpha_t)
        sqrt_alpha_next = torch.sqrt(alpha_next)
        sqrt_one_alpha_next = torch.sqrt(1.0 - alpha_next)

        # Coefficient for x_t — may be clipped for stability
        coeff_x = sqrt_alpha_next / sqrt_alpha_t
        # Clip to prevent explosion at early DDIM steps (cosine: ᾱ_T ≈ 0)
        coeff_x = torch.clamp(coeff_x, max=50.0)

        # Coefficient for ε_θ
        coeff_eps = sqrt_one_alpha_next - coeff_x * sqrt_one_alpha_t

        # DDIM update
        pred_dir = coeff_x * current_x + coeff_eps * predicted_noise

        # Random noise (eta > 0 — stochastic DDIM)
        if eta > 0 and t_next >= 0:
            sigma = eta * torch.sqrt((1.0 - alpha_next) / (1.0 - alpha_t)) \
                * torch.sqrt(1.0 - alpha_t / alpha_next)
            z = torch.randn_like(current_x)
            pred_dir = pred_dir + sigma * z

        return pred_dir

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def sample_steps(self, num_inference_steps: int) -> list[int]:
        """Return DDIM sampling schedule: evenly-spaced timesteps, descending.

        Parameters
        ----------
        num_inference_steps : int
            Number of DDIM steps (typ. 10–16).

        Returns
        -------
        steps : list[int]
            Timesteps from T-1 down to 0, evenly spaced.
        """
        T = self._num_train_steps
        step_ratio = T / num_inference_steps
        # Reverse order: T-1, T-1 - step_ratio, ..., 0
        raw = np.arange(0, num_inference_steps) * step_ratio
        raw = np.round(raw).astype(int)
        return (T - 1 - raw).tolist()

    def ddpm_loop(
        self,
        model_fn,
        shape: tuple[int, ...],
        cond: dict[str, torch.Tensor] | torch.Tensor,
        device: str | torch.device = "cpu",
    ) -> torch.Tensor:
        """Full DDPM sampling loop (all T steps, stochastic).

        Slower than DDIM (100 steps vs 16) but numerically stable — does not
        suffer from the division-by-near-zero issue when ᾱ_T ≈ 0.

        Parameters
        ----------
        model_fn : callable
            Signature: ``model_fn(x, timestep_tensor, cond) -> predicted_noise``.
        shape : tuple
            Desired output shape, e.g. ``(B, action_horizon, action_dim)``.
        cond : Tensor or dict
            Observation conditioning (passed through to ``model_fn``).
        device : str or torch.device

        Returns
        -------
        x_0 : Tensor
            Denoised action prediction.
        """
        x = torch.randn(shape, device=device, dtype=torch.float32)
        for t in range(self._num_train_steps - 1, -1, -1):
            t_tensor = torch.full((shape[0],), t, device=device, dtype=torch.long)
            predicted_noise = model_fn(x, t_tensor, cond)
            x = self.ddpm_step(predicted_noise, t, x)
        return x

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_train_steps(self) -> int:
        return self._num_train_steps

    @property
    def device(self) -> str | torch.device:
        return self._device

    def to(self, device: str | torch.device) -> "DDPMScheduler":
        """Move all pre-computed tensors to a new device (returns self)."""
        self._device = device if isinstance(device, str) else str(device)
        s = self._state
        for field_name in s.__dataclass_fields__:
            t = getattr(s, field_name)
            if isinstance(t, torch.Tensor):
                setattr(s, field_name, t.to(device))
        return self


# ===========================================================================
# Self-test
# ===========================================================================


def _test() -> None:
    """Quick smoke test of the scheduler."""
    print("=== Noise Scheduler Smoke Test ===")

    for schedule in ("linear", "cosine", "squared_cosine"):
        sch = DDPMScheduler(num_train_steps=100, schedule=schedule)
        print(f"\n{schedule}: T={sch.num_train_steps}")
        print(f"  β range: [{sch._state.betas[0]:.4f}, {sch._state.betas[-1]:.4f}]")
        print(f"  ᾱ_T:     {sch._state.alphas_cumprod[-1]:.6f}  (should be ≈ 0)")

        # Training: add noise
        B, Ta, Da = 4, 16, 7
        x0 = torch.randn(B, Ta, Da)
        noise = torch.randn(B, Ta, Da)
        t = torch.randint(0, 100, (B,))

        xt = sch.add_noise(x0, noise, t)
        assert xt.shape == (B, Ta, Da), f"Wrong shape: {xt.shape}"

        # SNR should decrease with t
        snr_early = sch._state.sqrt_alphas_cumprod[10] / sch._state.sqrt_one_minus_alphas_cumprod[10]
        snr_late = sch._state.sqrt_alphas_cumprod[90] / sch._state.sqrt_one_minus_alphas_cumprod[90]
        print(f"  SNR(t=10): {snr_early:.2f}, SNR(t=90): {snr_late:.4f}  (late should be << early)")
        assert snr_late < snr_early, "SNR should decrease with t"

        # DDIM steps
        ddim_steps = sch.sample_steps(16)
        print(f"  DDIM 16 steps: {ddim_steps[:4]}...{ddim_steps[-4:]}")
        assert len(ddim_steps) == 16
        assert ddim_steps[0] == 99  # T-1

    print("\n✅ All scheduler tests passed!")


if __name__ == "__main__":
    _test()
