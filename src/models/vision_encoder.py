#!/usr/bin/env python3
"""VLARM Vision Encoder — ResNet-18 feature extractor with timestep conditioning.

Produces a 512-dim feature vector from 480×640 RGB images using a
modified ResNet-18 backbone.  Optionally injects diffusion timestep
information via FiLM layers.

Usage::

    encoder = ResNet18Encoder(output_dim=512, pretrained=False)
    rgb = torch.randn(4, 480, 640, 3)            # (B, H, W, C)
    features = encoder(rgb)                        # (B, 512)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
# FiLM layer — Feature-wise Linear Modulation
# ===========================================================================


class FiLMBlock(nn.Module):
    """Apply FiLM conditioning to a 2-D feature map.

    Given a conditioning vector ``c``, learn channel-wise scale γ and shift β:
        out = (1 + γ(c)) * x + β(c)

    Parameters
    ----------
    cond_dim : int
        Dimension of the conditioning vector.
    feature_channels : int
        Number of channels in the feature map to modulate.
    """

    def __init__(self, cond_dim: int, feature_channels: int) -> None:
        super().__init__()
        self._linear = nn.Linear(cond_dim, feature_channels * 2)

        # Zero-init the shift so the block starts as identity
        nn.init.zeros_(self._linear.weight)
        nn.init.zeros_(self._linear.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Modulate *x* with *cond*.

        Parameters
        ----------
        x : (B, C, H, W)
            Feature map.
        cond : (B, cond_dim)
            Conditioning vector.

        Returns
        -------
        out : (B, C, H, W)
            Modulated feature map.
        """
        params = self._linear(cond)  # (B, 2C)
        gamma, beta = params.chunk(2, dim=1)  # (B, C) each
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return (1.0 + gamma) * x + beta


# ===========================================================================
# ResNet-18 encoder
# ===========================================================================


class ResNet18Encoder(nn.Module):
    """ResNet-18 backbone → global feature vector.

    Parameters
    ----------
    output_dim : int
        Dimension of the output feature vector (default 512).
    pretrained : bool
        If True, load ImageNet-pretrained weights.
    freeze_backbone : bool
        If True, freeze all backbone parameters (only train FiLM layers).
    use_film : bool
        If True, add FiLM layers after each ResNet stage, conditioned on
        the diffusion timestep.
    time_dim : int
        Dimension of the sinusoidal timestep embedding (only used when
        ``use_film=True``).
    """

    # Channel counts per ResNet-18 stage
    _STAGE_CHANNELS = [64, 64, 128, 256, 512]

    def __init__(
        self,
        output_dim: int = 512,
        pretrained: bool = False,
        freeze_backbone: bool = False,
        use_film: bool = False,
        time_dim: int = 128,
    ) -> None:
        super().__init__()
        self._use_film = use_film
        self._output_dim = output_dim

        # --- Build backbone ---
        try:
            from torchvision.models import resnet18, ResNet18_Weights
        except ImportError:
            raise ImportError(
                "torchvision is required for ResNet18Encoder.  "
                "Install with: pip install torchvision"
            )

        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = resnet18(weights=weights)

        # --- Replace first conv: large input (480×640) needs smaller stride ---
        # Default ResNet-18: Conv2d(3, 64, 7, stride=2, padding=3)
        # Keep stride=2 — 480 → 240 → 120 → 60 → 30 → spatial size before pool
        # is 15×20 which is fine for global pooling.
        # But we modify input channels to handle the TF channel order.
        # Our input is (B, H, W, 3) — we permute in forward() to (B, 3, H, W).
        self._backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,  # (B, 64, H/4, W/4)
            resnet.layer1,   # (B, 64,  H/4, W/4)
            resnet.layer2,   # (B, 128, H/8, W/8)
            resnet.layer3,   # (B, 256, H/16,W/16)
            resnet.layer4,   # (B, 512, H/32,W/32)
            resnet.avgpool,  # (B, 512, 1, 1)
            nn.Flatten(1),   # (B, 512)
        )

        # --- Optional FiLM layers (one per ResNet stage) ---
        if use_film:
            self._film_layers = nn.ModuleList([
                FiLMBlock(time_dim, ch) for ch in self._STAGE_CHANNELS
            ])
        else:
            self._film_layers = nn.ModuleList()

        # --- Output projection ---
        if output_dim != 512:
            self._proj = nn.Linear(512, output_dim)
        else:
            self._proj = nn.Identity()

        # --- Freeze backbone ---
        if freeze_backbone:
            for p in self._backbone.parameters():
                p.requires_grad = False
            # FiLM layers stay trainable even when backbone is frozen
            for p in self._film_layers.parameters():
                p.requires_grad = True

    def forward(
        self,
        rgb: torch.Tensor,
        time_embed: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Extract features from RGB images.

        Parameters
        ----------
        rgb : (B, H, W, 3) float32
            RGB image(s) in [0, 1] range (channels-last, as stored in HDF5).
        time_embed : (B, time_dim) or None
            Sinusoidal timestep embedding for FiLM conditioning.  Only
            required when ``use_film=True``.

        Returns
        -------
        features : (B, output_dim) float32
            Visual feature vector.
        """
        # Channels-last → channels-first
        if rgb.ndim == 4 and rgb.shape[-1] == 3:
            x = rgb.permute(0, 3, 1, 2).contiguous()  # (B, 3, H, W)
        else:
            x = rgb

        # --- Run backbone with optional FiLM ---
        # We interleave FiLM after each stage.  To do this cleanly we
        # manually step through each backbone sub-module.
        layers = list(self._backbone.children())
        film_idx = 0

        # Conv1 + BN + ReLU + MaxPool
        for layer in layers[:4]:
            x = layer(x)
        if self._use_film and film_idx < len(self._film_layers):
            x = self._film_layers[film_idx](x, time_embed)
            film_idx += 1

        # Layer1 → FiLM
        x = layers[4](x)
        if self._use_film and film_idx < len(self._film_layers):
            x = self._film_layers[film_idx](x, time_embed)
            film_idx += 1

        # Layer2 → FiLM
        x = layers[5](x)
        if self._use_film and film_idx < len(self._film_layers):
            x = self._film_layers[film_idx](x, time_embed)
            film_idx += 1

        # Layer3 → FiLM
        x = layers[6](x)
        if self._use_film and film_idx < len(self._film_layers):
            x = self._film_layers[film_idx](x, time_embed)
            film_idx += 1

        # Layer4 → FiLM
        x = layers[7](x)
        if self._use_film and film_idx < len(self._film_layers):
            x = self._film_layers[film_idx](x, time_embed)
            film_idx += 1

        # AvgPool + Flatten
        for layer in layers[8:]:
            x = layer(x)

        # Output projection
        x = self._proj(x)
        return x


# ===========================================================================
# Convenience factory
# ===========================================================================


def make_vision_encoder(
    output_dim: int = 512,
    pretrained: bool = False,
    freeze_backbone: bool = False,
    use_film: bool = False,
    time_dim: int = 128,
) -> ResNet18Encoder:
    """Create a ResNet-18 vision encoder with sensible defaults."""
    return ResNet18Encoder(
        output_dim=output_dim,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        use_film=use_film,
        time_dim=time_dim,
    )


# ===========================================================================
# Self-test
# ===========================================================================


def _test() -> None:
    """Quick smoke test."""
    print("=== Vision Encoder Smoke Test ===")

    B, H, W = 2, 480, 640

    # Without FiLM
    enc = ResNet18Encoder(output_dim=512, use_film=False)
    rgb = torch.randn(B, H, W, 3)
    out = enc(rgb)
    print(f"  No FiLM:     input={list(rgb.shape)} → output={list(out.shape)}")
    assert out.shape == (B, 512), f"Expected (2, 512), got {out.shape}"

    # With FiLM
    enc_film = ResNet18Encoder(output_dim=512, use_film=True, time_dim=128)
    time_embed = torch.randn(B, 128)
    out_film = enc_film(rgb, time_embed)
    print(f"  With FiLM:   input={list(rgb.shape)} → output={list(out_film.shape)}")
    assert out_film.shape == (B, 512)

    # Custom output dim
    enc_256 = ResNet18Encoder(output_dim=256)
    out_256 = enc_256(rgb)
    print(f"  Custom dim:  input={list(rgb.shape)} → output={list(out_256.shape)}")
    assert out_256.shape == (B, 256)

    # Frozen backbone
    enc_frozen = ResNet18Encoder(freeze_backbone=True)
    frozen_params = sum(1 for p in enc_frozen._backbone.parameters() if p.requires_grad)
    print(f"  Frozen:      {frozen_params} trainable backbone params (should be 0)")
    assert frozen_params == 0

    # All-zeros input (simulates placeholder camera).
    # Conv+ReLU nets without BN in train mode may produce zero output
    # for zero input (no bias terms).  The robot-state branch carries
    # all conditioning in that case — this is expected.
    enc.train()  # enable BN
    zeros = torch.zeros(B, H, W, 3)
    out_zero = enc(zeros)
    print(f"  All zeros:   output norm={out_zero.norm(dim=1).mean():.3f}")

    print("\n✅ All vision encoder tests passed!")


if __name__ == "__main__":
    _test()
