#!/usr/bin/env python3
"""VLARM Data Augmentation — Transforms for robot manipulation data.

All transforms maintain **temporal consistency**: the same augmentation
parameters are applied to every frame in an observation window so the
diffusion model sees a coherent sequence.

Usage::

    from data.augmentation import Compose, RandomColorJitter, JointNoise

    aug = Compose([
        RandomColorJitter(brightness=0.2, contrast=0.2),
        JointNoise(joint_std=0.005),
    ])
    augmented_sample = aug(sample)
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

# Lazy imports for torch / torchvision (only needed when augmentation runs)
_HAS_TORCH = False
_HAS_TV = False


def _ensure_torch():
    global _HAS_TORCH
    if not _HAS_TORCH:
        import torch
        _HAS_TORCH = True
    import torch
    return torch


def _ensure_tv():
    global _HAS_TV
    if not _HAS_TV:
        import torchvision.transforms.functional as F  # noqa: N811
        _HAS_TV = True
    import torchvision.transforms.functional as F
    return F


# ---------------------------------------------------------------------------
# Compose
# ---------------------------------------------------------------------------


class Compose:
    """Apply a list of transforms sequentially."""

    def __init__(self, transforms: list) -> None:
        self.transforms = transforms

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        for t in self.transforms:
            sample = t(sample)
        return sample

    def __repr__(self) -> str:
        ts = "\n  ".join(repr(t) for t in self.transforms)
        return f"Compose([\n  {ts}\n])"


# ---------------------------------------------------------------------------
# RGB augmentations (temporally consistent)
# ---------------------------------------------------------------------------


class RandomColorJitter:
    """Randomly jitter brightness, contrast, saturation, and hue of RGB images.

    The **same** jitter parameters are applied to all frames in the
    observation window so the temporal structure is preserved.

    Parameters
    ----------
    brightness : float
        Maximum brightness adjustment ``[1-b, 1+b]``.
    contrast : float
        Maximum contrast adjustment ``[1-c, 1+c]``.
    saturation : float
        Maximum saturation adjustment ``[1-s, 1+s]``.
    hue : float
        Maximum hue adjustment in ``[-h, h]``.
    p : float
        Probability of applying the transform.
    """

    def __init__(
        self,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2,
        hue: float = 0.1,
        p: float = 0.8,
    ) -> None:
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.p = p

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        if np.random.random() > self.p:
            return sample

        # Sample jitter parameters ONCE for the whole window
        brightness_factor = 1.0 + np.random.uniform(-self.brightness, self.brightness)
        contrast_factor = 1.0 + np.random.uniform(-self.contrast, self.contrast)
        saturation_factor = 1.0 + np.random.uniform(-self.saturation, self.saturation)
        hue_factor = np.random.uniform(-self.hue, self.hue)

        # Apply to all frames in the window
        rgb = np.asarray(sample.get("rgb", None))
        if rgb is None or rgb.size == 0:
            return sample

        orig_shape = rgb.shape
        T = orig_shape[0]  # number of frames

        augmented = np.empty_like(rgb)
        for t in range(T):
            frame = rgb[t].copy()  # (H, W, 3) uint8 or float

            # Convert to float [0, 1] for processing
            was_uint8 = frame.dtype == np.uint8
            if was_uint8:
                frame_f = frame.astype(np.float32) / 255.0
            else:
                frame_f = frame.astype(np.float32)

            # Brightness
            frame_f = np.clip(frame_f * brightness_factor, 0, 1)

            # Contrast (about mean gray)
            mean = 0.5
            frame_f = np.clip((frame_f - mean) * contrast_factor + mean, 0, 1)

            # Saturation (convert to grayscale, lerp)
            gray = np.mean(frame_f, axis=2, keepdims=True)
            frame_f = np.clip(gray + saturation_factor * (frame_f - gray), 0, 1)

            # Hue (rotate in RGB space — simplified)
            if abs(hue_factor) > 1e-6:
                frame_f = _shift_hue_numpy(frame_f, hue_factor)

            if was_uint8:
                augmented[t] = (frame_f * 255).astype(np.uint8)
            else:
                augmented[t] = frame_f

        sample["rgb"] = augmented
        return sample

    def __repr__(self) -> str:
        return (f"RandomColorJitter(brightness={self.brightness}, "
                f"contrast={self.contrast}, saturation={self.saturation}, "
                f"hue={self.hue}, p={self.p})")


class RandomCrop:
    """Randomly crop and resize RGB images.

    The **same** crop window is applied to all frames.

    Parameters
    ----------
    scale : tuple[float, float]
        Range of crop area relative to original ``(min, max)``.
    ratio : tuple[float, float]
        Range of aspect ratio ``(min, max)``.
    target_size : tuple[int, int]
        ``(H, W)`` to resize back to.
    p : float
        Probability of applying.
    """

    def __init__(
        self,
        scale: tuple[float, float] = (0.8, 1.0),
        ratio: tuple[float, float] = (0.9, 1.1),
        target_size: tuple[int, int] | None = None,
        p: float = 0.5,
    ) -> None:
        self.scale = scale
        self.ratio = ratio
        self.target_size = target_size
        self.p = p

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        if np.random.random() > self.p:
            return sample

        rgb = np.asarray(sample.get("rgb", None))
        if rgb is None or rgb.size == 0:
            return sample

        T, H, W, C = rgb.shape
        target_h, target_w = self.target_size or (H, W)

        # Sample crop params ONCE
        area = H * W
        for _ in range(10):  # rejection sampling
            target_area = np.random.uniform(*self.scale) * area
            aspect = np.random.uniform(*self.ratio)
            crop_h = int(np.sqrt(target_area * aspect))
            crop_w = int(np.sqrt(target_area / aspect))
            if crop_h <= H and crop_w <= W:
                break
        else:
            crop_h, crop_w = H, W  # fallback: no crop

        top = np.random.randint(0, max(1, H - crop_h + 1))
        left = np.random.randint(0, max(1, W - crop_w + 1))

        augmented = np.empty((T, target_h, target_w, C), dtype=rgb.dtype)
        for t in range(T):
            cropped = rgb[t, top:top + crop_h, left:left + crop_w]
            # Simple resize via nearest-neighbor (no cv2 dependency)
            augmented[t] = _resize_numpy(cropped, target_h, target_w)

        sample["rgb"] = augmented
        return sample

    def __repr__(self) -> str:
        return (f"RandomCrop(scale={self.scale}, ratio={self.ratio}, "
                f"target_size={self.target_size}, p={self.p})")


# ---------------------------------------------------------------------------
# State augmentations
# ---------------------------------------------------------------------------


class JointNoise:
    """Add Gaussian noise to joint positions, velocities, EE pose, etc.

    Noise is applied **independently per frame** (no temporal consistency
    needed — sensor noise is uncorrelated).

    Parameters
    ----------
    joint_std : float
        Standard deviation for joint position noise (rad).
    ee_pos_std : float
        Standard deviation for end-effector position noise (m).
    p : float
        Probability of applying.
    """

    def __init__(
        self,
        joint_std: float = 0.005,
        ee_pos_std: float = 0.002,
        p: float = 0.5,
    ) -> None:
        self.joint_std = joint_std
        self.ee_pos_std = ee_pos_std
        self.p = p

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        if np.random.random() > self.p:
            return sample

        obs = sample.get("observations", {})
        for key in ("joint_positions", "joint_velocities"):
            arr = obs.get(key)
            if arr is not None:
                noise = np.random.normal(0, self.joint_std, size=arr.shape).astype(arr.dtype)
                obs[key] = arr + noise

        ee_pos = obs.get("ee_position")
        if ee_pos is not None:
            noise = np.random.normal(0, self.ee_pos_std, size=ee_pos.shape).astype(ee_pos.dtype)
            obs["ee_position"] = ee_pos + noise

        return sample

    def __repr__(self) -> str:
        return (f"JointNoise(joint_std={self.joint_std}, "
                f"ee_pos_std={self.ee_pos_std}, p={self.p})")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _shift_hue_numpy(rgb: np.ndarray, delta: float) -> np.ndarray:
    """Shift hue of an RGB image (H, W, 3) float [0,1] by delta in [-0.5, 0.5].

    Uses HSV conversion for proper hue rotation.
    """
    # RGB → HSV (simple max/min method)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    max_c = np.maximum(np.maximum(r, g), b)
    min_c = np.minimum(np.minimum(r, g), b)
    delta_c = max_c - min_c

    # Hue
    hue = np.zeros_like(max_c)
    mask = delta_c > 1e-6
    # R is max
    r_mask = mask & (max_c == r)
    hue[r_mask] = 60 * ((g[r_mask] - b[r_mask]) / delta_c[r_mask] % 6)
    # G is max
    g_mask = mask & (max_c == g) & ~r_mask
    hue[g_mask] = 60 * ((b[g_mask] - r[g_mask]) / delta_c[g_mask] + 2)
    # B is max
    b_mask = mask & (max_c == b) & ~r_mask & ~g_mask
    hue[b_mask] = 60 * ((r[b_mask] - g[b_mask]) / delta_c[b_mask] + 4)

    # Shift hue
    hue = (hue + delta * 360) % 360
    hue = hue.astype(np.float32)

    # Saturation
    sat = np.zeros_like(max_c)
    sat[mask] = delta_c[mask] / (max_c[mask] + 1e-8)

    # Value
    val = max_c

    # HSV → RGB
    c = val * sat
    x = c * (1 - np.abs((hue / 60) % 2 - 1))
    m = val - c

    h_div = (hue // 60).astype(np.int32)
    result = np.zeros_like(rgb)

    # (R, G, B) coefficients for each 60° HSV sector:
    # Sector 0: R=c, G=x, B=0   Sector 1: R=x, G=c, B=0
    # Sector 2: R=0, G=c, B=x   Sector 3: R=0, G=x, B=c
    # Sector 4: R=x, G=0, B=c   Sector 5: R=c, G=0, B=x
    sector_rgb = [
        (c, x, 0), (x, c, 0), (0, c, x),
        (0, x, c), (x, 0, c), (c, 0, x),
    ]

    for i, (cr, cg, cb) in enumerate(sector_rgb):
        seg_mask = h_div == i
        if not np.any(seg_mask):
            continue
        # R channel
        if isinstance(cr, np.ndarray):
            result[..., 0][seg_mask] = cr[seg_mask] + m[seg_mask]
        else:
            result[..., 0][seg_mask] = m[seg_mask]
        # G channel
        if isinstance(cg, np.ndarray):
            result[..., 1][seg_mask] = cg[seg_mask] + m[seg_mask]
        else:
            result[..., 1][seg_mask] = m[seg_mask]
        # B channel
        if isinstance(cb, np.ndarray):
            result[..., 2][seg_mask] = cb[seg_mask] + m[seg_mask]
        else:
            result[..., 2][seg_mask] = m[seg_mask]

    return np.clip(result, 0, 1).astype(rgb.dtype)


def _resize_numpy(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Resize an image using simple bilinear interpolation (no cv2 dependency)."""
    H, W, C = img.shape
    if H == target_h and W == target_w:
        return img

    # Float for interpolation
    was_uint8 = img.dtype == np.uint8
    img_f = img.astype(np.float32)

    h_idx = np.linspace(0, H - 1, target_h)
    w_idx = np.linspace(0, W - 1, target_w)

    h_floor = np.floor(h_idx).astype(np.int32)
    h_ceil = np.minimum(h_floor + 1, H - 1)
    h_frac = (h_idx - h_floor).reshape(-1, 1, 1)

    w_floor = np.floor(w_idx).astype(np.int32)
    w_ceil = np.minimum(w_floor + 1, W - 1)
    w_frac = (w_idx - w_floor).reshape(1, -1, 1)

    # Bilinear
    top = img_f[h_floor, :, :] * (1 - w_frac) + img_f[h_floor, w_ceil, :] * w_frac
    bot = img_f[h_ceil, :, :] * (1 - w_frac) + img_f[h_ceil, w_ceil, :] * w_frac
    result = top * (1 - h_frac) + bot * h_frac

    if was_uint8:
        return result.astype(np.uint8)
    return result.astype(img.dtype)
