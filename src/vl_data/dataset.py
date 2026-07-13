#!/usr/bin/env python3
"""VLARM Episodic Dataset — PyTorch Dataset over HDF5 episode files.

Reads HDF5 files produced by ``EpisodeRecorder`` and provides windowed
samples for Diffusion Policy training.

Usage::

    from data.dataset import EpisodicDataset
    from torch.utils.data import DataLoader

    ds = EpisodicDataset("data/raw/", obs_horizon=2, action_horizon=16)
    loader = DataLoader(ds, batch_size=32, shuffle=True, collate_fn=ds.collate_fn)
    for batch in loader:
        obs = batch["observations"]   # {key: (B, obs_horizon, ...)}
        act = batch["actions"]        # (B, action_horizon, 9)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import h5py
import numpy as np

# Lazy torch import (this module may be used outside Isaac Sim)
_HAS_TORCH = False


def _ensure_torch():
    global _HAS_TORCH
    if not _HAS_TORCH:
        import torch
        _HAS_TORCH = True
    import torch
    return torch


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

# Default observation keys that the dataset expects in each HDF5 file.
_DEFAULT_OBS_KEYS = [
    "joint_positions",
    "joint_velocities",
    "ee_position",
    "ee_orientation",
    "gripper_width",
    "rgb",
    "depth",
]


class EpisodicDataset:
    """PyTorch Dataset over a directory of episode HDF5 files.

    Each sample is a time-aligned window:
    - *obs_horizon* past observation frames
    - *action_horizon* future action targets

    Parameters
    ----------
    data_dir : str or Path
        Directory containing ``episode_*.h5`` files.
    obs_horizon : int
        Number of past observation steps (default 2, as in Diffusion Policy).
    action_horizon : int
        Number of future actions to predict (default 16).
    action_downsample : int
        Stride between predicted action frames (default 1 = every step).
    obs_keys : list[str] or None
        Which observation keys to load.  ``None`` loads all default keys.
    transform : callable or None
        Optional augmentation to apply to each sample (e.g. ``DataAugmentation``).
    """
    # Class-level module cache so all Dataset instances share the same torch
    _torch: Any = None

    def __init__(
        self,
        data_dir: str | Path,
        obs_horizon: int = 2,
        action_horizon: int = 16,
        action_downsample: int = 1,
        obs_keys: list[str] | None = None,
        transform: Any | None = None,
    ) -> None:
        self._data_dir = Path(data_dir)
        self._obs_horizon = obs_horizon
        self._action_horizon = action_horizon
        self._action_downsample = action_downsample
        self._obs_keys = obs_keys or _DEFAULT_OBS_KEYS
        self._transform = transform

        # Scan episode files
        self._episode_files = sorted(self._data_dir.glob("episode_*.h5"))
        if not self._episode_files:
            raise FileNotFoundError(f"No episode_*.h5 files found in {self._data_dir}")

        # Build global index: [(file_idx, start_step), ...]
        self._index: list[tuple[int, int]] = []
        self._episode_lengths: list[int] = []

        for file_idx, fp in enumerate(self._episode_files):
            with h5py.File(fp, "r") as f:
                T = int(f.attrs.get("num_steps", 0))
                self._episode_lengths.append(T)

            # Valid windows: need obs_horizon steps before prediction start,
            # plus action_horizon * action_downsample steps of future actions.
            window_len = obs_horizon + action_horizon * action_downsample
            for start in range(max(0, T - window_len + 1)):
                self._index.append((file_idx, start))

        self._h5file_cache: dict[int, h5py.File] = {}
        self._cache_max_size = 4

    # ------------------------------------------------------------------
    # Length
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._index)

    # ------------------------------------------------------------------
    # Get item
    # ------------------------------------------------------------------

    def __getitem__(self, idx: int) -> dict[str, Any]:
        file_idx, start_step = self._index[idx]

        # Get (or open) the HDF5 file
        h5f = self._get_file(file_idx)
        T = self._episode_lengths[file_idx]

        obs_end = start_step + self._obs_horizon
        act_start = obs_end
        act_end = act_start + self._action_horizon * self._action_downsample

        # --- Read observation window ---
        observations: dict[str, np.ndarray] = {}
        for key in self._obs_keys:
            ds_path = f"observations/{key}"
            if ds_path in h5f:
                observations[key] = h5f[ds_path][start_step:obs_end]
            else:
                # Fill missing keys with zeros
                observations[key] = np.zeros((self._obs_horizon, 1), dtype=np.float32)

        # --- Read action sequence (subsampled) ---
        if "actions/joint_targets" in h5f:
            actions = h5f["actions/joint_targets"][
                act_start:act_end:self._action_downsample
            ]
        else:
            actions = h5f["actions/joint_targets"][
                act_start:act_end:self._action_downsample
            ]

        # --- Read reward sequences (for info, first 7 arm DOF of action) ---
        arm_actions = actions[:, :7].copy() if actions.shape[1] >= 7 else actions.copy()

        # --- RGB and depth (stored at observation key level for convenience) ---
        rgb = observations.get("rgb", np.zeros((self._obs_horizon, 480, 640, 3), dtype=np.uint8))
        depth = observations.get("depth", np.zeros((self._obs_horizon, 480, 640, 1), dtype=np.float32))

        # Read episode metadata for language embedding placeholder
        lang_embedding = np.zeros(768, dtype=np.float32)

        sample = {
            "observations": observations,
            "actions": arm_actions,
            "rgb": rgb,
            "depth": depth,
            "language_embedding": lang_embedding,
            "episode_id": int(h5f.attrs.get("episode_id", file_idx)),
        }

        # Apply augmentation
        if self._transform is not None:
            sample = self._transform(sample)

        # Convert to torch tensors
        sample = self._to_tensors(sample)

        return sample

    # ------------------------------------------------------------------
    # Collate function
    # ------------------------------------------------------------------

    def collate_fn(self, batch: list[dict]) -> dict[str, Any]:
        """Collate function for ``torch.utils.data.DataLoader``.

        Stacks individual samples into a batch, handling dict observations.
        """
        torch_mod = _ensure_torch()

        collated: dict[str, Any] = {}

        # Stack observation dicts
        obs_keys = batch[0]["observations"].keys()
        collated["observations"] = {}
        for key in obs_keys:
            collated["observations"][key] = torch_mod.stack(
                [item["observations"][key] for item in batch]
            )

        # Stack other tensors
        for key in ("actions", "rgb", "depth", "language_embedding"):
            if key in batch[0]:
                collated[key] = torch_mod.stack([item[key] for item in batch])

        # Collect scalar/episode metadata
        collated["episode_ids"] = torch_mod.tensor(
            [item["episode_id"] for item in batch], dtype=torch_mod.long
        )

        return collated

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_file(self, file_idx: int) -> h5py.File:
        """Return (possibly cached) HDF5 file handle for the given index."""
        if file_idx in self._h5file_cache:
            return self._h5file_cache[file_idx]

        # Evict oldest entry if cache is full
        if len(self._h5file_cache) >= self._cache_max_size:
            oldest = next(iter(self._h5file_cache))
            self._h5file_cache[oldest].close()
            del self._h5file_cache[oldest]

        fp = self._episode_files[file_idx]
        h5f = h5py.File(fp, "r")
        self._h5file_cache[file_idx] = h5f
        return h5f

    def _to_tensors(self, sample: dict) -> dict:
        """Convert numpy arrays in the sample dict to torch tensors."""
        torch_mod = _ensure_torch()

        result: dict[str, Any] = {}

        # Observation dict
        result["observations"] = {}
        for key, arr in sample["observations"].items():
            result["observations"][key] = torch_mod.from_numpy(
                np.asarray(arr).copy()
            ).float()

        # Action sequence
        result["actions"] = torch_mod.from_numpy(
            np.asarray(sample["actions"]).copy()
        ).float()

        # RGB (uint8 → keep as uint8 or convert?)
        if "rgb" in sample:
            result["rgb"] = torch_mod.from_numpy(
                np.asarray(sample["rgb"]).copy()
            ).float() / 255.0  # Normalize to [0, 1]

        # Depth (keep in meters)
        if "depth" in sample:
            result["depth"] = torch_mod.from_numpy(
                np.asarray(sample["depth"]).copy()
            ).float()

        # Language embedding
        result["language_embedding"] = torch_mod.from_numpy(
            np.asarray(sample["language_embedding"]).copy()
        ).float()

        result["episode_id"] = sample.get("episode_id", 0)

        return result

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def compute_statistics(self) -> dict[str, dict[str, float]]:
        """Compute min/mean/max/std for each observation field.

        Returns a dict suitable for normalization::

            {"joint_positions": {"min": ..., "max": ..., "mean": ..., "std": ...}, ...}
        """
        accum: dict[str, list[np.ndarray]] = {}

        for file_idx in range(len(self._episode_files)):
            with h5py.File(self._episode_files[file_idx], "r") as f:
                T = self._episode_lengths[file_idx]
                if T == 0:
                    continue
                for key in self._obs_keys:
                    ds_path = f"observations/{key}"
                    if ds_path in f:
                        arr = f[ds_path][:]
                        if key not in accum:
                            accum[key] = []
                        accum[key].append(arr.reshape(T, -1))

        stats: dict[str, dict[str, float]] = {}
        for key, arrays in accum.items():
            cat = np.concatenate(arrays, axis=0)
            stats[key] = {
                "min": float(np.min(cat)),
                "max": float(np.max(cat)),
                "mean": float(np.mean(cat)),
                "std": float(np.std(cat)),
            }

        return stats

    def close(self) -> None:
        """Close all cached HDF5 file handles."""
        for h5f in self._h5file_cache.values():
            try:
                h5f.close()
            except Exception:
                pass
        self._h5file_cache.clear()
