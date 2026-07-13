#!/usr/bin/env python3
"""VLARM Episode Recorder — Record robot episodes to HDF5.

Captures observations, actions, rewards, and camera frames from an IsaacEnv
and writes them to compressed HDF5 files, one per episode.

Usage (inside Isaac Sim's Python environment)::

    from envs.isaac_env import IsaacEnv
    from data.recorder import EpisodeRecorder, NullCameraSource

    env = IsaacEnv(simulation_app=simulation_app)
    camera = NullCameraSource()
    recorder = EpisodeRecorder(save_dir="data/raw/", camera_source=camera)

    obs = env.reset(seed=42)
    recorder.start_episode(env)

    for _ in range(200):
        action = policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        recorder.record_step(obs=obs, action=action,
                             reward=reward, terminated=terminated,
                             truncated=truncated)
        if terminated or truncated:
            break

    file_path = recorder.end_episode(success=terminated)
    recorder.close()

Data format (one .h5 file per episode)::

    episode_00000.h5
    ├── .attrs/  episode_id, timestamp, num_steps, success, config_json
    ├── observations/
    │   ├── joint_positions   (T, 9)  float32
    │   ├── joint_velocities  (T, 9)  float32
    │   ├── ee_position       (T, 3)  float32
    │   ├── ee_orientation    (T, 4)  float32
    │   ├── gripper_width     (T, 1)  float32
    │   ├── rgb               (T, H, W, 3) uint8
    │   └── depth             (T, H, W, 1) float32
    ├── actions/
    │   └── joint_targets     (T, 9)  float32
    ├── rewards               (T,)   float32
    ├── terminals             (T,)   bool
    └── truncations           (T,)   bool
"""

from __future__ import annotations

import json
import time as _time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import h5py
import numpy as np

# ---------------------------------------------------------------------------
# Camera Source interface
# ---------------------------------------------------------------------------


class CameraSource(ABC):
    """Abstract camera capture interface.

    Concrete implementations:
      - ``NullCameraSource`` — returns zeros (tests, no-camera envs).
      - ``IsaacSimCameraSource`` — uses Isaac Sim synthetic-data API (future).
      - ``ROS2CameraSource`` — subscribes to /rgb, /depth topics (future).

    Each ``capture()`` call should return the current rendered frame.
    """

    @abstractmethod
    def capture(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (rgb, depth) as numpy arrays.

        Returns
        -------
        rgb : np.ndarray
            Shape ``(H, W, 3)``, dtype ``uint8``.
        depth : np.ndarray
            Shape ``(H, W, 1)``, dtype ``float32``, in metres.
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """Release any GPU or ROS2 resources."""
        ...


class NullCameraSource(CameraSource):
    """Camera source that returns zero-filled images.

    Used for testing and environments where no real camera is available.
    """

    def __init__(self, height: int = 480, width: int = 640) -> None:
        self._rgb = np.zeros((height, width, 3), dtype=np.uint8)
        self._depth = np.zeros((height, width, 1), dtype=np.float32)

    def capture(self) -> tuple[np.ndarray, np.ndarray]:
        return self._rgb.copy(), self._depth.copy()

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Episode Recorder
# ---------------------------------------------------------------------------

# Observation keys stored under /observations/ in the HDF5 file.
# Map: obs_dict_key → (hdf5_dataset_name, dtype, shape_per_step)
_OBS_SCHEMA: list[tuple[str, str, type, tuple[int, ...]]] = [
    ("joint_positions",  "joint_positions",   np.float32, (9,)),
    ("joint_velocities", "joint_velocities",  np.float32, (9,)),
    ("ee_position",      "ee_position",       np.float32, (3,)),
    ("ee_orientation",   "ee_orientation",    np.float32, (4,)),
    ("gripper_width",    "gripper_width",     np.float32, (1,)),
    ("rgb",              "rgb",               np.uint8,   (480, 640, 3)),
    ("depth",            "depth",             np.float32, (480, 640, 1)),
]


class EpisodeRecorder:
    """Records environment interactions into HDF5 files, one per episode.

    Parameters
    ----------
    save_dir : str or Path
        Directory for output HDF5 files (e.g. ``data/raw/``).
    camera_source : CameraSource or None
        Source for RGB and depth frames.  If ``None``, uses ``NullCameraSource``.
    flush_every : int
        Number of steps to buffer in memory before flushing to disk.
        Lower values use less RAM; higher values give better I/O throughput.
    compress_gzip : int
        Gzip compression level for HDF5 datasets (0 = none, 4 = default).
    """

    def __init__(
        self,
        save_dir: str | Path = "data/raw/",
        camera_source: CameraSource | None = None,
        flush_every: int = 50,
        compress_gzip: int = 4,
    ) -> None:
        self._save_dir = Path(save_dir)
        self._save_dir.mkdir(parents=True, exist_ok=True)

        self._camera_source = camera_source or NullCameraSource()
        self._flush_every = max(1, flush_every)
        self._compress_gzip = compress_gzip

        # Internal state
        self._h5file: h5py.File | None = None
        self._episode_count: int = 0
        self._current_step: int = 0
        self._datasets: dict[str, h5py.Dataset] = {}
        self._buffer: dict[str, list[np.ndarray]] = {}
        self._buffered_steps: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_episode(self, env: Any, seed: int | None = None) -> int:
        """Begin recording a new episode.

        Must be called **after** ``env.reset()`` so the first observation
        can be captured.  Writes scene config from ``env._scene["config"]``
        as HDF5 attributes.

        Parameters
        ----------
        env : IsaacEnv
            The environment being recorded (used to read ``_scene["config"]``
            and optionally a ``_camera_source`` for fallback images).
        seed : int or None
            Random seed used for this episode (stored in metadata).

        Returns
        -------
        int
            The episode ID (0-based).
        """
        # Close any previous file
        if self._h5file is not None:
            self._h5file.close()

        # Determine episode ID from existing files
        existing = sorted(self._save_dir.glob("episode_*.h5"))
        episode_id = len(existing)

        # Create new HDF5 file
        file_path = self._save_dir / f"episode_{episode_id:05d}.h5"
        self._h5file = h5py.File(file_path, "w")

        # Write metadata
        self._h5file.attrs["episode_id"] = episode_id
        self._h5file.attrs["timestamp"] = datetime.now(timezone.utc).isoformat()
        self._h5file.attrs["seed"] = seed if seed is not None else -1
        self._h5file.attrs["env_version"] = "vlarm-0.1"
        self._h5file.attrs["camera_source"] = type(self._camera_source).__name__

        # Store scene config (from env)
        try:
            config = getattr(env, "_scene", {}).get("config", {})
            # Convert non-serializable values
            config_serializable = {}
            for k, v in config.items():
                if isinstance(v, tuple):
                    config_serializable[k] = list(v)
                elif isinstance(v, (int, float, str, bool, list, dict)):
                    config_serializable[k] = v
                else:
                    config_serializable[k] = str(v)
            self._h5file.attrs["config_json"] = json.dumps(config_serializable)
        except Exception:
            self._h5file.attrs["config_json"] = "{}"

        # Create groups
        obs_grp = self._h5file.create_group("observations")
        act_grp = self._h5file.create_group("actions")

        # Create extensible datasets (maxshape=None on axis 0 = unlimited)
        self._datasets = {}
        for obs_key, ds_name, dtype, shape in _OBS_SCHEMA:
            ds = obs_grp.create_dataset(
                ds_name,
                shape=(0, *shape),
                maxshape=(None, *shape),
                dtype=dtype,
                compression="gzip",
                compression_opts=self._compress_gzip,
                chunks=(1, *shape),
            )
            self._datasets[ds_name] = ds

        # Action dataset
        act_ds = act_grp.create_dataset(
            "joint_targets",
            shape=(0, 9),
            maxshape=(None, 9),
            dtype=np.float32,
            compression="gzip",
            compression_opts=self._compress_gzip,
            chunks=(1, 9),
        )
        self._datasets["joint_targets"] = act_ds

        # Scalar datasets
        for scalar_name in ("rewards", "terminals", "truncations"):
            dtype = np.float32 if scalar_name == "rewards" else np.bool_
            ds = self._h5file.create_dataset(
                scalar_name,
                shape=(0,),
                maxshape=(None,),
                dtype=dtype,
                compression="gzip",
                compression_opts=self._compress_gzip,
                chunks=(16,),
            )
            self._datasets[scalar_name] = ds

        # Reset buffers
        self._buffer = {k: [] for k in [*[s[1] for s in _OBS_SCHEMA],
                                          "joint_targets",
                                          "rewards", "terminals", "truncations"]}
        self._buffered_steps = 0
        self._current_step = 0
        self._episode_count = episode_id

        return episode_id

    def record_step(
        self,
        obs: dict[str, np.ndarray],
        action: np.ndarray,
        reward: float,
        terminated: bool,
        truncated: bool,
    ) -> None:
        """Append one timestep to the recording.

        Data is buffered in memory and flushed to disk every *flush_every*
        steps to avoid excessive HDF5 resize operations.

        Parameters
        ----------
        obs : dict
            Observation dict from ``IsaacEnv.step()`` or ``IsaacEnv.reset()``.
        action : np.ndarray
            Joint position targets — shape ``(7,)`` (arm only) or ``(9,)``
            (arm + gripper).  Arm-only is padded to 9 with gripper open.
        reward : float
            Scalar reward.
        terminated : bool
            Whether the episode reached a terminal state.
        truncated : bool
            Whether the episode was truncated (max steps).
        """
        if self._h5file is None:
            raise RuntimeError("start_episode() must be called before record_step()")

        # -- Capture camera frames from the source --
        rgb, depth = self._camera_source.capture()

        # -- Collect observation data --
        self._buffer["joint_positions"].append(
            np.asarray(obs.get("joint_positions", np.zeros(9)), dtype=np.float32).ravel())
        self._buffer["joint_velocities"].append(
            np.asarray(obs.get("joint_velocities", np.zeros(9)), dtype=np.float32).ravel())
        self._buffer["ee_position"].append(
            np.asarray(obs.get("ee_position", np.zeros(3)), dtype=np.float32).ravel())
        self._buffer["ee_orientation"].append(
            np.asarray(obs.get("ee_orientation", np.array([1, 0, 0, 0])), dtype=np.float32).ravel())
        self._buffer["gripper_width"].append(
            np.atleast_1d(np.asarray(obs.get("gripper_width", 0.08), dtype=np.float32)).ravel())
        self._buffer["rgb"].append(np.asarray(rgb, dtype=np.uint8))
        self._buffer["depth"].append(np.asarray(depth, dtype=np.float32))

        # -- Action (pad to 9 DOF if needed) --
        act = np.asarray(action, dtype=np.float32).ravel()
        if len(act) == 7:
            act = np.concatenate([act, np.array([0.04, 0.04], dtype=np.float32)])
        elif len(act) != 9:
            raise ValueError(f"Action must be shape (7,) or (9,), got {len(act)}")
        self._buffer["joint_targets"].append(act)

        # -- Scalars --
        self._buffer["rewards"].append(np.float32(reward))
        self._buffer["terminals"].append(np.bool_(terminated))
        self._buffer["truncations"].append(np.bool_(truncated))

        self._buffered_steps += 1
        self._current_step += 1

        # Flush to disk when buffer is full
        if self._buffered_steps >= self._flush_every:
            self._flush()

    def end_episode(self, success: bool = False) -> str:
        """Finalize the current episode and close the HDF5 file.

        Parameters
        ----------
        success : bool
            Whether the episode completed the task successfully.

        Returns
        -------
        str
            Path to the HDF5 file.
        """
        if self._h5file is None:
            raise RuntimeError("No episode in progress (call start_episode first)")

        # Flush remaining buffered data
        if self._buffered_steps > 0:
            self._flush()

        # Write final metadata
        self._h5file.attrs["num_steps"] = self._current_step
        self._h5file.attrs["success"] = bool(success)

        file_path = self._h5file.filename
        self._h5file.close()
        self._h5file = None
        self._datasets = {}
        self._buffer = {}
        self._buffered_steps = 0

        return file_path

    def close(self) -> None:
        """Close any open HDF5 file and release camera resources."""
        if self._h5file is not None:
            try:
                self._h5file.close()
            except Exception:
                pass
            self._h5file = None
        self._camera_source.close()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def episode_count(self) -> int:
        """Number of episodes completed so far."""
        return self._episode_count

    @property
    def current_step(self) -> int:
        """Step count within the current episode."""
        return self._current_step

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _flush(self) -> None:
        """Write buffered data to HDF5 datasets and clear buffers."""
        if self._h5file is None or self._buffered_steps == 0:
            return

        n = self._buffered_steps
        current_len = self._datasets["joint_positions"].shape[0]

        # Resize all datasets to accommodate new data
        new_len = current_len + n
        for ds in self._datasets.values():
            ds.resize(new_len, axis=0)

        # Write observation data
        for _, ds_name, _, _ in _OBS_SCHEMA:
            stacked = np.stack(self._buffer[ds_name], axis=0)
            self._datasets[ds_name][current_len:new_len] = stacked

        # Write actions
        self._datasets["joint_targets"][current_len:new_len] = np.stack(
            self._buffer["joint_targets"], axis=0)

        # Write scalars
        self._datasets["rewards"][current_len:new_len] = np.array(
            self._buffer["rewards"], dtype=np.float32)
        self._datasets["terminals"][current_len:new_len] = np.array(
            self._buffer["terminals"], dtype=np.bool_)
        self._datasets["truncations"][current_len:new_len] = np.array(
            self._buffer["truncations"], dtype=np.bool_)

        # Clear buffers
        for key in self._buffer:
            self._buffer[key] = []
        self._buffered_steps = 0


# ---------------------------------------------------------------------------
# Data validation
# ---------------------------------------------------------------------------

# Franka Panda joint limits (rad) — for range checks
_FRANKA_LIMITS_LOW = np.array(
    [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, 0.0, 0.0],
    dtype=np.float32,
)
_FRANKA_LIMITS_HIGH = np.array(
    [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973, 0.04, 0.04],
    dtype=np.float32,
)

# Required datasets in a valid episode file
_REQUIRED_DATASETS = [
    "observations/joint_positions",
    "observations/joint_velocities",
    "observations/ee_position",
    "observations/ee_orientation",
    "observations/gripper_width",
    "observations/rgb",
    "observations/depth",
    "actions/joint_targets",
    "rewards",
    "terminals",
    "truncations",
]

_REQUIRED_ATTRS = ["episode_id", "timestamp", "num_steps", "success"]


def validate_episode(file_path: str | Path) -> dict:
    """Validate a single episode HDF5 file.

    Parameters
    ----------
    file_path : str or Path
        Path to an ``episode_*.h5`` file.

    Returns
    -------
    dict
        Keys: ``valid`` (bool), ``num_steps`` (int), ``issues`` (list[str]),
        ``stats`` (dict with per-key min/max/mean summaries).
    """
    issues: list[str] = []
    stats: dict[str, dict[str, float]] = {}
    num_steps = 0

    try:
        f = h5py.File(file_path, "r")
    except OSError as e:
        return {"valid": False, "num_steps": 0, "issues": [f"Cannot open: {e}"], "stats": {}}

    try:
        # Check attributes
        for attr in _REQUIRED_ATTRS:
            if attr not in f.attrs:
                issues.append(f"Missing attribute: {attr}")

        num_steps = int(f.attrs.get("num_steps", 0))
        if num_steps <= 0:
            issues.append(f"num_steps={num_steps} (must be > 0)")

        # Check required datasets
        for ds_path in _REQUIRED_DATASETS:
            if ds_path not in f:
                issues.append(f"Missing dataset: {ds_path}")
                continue

            ds = f[ds_path]
            if ds.shape[0] != num_steps:
                issues.append(
                    f"{ds_path}: expected T={num_steps}, got shape[0]={ds.shape[0]}"
                )

            # Collect stats
            try:
                arr = ds[:]
                stats[ds_path] = {
                    "min": float(np.min(arr)),
                    "max": float(np.max(arr)),
                    "mean": float(np.mean(arr)),
                }
            except Exception:
                pass

        # Check joint position ranges (account for revolute joint wrapping)
        jp_ds = f.get("observations/joint_positions")
        if jp_ds is not None and jp_ds.shape[0] > 0:
            jp = jp_ds[:]
            if np.any(np.isnan(jp)):
                issues.append("joint_positions contains NaN")
            # For each revolute joint, check if any wrap of the value
            # (value + k*2π) falls within the joint limits.
            for j in range(7):
                col = jp[:, j]
                low = _FRANKA_LIMITS_LOW[j]
                high = _FRANKA_LIMITS_HIGH[j]
                # Revolute joints can wrap multiple revolutions in
                # simulation.  Check all wraps ±k·2π for k=0…5 so
                # even ~31 rad (±5 revs) values resolve correctly.
                margin = 0.20
                k_vals = np.arange(-5, 6)  # [-5, -4, …, 4, 5]
                vals = col[:, None] + 2.0 * np.pi * k_vals[None, :]
                in_range = np.any(
                    (vals >= low - margin) & (vals <= high + margin), axis=1
                )
                bad = col[~in_range]
                if len(bad) > 0:
                    issues.append(
                        f"joint_{j+1} out of range [{low:.2f}, {high:.2f}]: "
                        f"worst={np.max(np.abs(bad)):.2f} rad "
                        f"({len(bad)}/{len(col)} steps)"
                    )

        # Check RGB range
        rgb_ds = f.get("observations/rgb")
        if rgb_ds is not None and rgb_ds.shape[0] > 0:
            rgb = rgb_ds[:]
            if rgb.min() < 0 or rgb.max() > 255:
                issues.append(f"rgb values out of [0,255]: min={rgb.min()}, max={rgb.max()}")

        # Check depth
        depth_ds = f.get("observations/depth")
        if depth_ds is not None and depth_ds.shape[0] > 0:
            depth = depth_ds[:]
            if np.any(np.isnan(depth)):
                issues.append("depth contains NaN")
            if np.any(depth < 0):
                issues.append(f"depth has negative values: min={depth.min():.4f}")

        # Check gripper width
        gw_ds = f.get("observations/gripper_width")
        if gw_ds is not None and gw_ds.shape[0] > 0:
            gw = gw_ds[:]
            if np.min(gw) < -0.05 or np.max(gw) > 0.50:
                issues.append(f"gripper_width out of [-0.02, 0.15]: min={np.min(gw):.4f}, max={np.max(gw):.4f}")

    finally:
        f.close()

    return {
        "valid": len(issues) == 0,
        "num_steps": num_steps,
        "issues": issues,
        "stats": stats,
    }


def validate_dataset(data_dir: str | Path) -> tuple[int, int, list[str]]:
    """Run ``validate_episode`` on all .h5 files in *data_dir*.

    Returns
    -------
    tuple[int, int, list[str]]
        ``(num_valid, num_total, all_issues)``
    """
    data_dir = Path(data_dir)
    h5_files = sorted(data_dir.glob("episode_*.h5"))
    all_issues: list[str] = []
    num_valid = 0

    for fp in h5_files:
        result = validate_episode(fp)
        if result["valid"]:
            num_valid += 1
        else:
            all_issues.append(f"{fp.name}: {'; '.join(result['issues'])}")

    return num_valid, len(h5_files), all_issues
