#!/usr/bin/env python3
"""VLARM Configuration Loader.

Loads and merges YAML configuration files into structured dataclasses.
Provides dot-notation access (``cfg.training.batch_size``) with sensible
defaults for every field.

Usage::

    from utils.config import load_config

    cfg = load_config("configs/train_config.yaml")
    print(cfg.training.batch_size)  # 64

    # Override with command-line args
    cfg = load_config("configs/train_config.yaml", overrides={"batch_size": 32})
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Optional

# Lazy import — yaml is not installed in the Isaac Sim Python environment,
# but this module can still be imported for its dataclass defaults.
_YAML_AVAILABLE = False


def _lazy_import_yaml():
    global _YAML_AVAILABLE
    if not _YAML_AVAILABLE:
        import yaml as _yaml_mod
        globals()["yaml"] = _yaml_mod
        _YAML_AVAILABLE = True
    return globals()["yaml"]


# =============================================================================
# Dataclass definitions
# =============================================================================


@dataclass
class DiffusionConfig:
    """Diffusion process hyperparameters."""
    num_train_steps: int = 100
    num_inference_steps: int = 16
    schedule: str = "cosine"  # linear | cosine | squared_cosine


@dataclass
class DataConfig:
    """Dataset and data loading configuration."""
    data_dir: str = "data/raw"
    obs_horizon: int = 2
    action_horizon: int = 16
    action_downsample: int = 1
    val_ratio: float = 0.2
    exclude_episodes: list[int] = field(default_factory=list)


@dataclass
class TrainingConfig:
    """Training loop hyperparameters."""
    batch_size: int = 64
    epochs: int = 200
    lr: float = 1.0e-4
    weight_decay: float = 1.0e-6
    grad_clip_norm: float = 1.0
    num_workers: int = 4
    use_amp: bool = False
    device: str = "auto"


@dataclass
class LRScheduleConfig:
    """Learning rate scheduler configuration."""
    name: str = "cosine"         # cosine | step | plateau | none
    warmup_epochs: int = 5
    step_size: int = 50
    step_gamma: float = 0.5
    plateau_patience: int = 15
    plateau_factor: float = 0.5


@dataclass
class ModelConfig:
    """Diffusion Policy model architecture."""
    action_dim: int = 7
    action_horizon: int = 16
    obs_horizon: int = 2
    state_dim: int = 13
    vision_output_dim: int = 512
    state_output_dim: int = 256
    time_dim: int = 128
    unet_base_channels: int = 64
    use_vision: bool = False
    pretrained_vision: bool = False


@dataclass
class LoggingConfig:
    """Logging and checkpointing configuration."""
    log_dir: str = "logs"
    checkpoint_dir: str = "checkpoints"
    save_every_epochs: int = 20
    save_best: bool = True
    log_every_steps: int = 50
    eval_every_epochs: int = 5


@dataclass
class TrainConfig:
    """Top-level training configuration.

    Aggregates all sub-configs for a full training run.  Use ``load_config()``
    to build an instance from YAML files and optional overrides.
    """
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    lr_schedule: LRScheduleConfig = field(default_factory=LRScheduleConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    seed: int = 42


# =============================================================================
# YAML → dataclass population
# =============================================================================


def _populate_dataclass(obj: Any, data: dict[str, Any]) -> None:
    """Recursively populate a dataclass instance from a dict.

    Skips keys not present in the dataclass.  Converts list/dict defaults
    to the correct types where possible.
    """
    cls_fields = {f.name: f for f in fields(obj)}
    for key, value in data.items():
        if key not in cls_fields:
            continue
        field_obj = cls_fields[key]
        target = getattr(obj, key)

        if isinstance(value, dict) and hasattr(target, "__dataclass_fields__"):
            # Nested dataclass — recurse
            _populate_dataclass(target, value)
        else:
            # Leaf value — cast to expected type if possible
            try:
                setattr(obj, key, _cast_value(value, field_obj.type))
            except (TypeError, ValueError):
                setattr(obj, key, value)


def _cast_value(value: Any, target_type: type) -> Any:
    """Cast a YAML value to the expected dataclass field type."""
    origin = getattr(target_type, "__origin__", None)
    if origin is list:
        # list[int], list[str], etc.
        item_type = getattr(target_type, "__args__", (str,))[0]
        return [item_type(v) for v in value]
    if origin is dict:
        return value  # keep as-is
    return target_type(value) if not isinstance(value, target_type) else value


# =============================================================================
# Public API
# =============================================================================


def load_config(
    config_path: str | Path = "configs/train_config.yaml",
    overrides: dict[str, Any] | None = None,
) -> TrainConfig:
    """Load training configuration from a YAML file with optional overrides.

    Parameters
    ----------
    config_path : str or Path
        Path to the YAML config file.
    overrides : dict or None
        Flat key → value overrides applied after loading.
        Keys match the YAML top-level keys (e.g. ``{"lr": 1e-3, "batch_size": 32}``).

    Returns
    -------
    TrainConfig
        Populated configuration object with dot-notation access.
    """
    cfg = TrainConfig()

    config_path = Path(config_path).resolve()
    if config_path.exists():
        _lazy_import_yaml()
        with open(config_path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        if raw:
            _populate_dataclass(cfg, raw)
    else:
        import warnings
        warnings.warn(f"Config file not found: {config_path} — using defaults.")

    # Apply command-line style overrides (flat keys)
    if overrides:
        _apply_overrides(cfg, overrides)

    # Resolve device
    if cfg.training.device == "auto":
        import torch
        cfg.training.device = "cuda" if torch.cuda.is_available() else "cpu"

    return cfg


def _apply_overrides(cfg: TrainConfig, overrides: dict[str, Any]) -> None:
    """Apply flat key overrides to the config.

    Supports nested keys with ``.`` separator, e.g. ``"training.batch_size"``.
    Falls back to top-level key if no ``.`` is present.
    """
    for key, val in overrides.items():
        if val is None:
            continue
        parts = key.split(".")
        target = cfg
        # Walk (or fallback to flat) to find the sub-dataclass
        for i, part in enumerate(parts[:-1]):
            if hasattr(target, part):
                target = getattr(target, part)
            else:
                break
        leaf = parts[-1]
        if hasattr(target, leaf):
            field_type = None
            try:
                field_type = type(getattr(target, leaf))
            except Exception:
                pass
            try:
                setattr(target, leaf, field_type(val) if field_type else val)
            except (TypeError, ValueError):
                pass
        elif hasattr(cfg, key):
            # Fallback: treat the whole key as a top-level attribute
            try:
                setattr(cfg, key, val)
            except Exception:
                pass


def save_config(cfg: TrainConfig, path: str | Path) -> None:
    """Save the current configuration back to a YAML file.

    Useful for recording the exact config used for a training run.
    """
    _lazy_import_yaml()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert dataclass → dict
    raw = _dataclass_to_dict(cfg)

    with open(path, "w", encoding="utf-8") as fh:
        yaml.dump(raw, fh, default_flow_style=False, allow_unicode=True, sort_keys=False)


def _dataclass_to_dict(obj: Any) -> Any:
    """Recursively convert a dataclass to a plain dict."""
    if hasattr(obj, "__dataclass_fields__"):
        result: dict[str, Any] = {}
        for f in fields(obj):
            value = getattr(obj, f.name)
            result[f.name] = _dataclass_to_dict(value)
        return result
    return obj


# =============================================================================
# Self-test
# =============================================================================


def _test() -> None:
    """Quick smoke test of config loading."""
    print("=== Config Loader Test ===")

    # Test 1: Default config
    cfg = load_config("nonexistent.yaml")
    print(f"  ✅ Defaults: batch_size={cfg.training.batch_size}, lr={cfg.training.lr}")

    # Test 2: Load actual config
    config_path = Path(__file__).resolve().parent.parent / "configs" / "train_config.yaml"
    if config_path.exists():
        cfg = load_config(config_path)
        print(f"  ✅ Loaded:   batch_size={cfg.training.batch_size}, schedule={cfg.diffusion.schedule}")
        print(f"  ✅ Device:   {cfg.training.device}")
        print(f"  ✅ Exclude:  {cfg.data.exclude_episodes}")

    # Test 3: Overrides
    cfg = load_config(config_path, overrides={"training.batch_size": 128, "lr": "5e-4"})
    if cfg.training.batch_size == 128:
        print(f"  ✅ Override: batch_size={cfg.training.batch_size}, lr={cfg.training.lr}")
    else:
        print(f"  ❌ Override failed: batch_size={cfg.training.batch_size}")

    # Test 4: Save
    out_path = Path("/tmp") / "train_config_test.yaml"
    save_config(cfg, out_path)
    if out_path.exists():
        print(f"  ✅ Saved:    {out_path} ({out_path.stat().st_size} bytes)")
        out_path.unlink()

    print("✅ All config tests passed!\n")


if __name__ == "__main__":
    _test()
