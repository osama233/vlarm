"""VLARM Data Pipeline Package.

Provides episode recording, PyTorch dataset loading, and data augmentation
for the VLARM robot manipulation project.

Modules:
    recorder.py     — EpisodeRecorder (HDF5 output) + CameraSource interface
    dataset.py      — EpisodicDataset (PyTorch Dataset over HDF5 episodes)
    augmentation.py — Compose, RandomColorJitter, RandomCrop, JointNoise

Usage::

    from vl_data.recorder import EpisodeRecorder, NullCameraSource
    from vl_data.dataset import EpisodicDataset
    from vl_data.augmentation import Compose, RandomColorJitter, JointNoise

    # Recording
    recorder = EpisodeRecorder(save_dir="data/raw/")
    recorder.start_episode(env)
    # ... call recorder.record_step(...) each step ...
    recorder.end_episode(success=True)

    # Loading
    ds = EpisodicDataset("data/raw/", obs_horizon=2, action_horizon=16)
    sample = ds[0]  # {observations, actions, rgb, depth, language_embedding}

    # Augmentation
    aug = Compose([RandomColorJitter(), JointNoise()])
    sample = aug(sample)
"""

from vl_data.augmentation import (
    Compose,
    JointNoise,
    RandomColorJitter,
    RandomCrop,
)
from vl_data.dataset import EpisodicDataset
from vl_data.recorder import (
    CameraSource,
    EpisodeRecorder,
    NullCameraSource,
    validate_dataset,
    validate_episode,
)

__all__ = [
    # Recorder
    "EpisodeRecorder",
    "CameraSource",
    "NullCameraSource",
    "validate_episode",
    "validate_dataset",
    # Dataset
    "EpisodicDataset",
    # Augmentation
    "Compose",
    "RandomColorJitter",
    "RandomCrop",
    "JointNoise",
]
