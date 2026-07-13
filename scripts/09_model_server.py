#!/usr/bin/env python3
"""Day 9 — Model inference server (runs in conda vlarm).

Loads the trained model once, then loops waiting for observation requests
from the Isaac Sim process.  Communication via temp files.

Usage::

    conda activate vlarm
    PYTHONPATH=src python scripts/09_model_server.py \
        --checkpoint checkpoints/20260713_165202/best.pt

The server watches for ``/tmp/vlarm_server/request.npz``.  When it appears,
the server runs DDPM inference and writes the predicted action trajectory
to ``/tmp/vlarm_server/response.npy``.  A ``/tmp/vlarm_server/ready``
flag file signals completion.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_src = str(Path(__file__).resolve().parent.parent / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

import torch

from models.diffusion_policy import DiffusionPolicy
from models.noise_scheduler import DDPMScheduler


# =============================================================================
# Server
# =============================================================================


class ModelServer:
    """Load a trained Diffusion Policy and serve predictions via files."""

    def __init__(self, checkpoint_path: str, work_dir: str = "/tmp/vlarm_server"):
        self._work_dir = Path(work_dir)
        self._work_dir.mkdir(parents=True, exist_ok=True)
        self._request_file = self._work_dir / "request.npz"
        self._response_file = self._work_dir / "response.npy"
        self._ready_file = self._work_dir / "ready"
        self._shutdown_file = self._work_dir / "shutdown"
        self._info_file = self._work_dir / "info.json"
        self._device = torch.device("cpu")

        # --- Load model ---
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self._model = DiffusionPolicy(
            action_dim=7, action_horizon=16, obs_horizon=2, use_vision=False,
        ).to(self._device)
        self._model.load_state_dict(ckpt["model_state_dict"])
        self._model.eval()

        # Normalization
        self._act_mean = ckpt.get("act_mean")
        self._act_std = ckpt.get("act_std")
        if self._act_mean is not None:
            self._act_mean = self._act_mean.to(self._device).view(1, 1, -1)
            self._act_std = self._act_std.to(self._device).view(1, 1, -1)

        # Scheduler
        self._scheduler = DDPMScheduler(
            num_train_steps=100, schedule="cosine", device=self._device,
        )

        # Write info
        info = {
            "checkpoint": str(checkpoint_path),
            "epoch": ckpt.get("epoch", "?"),
            "val_loss": float(ckpt.get("best_val_loss", float("nan"))),
            "has_normalization": self._act_mean is not None,
            "n_params": sum(p.numel() for p in self._model.parameters()),
        }
        with open(self._info_file, "w") as f:
            json.dump(info, f)

        print(f"ModelServer ready: {checkpoint_path}")
        print(f"  Epoch: {info['epoch']}, val_loss: {info['val_loss']:.4f}")
        print(f"  Work dir: {self._work_dir}")
        print(f"  Waiting for requests...")

    def run(self) -> None:
        """Main loop: wait for request, predict, write response."""
        while True:
            # Check for shutdown
            if self._shutdown_file.exists():
                self._shutdown_file.unlink()
                print("Shutdown received.")
                break

            # Wait for request
            if not self._request_file.exists():
                time.sleep(0.05)
                continue

            # Read request
            try:
                data = np.load(self._request_file, allow_pickle=True)
                obs = dict(data.items())
                # Convert back from numpy arrays saved in npz
                obs_tensors = {}
                for key, arr in obs.items():
                    if isinstance(arr, np.ndarray) and arr.dtype == np.float32:
                        obs_tensors[key] = torch.from_numpy(arr).unsqueeze(0).to(self._device)
            except Exception as e:
                print(f"Error reading request: {e}")
                self._request_file.unlink()
                continue

            # Remove request file
            self._request_file.unlink()

            # Predict
            t0 = time.monotonic()
            with torch.no_grad():
                pred_norm = self._model.predict_action(
                    obs_tensors, self._scheduler,
                    device=self._device, use_ddpm=True,
                )
            dt = (time.monotonic() - t0) * 1000

            # Denormalize
            if self._act_mean is not None:
                pred = (pred_norm * self._act_std + self._act_mean).cpu().numpy()
            else:
                pred = pred_norm.cpu().numpy()

            # Write response
            np.save(str(self._response_file), pred.astype(np.float32))

            # Signal ready
            self._ready_file.touch()

            print(f"  Prediction: {dt:.0f} ms, range=[{pred.min():.2f}, {pred.max():.2f}]")


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="VLARM — Model Inference Server")
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/20260713_165202/best.pt")
    parser.add_argument("--work-dir", type=str, default="/tmp/vlarm_server")
    args = parser.parse_args()

    server = ModelServer(args.checkpoint, work_dir=args.work_dir)
    server.run()


if __name__ == "__main__":
    main()
