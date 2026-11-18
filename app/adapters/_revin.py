"""Reversible instance normalization (RevIN), as used by MORPH.

Each trajectory gets its own mean and variance per field.
  mode "trajectory": statistics over the whole trajectory. This matches MORPH's
                     official evaluation scripts (they normalize the full test
                     trajectory before predicting).
  mode "context":    statistics from the input frames only, so nothing from the
                     future leaks into the prediction.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-6


class RevIN:
    def __init__(self, mode: str = "trajectory"):
        if mode not in ("trajectory", "context", "none"):
            raise ValueError(f"normalization must be trajectory, context or none (got {mode})")
        self.mode = mode
        self.mu = self.sd = None

    def fit(self, trajectories: np.ndarray) -> None:
        # (n, T, F, C, D, H, W) -> stats (n, F)
        if self.mode != "trajectory":
            return
        x = trajectories.transpose(0, 2, 1, 3, 4, 5, 6).reshape(trajectories.shape[0], trajectories.shape[2], -1)
        self.mu = x.mean(axis=2).astype(np.float32)
        self.sd = np.sqrt(x.var(axis=2) + EPS).astype(np.float32)

    def stats(self, frames: np.ndarray, traj: np.ndarray):
        """Return mu, sd broadcastable to (B, F, C, D, H, W)."""
        if self.mode == "none":
            return None, None
        if self.mode == "trajectory":
            mu, sd = self.mu[traj], self.sd[traj]
        else:
            x = frames.transpose(0, 2, 1, 3, 4, 5, 6).reshape(frames.shape[0], frames.shape[2], -1)
            mu = x.mean(axis=2)
            sd = np.sqrt(x.var(axis=2) + EPS)
        return mu[:, :, None, None, None, None], sd[:, :, None, None, None, None]

    def normalize(self, frames: np.ndarray, traj: np.ndarray):
        mu, sd = self.stats(frames, traj)
        if mu is None:
            return frames, (None, None)
        return (frames - mu[:, None]) / sd[:, None], (mu, sd)

    @staticmethod
    def denormalize(out: np.ndarray, mu, sd) -> np.ndarray:
        return out if mu is None else out * sd + mu
