"""Reference baselines. They need no weights, so they always run, and they give
every comparison a floor: a learned model should beat them."""
from __future__ import annotations

import numpy as np

from . import register
from .base import Adapter


@register("persistence")
class Persistence(Adapter):
    """Predicts that nothing changes: next frame = current frame."""

    needs_weights = False
    ar_order = 1

    def predict(self, frames: np.ndarray, traj: np.ndarray) -> np.ndarray:
        return frames[:, -1].copy()


@register("linear_extrapolation")
class LinearExtrapolation(Adapter):
    """Continues the most recent change: next = 2*current - previous."""

    needs_weights = False
    ar_order = 2

    def predict(self, frames: np.ndarray, traj: np.ndarray) -> np.ndarray:
        return 2.0 * frames[:, -1] - frames[:, -2]
