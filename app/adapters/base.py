"""The adapter contract: how the app talks to any model.

An adapter wraps one family of models. The app only ever calls the methods
below, so plugging in a new model means writing one small class (see
``template.py``) and a model card that names it.

Shapes always use the 7-D layout (B, k, F, C, D, H, W):
  B batch, k input frames (the model's autoregressive order), F fields,
  C components, D x H x W grid. Values are in the dataset's physical units;
  any normalization is the adapter's own business.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np

from ..catalog import ModelCard


class IncompatibleData(Exception):
    """Raised by ``check`` when a model can't run on a dataset."""


class Adapter:
    #: number of past frames the model needs to predict the next one
    ar_order: int = 1
    #: does this adapter need a weights file (card.source)?
    needs_weights: bool = True

    def __init__(self, card: ModelCard, device: str, log: Callable[[str], None]):
        self.card = card
        self.cfg = card.config or {}
        self.device = device
        self.log = log
        self.ar_order = int(self.cfg.get("ar_order", self.ar_order))

    def load(self, weights: Optional[Path]) -> None:
        """Build the model and load weights. Called once per run."""

    def check(self, shape: tuple[int, ...]) -> None:
        """Raise IncompatibleData(reason) if the dataset shape (N,T,F,C,D,H,W) can't be used."""

    def begin(self, trajectories: np.ndarray) -> None:
        """Called before predicting on a dataset, with all evaluation trajectories
        (n, T, F, C, D, H, W). Use it for per-trajectory statistics."""

    def predict(self, frames: np.ndarray, traj: np.ndarray) -> np.ndarray:
        """frames (B, k, F, C, D, H, W) -> next frame (B, F, C, D, H, W).
        ``traj`` (B,) says which evaluation trajectory each row came from."""
        raise NotImplementedError

    def info(self) -> dict:
        """Extra facts to show in the results (e.g. parameter count)."""
        return {}

    def unload(self) -> None:
        """Free memory (GPU) when the run moves on to another model."""
