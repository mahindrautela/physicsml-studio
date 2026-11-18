"""TEMPLATE: copy this file to add a new model type.

1. Copy it to app/adapters/<name>.py, e.g. app/adapters/my_fno.py.
2. Change @register("my_model") to the same name, e.g. @register("my_fno").
3. Fill in load() and predict(). check() and begin() are optional.
4. Add a model card with `adapter: my_fno` (the "Add model" form, or a YAML
   file in catalog/models/), then restart `python studio.py`.

This file itself is skipped at startup, so it never shows up in the app.
See morph.py for a complete example.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from . import register
from ._torch import run_in_batches
from .base import Adapter, IncompatibleData


@register("my_model")
class MyModel(Adapter):
    ar_order = 1          # how many past frames the model reads
    needs_weights = True  # False if the model has no weights file

    def load(self, weights: Optional[Path]) -> None:
        """Build the model and load its weights. self.cfg is the card's `config:` block."""
        # import torch
        # from my_package import MyNet
        # self.model = MyNet(**self.cfg)
        # self.model.load_state_dict(torch.load(weights, map_location="cpu"))
        # self.model.to(self.device).eval()
        raise NotImplementedError("Fill in load()")

    def check(self, shape) -> None:
        """Refuse datasets the model can't handle; the run marks them 'skipped'."""
        N, T, F, C, D, H, W = shape
        if D > 1:
            raise IncompatibleData("This model only handles 1D and 2D grids")

    def predict(self, frames: np.ndarray, traj: np.ndarray) -> np.ndarray:
        """frames (B, k, F, C, D, H, W) in physical units -> next frame (B, F, C, D, H, W)."""
        return run_in_batches(self.model, frames, batch_size=8, device=self.device)
