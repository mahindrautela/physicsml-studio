"""TEMPLATE: copy this file to add a new kind of model.

1. Copy to e.g. ``app/adapters/my_fno.py`` and rename the class and the
   ``@register("...")`` name.
2. Fill in ``load`` and ``predict`` (``check`` and ``begin`` are optional).
3. Add a model card with ``adapter: my_fno`` (use the "Add model" form or put a
   YAML file in catalog/models).
4. Restart the server. The model appears in the list.

This file itself is skipped at startup, so it never shows up in the app.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from . import register
from .base import Adapter, IncompatibleData


@register("my_model")
class MyModel(Adapter):
    ar_order = 1          # how many past frames the model reads
    needs_weights = True  # False if the model has no weights file

    def load(self, weights: Optional[Path]) -> None:
        import torch

        # self.cfg is the card's `config:` block, e.g. {"width": 64}
        # from my_package import MyNet
        # self.model = MyNet(**self.cfg)
        # self.model.load_state_dict(torch.load(weights, map_location="cpu"))
        # self.model.to(self.device).eval()
        self.torch = torch
        raise NotImplementedError("Fill in load()")

    def check(self, shape) -> None:
        N, T, F, C, D, H, W = shape
        if D > 1:
            raise IncompatibleData("This model only handles 1D and 2D grids")

    def predict(self, frames: np.ndarray, traj: np.ndarray) -> np.ndarray:
        # frames: (B, k, F, C, D, H, W) in physical units -> return (B, F, C, D, H, W)
        x = self.torch.from_numpy(frames).to(self.device)
        with self.torch.inference_mode():
            y = self.model(x)
        return y.cpu().numpy()
