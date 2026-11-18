"""Any PyTorch model saved as a single file, with no Python code needed in the app.

Two file types work:
  .pt2  torch.export (recommended in current PyTorch)
            torch.export.save(torch.export.export(model, (example,)), "my_model.pt2")
        Export with a batch of ``batch_size`` frames (default 8); smaller
        batches are padded to that size, so dynamic shapes aren't needed.
  .pt   TorchScript (older; PyTorch has deprecated it, but it still loads)
            torch.jit.trace(model, example).save("my_model.pt")

The model takes one tensor shaped (B, k, F, C, D, H, W) and returns the next
frame (B, F, C, D, H, W), or a tuple whose last item is that.

Card ``config`` keys (all optional):
  ar_order       input frames (default 1)
  normalization  none | trajectory | context  (default none; see _revin.py)
  batch_size     frames per forward pass      (default 8)
  max_fields, max_components   limits checked before running
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from . import register
from ._revin import RevIN
from ._torch import run_in_batches
from .base import Adapter, IncompatibleData


@register("pytorch")
class PyTorchFileAdapter(Adapter):
    def load(self, weights: Optional[Path]) -> None:
        import torch

        if weights is None:
            raise RuntimeError("This model card has no source for the model file")
        self.exported = Path(weights).suffix == ".pt2"
        if self.exported:
            self.model = torch.export.load(str(weights)).module().to(self.device)
        else:
            self.model = torch.jit.load(str(weights), map_location=self.device).eval()
        self.revin = RevIN(str(self.cfg.get("normalization", "none")))
        self.batch_size = int(self.cfg.get("batch_size", 8))

    def check(self, shape) -> None:
        _, T, F, C, *_ = shape
        if F > int(self.cfg.get("max_fields", F)):
            raise IncompatibleData(f"Model supports {self.cfg['max_fields']} fields, dataset has {F}")
        if C > int(self.cfg.get("max_components", C)):
            raise IncompatibleData(f"Model supports {self.cfg['max_components']} components, dataset has {C}")
        if T <= self.ar_order:
            raise IncompatibleData(f"Need more than {self.ar_order} time steps")

    def begin(self, trajectories: np.ndarray) -> None:
        self.revin.fit(trajectories)

    def predict(self, frames: np.ndarray, traj: np.ndarray) -> np.ndarray:
        x, (mu, sd) = self.revin.normalize(frames, traj)
        try:
            out = run_in_batches(self.model, x, self.batch_size, self.device, pad_to_batch=self.exported)
        except Exception as e:  # noqa: BLE001
            # An exported model only accepts the shapes it was exported with.
            if self.exported and ("Guard failed" in str(e) or "shape" in str(e).lower()):
                raise IncompatibleData(f"The exported model was built for a different input shape than "
                                       f"{tuple(x.shape[1:])} ({str(e)[:120]})") from e
            raise
        return RevIN.denormalize(out.reshape(frames[:, -1].shape), mu, sd)

    def info(self) -> dict:
        n_params = sum(p.numel() for p in self.model.parameters())
        return {"parameters_m": round(n_params / 1e6, 3),
                "format": ".pt2 (torch.export)" if self.exported else ".pt (TorchScript)"}

    def unload(self) -> None:
        self.model = None
