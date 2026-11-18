"""Any PyTorch model saved as a self-contained file — no Python code needed in the app.

Two file types work:
  .pt2  torch.export (recommended in current PyTorch)
            ep = torch.export.export(model, (example,))
            torch.export.save(ep, "my_model.pt2")
        Export with an example batch of ``batch_size`` frames (default 8); the
        adapter pads smaller batches to that size, so dynamic shapes aren't needed.
  .pt   TorchScript (older, deprecated by PyTorch but still loadable)
            torch.jit.trace(model, example).save("my_model.pt")

The model must take one tensor shaped (B, k, F, C, D, H, W) and return the next
frame (B, F, C, D, H, W), or a tuple whose last item is that.

Card ``config`` keys:
  ar_order       input frames (default 1)
  normalization  none | trajectory | context  (default none; see _revin.py)
  batch_size     frames per forward pass      (default 8)
  max_fields / max_components   optional limits checked before running
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from . import register
from ._revin import RevIN
from .base import Adapter, IncompatibleData


@register("pytorch")
class PyTorchFileAdapter(Adapter):
    def load(self, weights: Optional[Path]) -> None:
        import torch

        self.torch = torch
        if weights is None:
            raise RuntimeError("This model card has no source for the model file")
        weights = Path(weights)
        self.exported = weights.suffix == ".pt2"
        if self.exported:
            self.model = torch.export.load(str(weights)).module()
            self.model.to(self.device)
        else:
            self.model = torch.jit.load(str(weights), map_location=self.device).eval()
        self.revin = RevIN(str(self.cfg.get("normalization", "none")))
        self.batch_size = int(self.cfg.get("batch_size", 8))

    def check(self, shape) -> None:
        _, T, F, C, *_ = shape
        if "max_fields" in self.cfg and F > int(self.cfg["max_fields"]):
            raise IncompatibleData(f"Model supports {self.cfg['max_fields']} fields, dataset has {F}")
        if "max_components" in self.cfg and C > int(self.cfg["max_components"]):
            raise IncompatibleData(f"Model supports {self.cfg['max_components']} components, dataset has {C}")
        if T <= self.ar_order:
            raise IncompatibleData(f"Need more than {self.ar_order} time steps")

    def begin(self, trajectories: np.ndarray) -> None:
        self.revin.fit(trajectories)

    def predict(self, frames: np.ndarray, traj: np.ndarray) -> np.ndarray:
        torch = self.torch
        x, (mu, sd) = self.revin.normalize(frames, traj)
        outs = []
        bs = self.batch_size
        with torch.inference_mode():
            for i in range(0, x.shape[0], bs):
                chunk = np.ascontiguousarray(x[i:i + bs], dtype=np.float32)
                n = chunk.shape[0]
                if self.exported and n < bs:  # exported programs keep the example's batch size
                    chunk = np.concatenate([chunk, np.zeros((bs - n, *chunk.shape[1:]), np.float32)])
                try:
                    out = self.model(torch.from_numpy(chunk).to(self.device))
                except Exception as e:  # noqa: BLE001
                    if self.exported and ("Guard failed" in str(e) or "shape" in str(e).lower()):
                        raise IncompatibleData(
                            f"The exported model was built for a different input shape than "
                            f"{tuple(chunk.shape)} ({str(e)[:120]})") from e
                    raise
                if isinstance(out, (tuple, list)):
                    out = out[-1]
                outs.append(out[:n].float().cpu().numpy())
        out = np.concatenate(outs, axis=0).reshape(frames[:, -1].shape)
        return RevIN.denormalize(out, mu, sd)

    def info(self) -> dict:
        n = sum(p.numel() for p in self.model.parameters())
        return {"parameters_m": round(n / 1e6, 3), "format": ".pt2 (torch.export)" if self.exported else ".pt (TorchScript)"}

    def unload(self) -> None:
        self.model = None
