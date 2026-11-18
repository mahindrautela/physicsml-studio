"""Slices of saved predictions for the field viewer.

The viewer shows one trajectory, field and component at a time (and, for 3D
data, one plane through the grid). It has two views:

  frame()    one time step: the truth, plus each model's prediction and error
  rollout()  every rollout step: the truth and one model's prediction

Both also return the truth's value range, so colour scales stay fixed while
the user moves through time.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .runs import prediction_file, truth_file

MAX_SIDE = 256          # 2D slices larger than this are thinned out before sending
MAX_SIDE_ROLLOUT = 96   # the all-steps grid shows many small tiles, so it needs less detail
MAX_POINTS_1D = 4096


class Slicer:
    """Picks the requested 1D line or 2D plane out of saved (samples, T, F, C, D, H, W) arrays."""

    def __init__(self, shape: tuple, sample: int, field: int, component: int, axis: str, index: Optional[int]):
        samples, T, F, C, D, H, W = shape
        if not (0 <= sample < samples and 0 <= field < F and 0 <= component < C):
            raise ValueError("Trajectory, field or component out of range")
        self.T = T
        self.grid = [D, H, W]
        self.kind = "1d" if D == 1 and H == 1 else "2d" if D == 1 else "3d"
        self.sample, self.field, self.component, self.axis = sample, field, component, axis
        self.index = None
        if self.kind == "3d":
            size = {"d": D, "h": H, "w": W}[axis]
            self.index = min(max(size // 2 if index is None else int(index), 0), size - 1)

    def all_times(self, arr: np.ndarray) -> np.ndarray:
        """The slice at every time step: (T, n) for 1D data, (T, rows, cols) otherwise."""
        vol = np.asarray(arr[self.sample, :, self.field, self.component], dtype=np.float32)  # (T, D, H, W)
        if self.kind == "1d":
            return vol[:, 0, 0]
        if self.kind == "2d":
            return vol[:, 0]
        return {"d": vol[:, self.index], "h": vol[:, :, self.index], "w": vol[:, :, :, self.index]}[self.axis]

    def describe(self) -> dict:
        return {"kind": self.kind, "T": self.T, "grid": self.grid, "axis": self.axis, "index": self.index}


def frame(run_id: str, dataset: str, models: list[str], sample: int, t: int, mode: str,
          field: int, component: int, axis: str = "d", index: Optional[int] = None) -> dict:
    truth = np.load(truth_file(run_id, dataset), mmap_mode="r")
    slicer = Slicer(truth.shape, sample, field, component, axis, index)
    if not 0 <= t < slicer.T:
        raise ValueError("Time step out of range")
    truth_slices = slicer.all_times(truth)
    result = slicer.describe() | {
        "t": t, "sample": sample,
        "range": _range(truth_slices),
        "truth": _pack(truth_slices[t]),
        "models": {},
        "error_max": 0.0,  # largest error over all time steps: the error colour scale stays fixed
    }
    for model in models:
        file = prediction_file(run_id, model, dataset, mode)
        if not file.exists():
            result["models"][model] = None
            continue
        pred = slicer.all_times(np.load(file, mmap_mode="r"))
        err = np.abs(pred - truth_slices)
        if np.isfinite(err).any():
            result["error_max"] = max(result["error_max"], float(np.nanmax(err)))
        has_prediction = np.isfinite(pred[t]).any()  # NaN where the model made no prediction
        result["models"][model] = {"pred": _pack(pred[t]), "error": _pack(err[t])} if has_prediction else None
    return result


def rollout(run_id: str, dataset: str, model: str, sample: int, field: int, component: int,
            axis: str = "d", index: Optional[int] = None) -> dict:
    """The truth and one model's rollout prediction at every rollout step."""
    truth = np.load(truth_file(run_id, dataset), mmap_mode="r")
    slicer = Slicer(truth.shape, sample, field, component, axis, index)
    file = prediction_file(run_id, model, dataset, "rollout")
    if not file.exists():
        raise FileNotFoundError(f"No rollout of '{model}' on '{dataset}' in this run")
    truth_slices = slicer.all_times(truth)
    pred = slicer.all_times(np.load(file, mmap_mode="r"))
    flat = pred.reshape(slicer.T, -1)
    steps = [t for t in range(slicer.T) if np.isfinite(flat[t]).any()]  # the rollout predicted these
    max_side = MAX_SIDE if slicer.kind == "1d" else MAX_SIDE_ROLLOUT
    return slicer.describe() | {
        "model": model, "sample": sample, "steps": steps,
        "start": steps[0] - 1 if steps else None,  # the last true frame the model was given
        "range": _range(truth_slices),
        "truth": [_pack(truth_slices[t], max_side) for t in steps],
        "pred": [_pack(pred[t], max_side) for t in steps],
    }


def _range(a: np.ndarray) -> list[float]:
    return [float(np.nanmin(a)), float(np.nanmax(a))]


def _pack(a: np.ndarray, max_side: int = MAX_SIDE) -> list:
    """JSON-friendly: thin out very large slices, round to ~5 significant digits, NaN -> null."""
    if a.ndim == 2:
        a = a[::int(np.ceil(a.shape[0] / max_side)), ::int(np.ceil(a.shape[1] / max_side))]
    elif a.shape[0] > MAX_POINTS_1D:
        a = a[::int(np.ceil(a.shape[0] / MAX_POINTS_1D))]
    finite = a[np.isfinite(a)]
    scale = float(np.max(np.abs(finite))) if finite.size else 0.0
    decimals = int(min(12, max(0, 5 - np.floor(np.log10(scale))))) if scale > 0 else 5
    a = np.round(a.astype(np.float64), decimals)
    return np.where(np.isfinite(a), a, None).tolist()
