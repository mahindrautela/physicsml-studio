"""Slices of saved predictions for the field viewer.

The viewer asks for one time step of one trajectory, field and component (and,
for 3D data, one plane through the grid). It gets back the truth, each model's
prediction and absolute error, and the value ranges that keep the colour scales
fixed while the user scrubs through time.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .runs import prediction_file, truth_file

MAX_SIDE = 256      # larger 2D slices are thinned out before sending
MAX_POINTS_1D = 4096


def frame(run_id: str, dataset: str, models: list[str], sample: int, t: int, mode: str,
          field: int, component: int, axis: str = "d", index: Optional[int] = None) -> dict:
    truth = np.load(truth_file(run_id, dataset), mmap_mode="r")  # (samples, T, F, C, D, H, W)
    samples, T, F, C, D, H, W = truth.shape
    if not (0 <= sample < samples and 0 <= t < T and 0 <= field < F and 0 <= component < C):
        raise ValueError("Trajectory, time step, field or component out of range")

    kind = "1d" if D == 1 and H == 1 else "2d" if D == 1 else "3d"
    if kind == "3d":
        size = {"d": D, "h": H, "w": W}[axis]
        index = min(max(size // 2 if index is None else int(index), 0), size - 1)

    def all_times(arr: np.ndarray) -> np.ndarray:
        """The requested slice at every time step: (T, ...) with 1 or 2 spatial dims."""
        vol = np.asarray(arr[sample, :, field, component], dtype=np.float32)  # (T, D, H, W)
        if kind == "1d":
            return vol[:, 0, 0]
        if kind == "2d":
            return vol[:, 0]
        return {"d": vol[:, index], "h": vol[:, :, index], "w": vol[:, :, :, index]}[axis]

    truth_slices = all_times(truth)
    result = {
        "kind": kind, "t": t, "T": T, "sample": sample, "grid": [D, H, W], "axis": axis, "index": index,
        "range": [float(np.nanmin(truth_slices)), float(np.nanmax(truth_slices))],
        "truth": _pack(truth_slices[t]),
        "models": {},
        "error_max": 0.0,  # largest error over all time steps: the error colour scale stays fixed
    }
    for model in models:
        file = prediction_file(run_id, model, dataset, mode)
        if not file.exists():
            result["models"][model] = None
            continue
        pred = all_times(np.load(file, mmap_mode="r"))
        err = np.abs(pred - truth_slices)
        if np.isfinite(err).any():
            result["error_max"] = max(result["error_max"], float(np.nanmax(err)))
        has_prediction = np.isfinite(pred[t]).any()  # NaN where the model made no prediction
        result["models"][model] = {"pred": _pack(pred[t]), "error": _pack(err[t])} if has_prediction else None
    return result


def _pack(a: np.ndarray) -> list:
    """JSON-friendly: thin out very large slices, round to ~5 significant digits, NaN -> null."""
    if a.ndim == 2:
        a = a[::int(np.ceil(a.shape[0] / MAX_SIDE)), ::int(np.ceil(a.shape[1] / MAX_SIDE))]
    elif a.shape[0] > MAX_POINTS_1D:
        a = a[::int(np.ceil(a.shape[0] / MAX_POINTS_1D))]
    finite = a[np.isfinite(a)]
    scale = float(np.max(np.abs(finite))) if finite.size else 0.0
    decimals = int(min(12, max(0, 5 - np.floor(np.log10(scale))))) if scale > 0 else 5
    a = np.round(a.astype(np.float64), decimals)
    return np.where(np.isfinite(a), a, None).tolist()
