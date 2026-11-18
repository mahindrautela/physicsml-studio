"""Error metrics, computed in the data's physical units.

Definitions follow MORPH's evaluation code (morph_pde.utils.metrics_3d):
  NRMSE = sqrt( mean((pred - true)^2) / mean(true^2) )            per trajectory, then averaged
  VRMSE = sqrt( mean((pred - true)^2) / var(true) )               per trajectory, then averaged
NRMSE and VRMSE have no units, so they can be compared across datasets.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-7


def _per_traj(x: np.ndarray) -> np.ndarray:
    return x.reshape(x.shape[0], -1).mean(axis=1)


def summary(pred: np.ndarray, true: np.ndarray) -> dict:
    """pred/true: (n, steps, F, C, D, H, W) with matching shapes."""
    diff = pred.astype(np.float64) - true.astype(np.float64)
    t64 = true.astype(np.float64)
    mse = float(np.mean(diff ** 2))
    mse_traj = _per_traj(diff ** 2)
    ms_true = _per_traj(t64 ** 2)
    var_true = t64.reshape(t64.shape[0], -1).var(axis=1)
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(diff))),
        "nrmse": float(np.mean(np.sqrt(mse_traj / (ms_true + EPS)))),
        "vrmse": float(np.mean(np.sqrt(mse_traj / (var_true + EPS)))),
    }


def per_field_nrmse(pred: np.ndarray, true: np.ndarray) -> list[float]:
    out = []
    for f in range(true.shape[2]):
        d = pred[:, :, f].astype(np.float64) - true[:, :, f]
        num = _per_traj(d ** 2)
        den = _per_traj(true[:, :, f].astype(np.float64) ** 2)
        out.append(float(np.mean(np.sqrt(num / (den + EPS)))))
    return out


def per_step_nrmse(pred: np.ndarray, true: np.ndarray) -> list[float]:
    """Error at each rollout step, averaged over trajectories."""
    out = []
    for s in range(true.shape[1]):
        d = pred[:, s].astype(np.float64) - true[:, s]
        num = _per_traj(d ** 2)
        den = _per_traj(true[:, s].astype(np.float64) ** 2)
        out.append(float(np.mean(np.sqrt(num / (den + EPS)))))
    return out


# ---------------------------------------------------------------- export
CSV_COLUMNS = [
    "model", "dataset", "status", "trajectories",
    "onestep_mse", "onestep_rmse", "onestep_mae", "onestep_nrmse", "onestep_vrmse",
    "rollout_steps", "rollout_nrmse", "rollout_vrmse", "rollout_final_nrmse",
    "ms_per_frame", "parameters_m", "error",
]


def to_csv(run: dict) -> str:
    """One row per model × dataset of a run record."""
    lines = [",".join(CSV_COLUMNS)]
    for pair in run["pairs"]:
        m = pair.get("metrics") or {}
        one, roll = m.get("onestep", {}), m.get("rollout", {})
        curve = roll.get("per_step_nrmse") or [None]
        row = [
            pair["model"], pair["dataset"], pair["status"], m.get("trajectories"),
            one.get("mse"), one.get("rmse"), one.get("mae"), one.get("nrmse"), one.get("vrmse"),
            m.get("rollout_steps"), roll.get("nrmse"), roll.get("vrmse"), curve[-1],
            (pair.get("timing") or {}).get("ms_per_frame"), (pair.get("info") or {}).get("parameters_m"),
            (pair.get("error") or "").replace(",", ";").replace("\n", " "),
        ]
        lines.append(",".join("" if x is None else str(x) for x in row))
    return "\n".join(lines) + "\n"
