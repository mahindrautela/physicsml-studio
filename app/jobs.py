"""Runs: every selected model on every selected dataset, in the background.

A run is stored in ``workspace/runs/<run_id>/``:
    run.json                       status, settings, metrics (what the UI polls)
    data/<dataset>/truth.npy       ground truth for the trajectories shown in the viewer
    pairs/<model>__<dataset>/onestep.npy, rollout.npy   predictions for those trajectories

Two kinds of prediction are made for each (model, dataset) pair:
  one-step  every frame predicted from the true previous frame(s)
  rollout   the model is fed its own predictions, starting from one true frame,
            which shows how errors build up over time
"""
from __future__ import annotations

import json
import shutil
import threading
import time
import traceback
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field

from . import catalog, metrics
from .adapters import REGISTRY, IncompatibleData
from .config import RUNS_DIR
from .datasets import DatasetHandle
from .sources import fetch

DATA_CACHE_BYTES = 2 * 1024 ** 3  # keep up to 2 GB of loaded trajectories between models
MAX_PAIR_BYTES = 3 * 1024 ** 3


class RunSettings(BaseModel):
    max_trajectories: int = Field(8, ge=1, le=10000, description="Trajectories evaluated per dataset")
    rollout_steps: int = Field(0, ge=0, le=10000, description="0 = as many as the data allows")
    rollout_start: int = Field(0, ge=0, description="Time index of the last true frame given to the model")
    viz_samples: int = Field(3, ge=1, le=16, description="Trajectories saved for the viewer")
    device: str = "auto"


def pick_device(name: str) -> str:
    try:
        import torch
    except ImportError:
        return "cpu"
    if name and name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Cancelled(Exception):
    pass


class RunManager:
    def __init__(self):
        self.lock = threading.RLock()
        self.active: dict[str, dict] = {}
        self.cancel_flags: dict[str, threading.Event] = {}
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="run")
        self._mark_interrupted()

    # ------------------------------------------------------------ storage
    def _dir(self, run_id: str) -> Path:
        if not run_id.replace("-", "").isalnum():
            raise KeyError(run_id)
        return RUNS_DIR / run_id

    def _save(self, run: dict) -> None:
        d = self._dir(run["id"])
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / "run.json.tmp"
        tmp.write_text(json.dumps(run, indent=1, default=str))
        tmp.replace(d / "run.json")

    def _mark_interrupted(self) -> None:
        for f in RUNS_DIR.glob("*/run.json"):
            try:
                run = json.loads(f.read_text())
            except Exception:  # noqa: BLE001
                continue
            if run.get("status") in ("queued", "running"):
                run["status"] = "interrupted"
                for p in run["pairs"]:
                    if p["status"] in ("queued", "running"):
                        p["status"] = "interrupted"
                self._save(run)

    def get(self, run_id: str) -> dict:
        with self.lock:
            if run_id in self.active:
                return json.loads(json.dumps(self.active[run_id], default=str))
        f = self._dir(run_id) / "run.json"
        if not f.exists():
            raise KeyError(run_id)
        return json.loads(f.read_text())

    def list(self) -> list[dict]:
        out = []
        for f in RUNS_DIR.glob("*/run.json"):
            try:
                run = self.get(f.parent.name)
            except Exception:  # noqa: BLE001
                continue
            out.append({k: run.get(k) for k in ("id", "name", "created", "status", "models", "datasets")}
                       | {"done": sum(p["status"] == "done" for p in run["pairs"]), "total": len(run["pairs"])})
        out.sort(key=lambda r: r["created"], reverse=True)
        return out

    def delete(self, run_id: str) -> None:
        with self.lock:
            if run_id in self.active:
                raise RuntimeError("Stop the run before deleting it")
        shutil.rmtree(self._dir(run_id), ignore_errors=False)

    def cancel(self, run_id: str) -> None:
        with self.lock:
            if run_id not in self.cancel_flags:
                raise KeyError(run_id)
            self.cancel_flags[run_id].set()

    # ------------------------------------------------------------ create
    def create(self, model_ids: list[str], dataset_ids: list[str], settings: RunSettings,
               name: Optional[str] = None) -> dict:
        if not model_ids or not dataset_ids:
            raise ValueError("Pick at least one model and one dataset")
        models = [catalog.get("models", m) for m in model_ids]
        datasets = [catalog.get("datasets", d) for d in dataset_ids]
        for m in models:
            if m.adapter not in REGISTRY:
                raise ValueError(f"Model '{m.name}' uses adapter '{m.adapter}', which isn't installed")
        ks = [int(m.config.get("ar_order", REGISTRY[m.adapter].ar_order)) for m in models]
        max_k = max(ks)
        run_id = datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
        run = {
            "id": run_id,
            "name": name or f"{len(models)} model{'s' * (len(models) > 1)} × {len(datasets)} dataset{'s' * (len(datasets) > 1)}",
            "created": now(), "started": None, "finished": None, "status": "queued",
            "settings": settings.model_dump(), "device": None,
            "models": model_ids, "datasets": dataset_ids,
            "model_cards": {m.id: m.model_dump() for m in models},
            "dataset_cards": {d.id: d.model_dump() for d in datasets},
            "max_ar_order": max_k,
            "dataset_info": {},
            "pairs": [{"model": m.id, "dataset": d.id, "status": "queued", "stage": "Waiting", "progress": 0.0,
                       "error": None, "metrics": None, "timing": None, "info": None}
                      for m in models for d in datasets],
            "log": [],
        }
        with self.lock:
            self.active[run_id] = run
            self.cancel_flags[run_id] = threading.Event()
            self._save(run)
        self.pool.submit(self._execute, run_id)
        return self.get(run_id)

    # ------------------------------------------------------------ execution
    def _log(self, run: dict, msg: str) -> None:
        with self.lock:
            run["log"].append({"t": now(), "msg": msg})
            run["log"] = run["log"][-300:]

    def _pair(self, run: dict, m: str, d: str) -> dict:
        return next(p for p in run["pairs"] if p["model"] == m and p["dataset"] == d)

    def _set(self, run: dict, pair: dict, **kw) -> None:
        with self.lock:
            pair.update(kw)

    def _execute(self, run_id: str) -> None:
        run = self.active[run_id]
        flag = self.cancel_flags[run_id]
        settings = RunSettings(**run["settings"])
        device = pick_device(settings.device)
        last_save = [0.0]

        def save(force=False):
            if force or time.time() - last_save[0] > 1.0:
                with self.lock:
                    self._save(run)
                last_save[0] = time.time()

        with self.lock:
            run.update(status="running", started=now(), device=device)
        self._log(run, f"Running on {device}")
        save(True)
        data_cache: OrderedDict[str, tuple] = OrderedDict()
        data_errors: dict[str, str] = {}

        def get_data(ds_id: str, pair: dict):
            if ds_id in data_errors:  # don't retry a broken dataset for every model
                raise RuntimeError(data_errors[ds_id])
            if ds_id in data_cache:
                data_cache.move_to_end(ds_id)
                return data_cache[ds_id]
            card = catalog.DatasetCard(**run["dataset_cards"][ds_id])
            try:
                self._set(run, pair, stage="Getting dataset files")
                path = fetch(card.source, lambda msg, frac: self._set(run, pair, stage=msg))
                handle = DatasetHandle(card, path)
                idx = handle.eval_indices(settings.max_trajectories)
                self._set(run, pair, stage=f"Reading {len(idx)} trajectories")
                X = handle.read(idx)
            except Exception as e:  # noqa: BLE001
                data_errors[ds_id] = f"Dataset '{card.name}': {e}"
                raise RuntimeError(data_errors[ds_id]) from e
            info = handle.info() | {"eval_indices": idx}
            with self.lock:
                run["dataset_info"][ds_id] = info
            self._save_truth(run_id, ds_id, X, settings.viz_samples)
            data_cache[ds_id] = (handle, X)
            while sum(v[1].nbytes for v in data_cache.values()) > DATA_CACHE_BYTES and len(data_cache) > 1:
                data_cache.popitem(last=False)
            return handle, X

        try:
            for m_id in run["models"]:
                mcard = catalog.ModelCard(**run["model_cards"][m_id])
                pairs = [self._pair(run, m_id, d) for d in run["datasets"]]
                adapter = None
                try:
                    if flag.is_set():
                        raise Cancelled()
                    cls = REGISTRY[mcard.adapter]
                    adapter = cls(mcard, device, lambda msg: self._log(run, f"[{mcard.name}] {msg}"))
                    for p in pairs:
                        self._set(run, p, stage="Preparing model")
                    save()
                    t0 = time.time()
                    weights = None
                    if cls.needs_weights:
                        self._set(run, pairs[0], status="running", stage="Getting model weights")
                        save(True)
                        weights = fetch(mcard.source, lambda msg, frac: (self._set(run, pairs[0], stage=msg,
                                                                                    progress=frac or 0.0), save()))
                    self._set(run, pairs[0], status="running", stage="Loading model")
                    save(True)
                    adapter.load(weights)
                    load_s = time.time() - t0
                    self._log(run, f"Loaded {mcard.name} in {load_s:.1f}s")
                except Cancelled:
                    raise
                except Exception as e:  # noqa: BLE001 - model failed: every pair with it fails
                    msg = str(e) or e.__class__.__name__
                    self._log(run, f"[{mcard.name}] failed to load: {msg}")
                    self._log(run, traceback.format_exc(limit=4))
                    for p in pairs:
                        self._set(run, p, status="failed", stage="Model failed to load", error=msg)
                    save(True)
                    continue

                for p in pairs:
                    if flag.is_set():
                        raise Cancelled()
                    try:
                        self._set(run, p, status="running", progress=0.0)
                        save(True)
                        handle, X = get_data(p["dataset"], p)
                        self._run_pair(run, p, adapter, X, settings, flag, save, load_s)
                        self._set(run, p, status="done", stage="Done", progress=1.0)
                    except Cancelled:
                        raise
                    except IncompatibleData as e:
                        self._set(run, p, status="skipped", stage="Not compatible", error=str(e))
                        self._log(run, f"[{mcard.name} × {p['dataset']}] skipped: {e}")
                    except Exception as e:  # noqa: BLE001
                        self._set(run, p, status="failed", stage="Failed", error=str(e) or e.__class__.__name__)
                        self._log(run, f"[{mcard.name} × {p['dataset']}] failed: {e}")
                        self._log(run, traceback.format_exc(limit=4))
                    save(True)
                adapter.unload()
            status = "done"
            if any(p["status"] in ("failed", "skipped") for p in run["pairs"]):
                status = "done_with_issues"
        except Cancelled:
            status = "cancelled"
            for p in run["pairs"]:
                if p["status"] in ("queued", "running"):
                    self._set(run, p, status="cancelled", stage="Stopped")
            self._log(run, "Stopped by user")
        except Exception as e:  # noqa: BLE001
            status = "failed"
            self._log(run, f"Run failed: {e}")
            self._log(run, traceback.format_exc(limit=4))
        with self.lock:
            run.update(status=status, finished=now())
            self._save(run)
            self.active.pop(run_id, None)
            self.cancel_flags.pop(run_id, None)

    def _save_truth(self, run_id: str, ds_id: str, X: np.ndarray, viz: int) -> None:
        d = self._dir(run_id) / "data" / ds_id
        if (d / "truth.npy").exists():
            return
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "truth.npy", _compact(X[:viz]))

    def _run_pair(self, run, pair, adapter, X, settings: RunSettings, flag, save, load_s) -> None:
        n, T = X.shape[:2]
        if 2 * X.nbytes > MAX_PAIR_BYTES:
            raise RuntimeError(f"{n} trajectories need {2 * X.nbytes / 1e9:.1f} GB of memory; "
                               "lower 'Trajectories per dataset'")
        adapter.check(X.shape)
        k = adapter.ar_order
        max_k = run["max_ar_order"]
        adapter.begin(X)
        traj = np.arange(n)

        # one-step predictions for every target time t >= k
        t0 = time.time()
        onestep = np.full(X.shape, np.nan, dtype=np.float32)
        targets = list(range(k, T))
        for j, t in enumerate(targets):
            if flag.is_set():
                raise Cancelled()
            onestep[:, t] = adapter.predict(X[:, t - k:t], traj)
            self._set(run, pair, stage=f"One-step predictions {j + 1}/{len(targets)}",
                      progress=0.5 * (j + 1) / len(targets))
            save()
        onestep_s = time.time() - t0

        # rollout from a common start so all models in the run are comparable
        s0 = max(settings.rollout_start, max_k - 1)
        if s0 >= T - 1:
            raise IncompatibleData(f"Rollout start {s0} leaves no steps to predict (dataset has {T} steps)")
        H = T - 1 - s0 if settings.rollout_steps == 0 else min(settings.rollout_steps, T - 1 - s0)
        rollout = np.full(X.shape, np.nan, dtype=np.float32)
        window = X[:, s0 - k + 1:s0 + 1].copy()
        t0 = time.time()
        for h in range(1, H + 1):
            if flag.is_set():
                raise Cancelled()
            p = adapter.predict(window, traj).astype(np.float32)
            rollout[:, s0 + h] = p
            window = np.concatenate([window[:, 1:], p[:, None]], axis=1)
            self._set(run, pair, stage=f"Rollout step {h}/{H}", progress=0.5 + 0.5 * h / H)
            save()
        rollout_s = time.time() - t0

        frm = max_k  # same one-step targets for every model in the run
        one = metrics.summary(onestep[:, frm:], X[:, frm:])
        one["per_field_nrmse"] = metrics.per_field_nrmse(onestep[:, frm:], X[:, frm:])
        one["per_step_nrmse"] = metrics.per_step_nrmse(onestep[:, frm:], X[:, frm:])
        ro_slice = slice(s0 + 1, s0 + H + 1)
        ro = metrics.summary(rollout[:, ro_slice], X[:, ro_slice])
        ro["per_step_nrmse"] = metrics.per_step_nrmse(rollout[:, ro_slice], X[:, ro_slice])
        frames = n * (len(targets) + H)

        d = self._dir(run["id"]) / "pairs" / f"{pair['model']}__{pair['dataset']}"
        d.mkdir(parents=True, exist_ok=True)
        v = settings.viz_samples
        np.save(d / "onestep.npy", _compact(onestep[:v]))
        np.save(d / "rollout.npy", _compact(rollout[:v]))
        self._set(run, pair,
                  metrics={"onestep": one, "rollout": ro, "onestep_from": frm,
                           "rollout_start": s0, "rollout_steps": H, "trajectories": n, "ar_order": k},
                  timing={"load_s": round(load_s, 2), "onestep_s": round(onestep_s, 2),
                          "rollout_s": round(rollout_s, 2),
                          "ms_per_frame": round(1000 * (onestep_s + rollout_s) / max(frames, 1), 2)},
                  info=adapter.info())

    # ------------------------------------------------------------ viewer data
    def frame(self, run_id: str, dataset: str, models: list[str], sample: int, t: int, mode: str,
              field: int, component: int, axis: str, index: Optional[int]) -> dict:
        run = self.get(run_id)
        base = self._dir(run_id)
        truth = np.load(base / "data" / dataset / "truth.npy", mmap_mode="r")
        v, T, F, C, D, H, W = truth.shape
        if not (0 <= sample < v and 0 <= t < T and 0 <= field < F and 0 <= component < C):
            raise ValueError("Sample, time, field or component out of range")
        kind = "1d" if (D == 1 and H == 1) else ("2d" if D == 1 else "3d")
        size = {"d": D, "h": H, "w": W}[axis] if kind == "3d" else 1
        idx = (size // 2 if index is None else int(index)) if kind == "3d" else 0
        idx = min(max(idx, 0), max(size - 1, 0))

        def sl(arr, tt=None):  # arr (v,T,F,C,D,H,W) -> 1D or 2D slice, or all times when tt is None
            a = arr[sample, :, field, component] if tt is None else arr[sample, tt, field, component]
            lead = (slice(None),) if tt is None else ()
            if kind == "1d":
                return np.asarray(a[lead + (0, 0)], dtype=np.float32)
            if kind == "2d":
                return np.asarray(a[lead + (0,)], dtype=np.float32)
            pick = {"d": (idx,), "h": (slice(None), idx), "w": (slice(None), slice(None), idx)}[axis]
            return np.asarray(a[lead + pick], dtype=np.float32)

        truth_all = sl(truth)
        lo, hi = float(np.nanmin(truth_all)), float(np.nanmax(truth_all))
        out = {"kind": kind, "t": t, "T": T, "sample": sample, "axis": axis, "index": idx,
               "grid": [D, H, W], "range": [lo, hi], "truth": _pack(sl(truth, t)),
               "models": {}, "error_max": 0.0}
        err_max = 0.0
        for m in models:
            f = base / "pairs" / f"{m}__{dataset}" / f"{mode}.npy"
            if not f.exists():
                out["models"][m] = None
                continue
            pred = np.load(f, mmap_mode="r")
            p_all = sl(pred)
            e_all = np.abs(p_all - truth_all)
            if np.isfinite(e_all).any():
                err_max = max(err_max, float(np.nanmax(e_all)))
            p = p_all[t]
            out["models"][m] = None if not np.isfinite(p).any() else {
                "pred": _pack(p), "error": _pack(np.abs(p - truth_all[t]))}
        out["error_max"] = err_max
        return out


def _compact(a: np.ndarray) -> np.ndarray:
    """Store viewer arrays as float16 when they are large (halves disk use)."""
    return a.astype(np.float16) if a.nbytes > 200 * 1024 ** 2 else a.astype(np.float32)


def _pack(a: np.ndarray, max_side: int = 256):
    """Downsample big slices, round to ~5 significant digits, NaN -> None."""
    if a.ndim == 2:
        sy = max(1, int(np.ceil(a.shape[0] / max_side)))
        sx = max(1, int(np.ceil(a.shape[1] / max_side)))
        a = a[::sy, ::sx]
    elif a.ndim == 1 and a.shape[0] > 4096:
        a = a[:: int(np.ceil(a.shape[0] / 4096))]
    finite = a[np.isfinite(a)]
    scale = float(np.max(np.abs(finite))) if finite.size else 1.0
    dec = int(max(0, 5 - np.floor(np.log10(scale)))) if scale > 0 else 5
    a = np.round(a.astype(np.float64), min(dec, 12))
    return np.where(np.isfinite(a), a, None).tolist()


def metrics_csv(run: dict) -> str:
    cols = ["model", "dataset", "status", "trajectories",
            "onestep_mse", "onestep_rmse", "onestep_mae", "onestep_nrmse", "onestep_vrmse",
            "rollout_steps", "rollout_nrmse", "rollout_vrmse", "rollout_final_nrmse",
            "ms_per_frame", "parameters_m", "error"]
    lines = [",".join(cols)]
    for p in run["pairs"]:
        m = p.get("metrics") or {}
        one, ro = m.get("onestep", {}), m.get("rollout", {})
        curve = ro.get("per_step_nrmse") or [None]
        row = [p["model"], p["dataset"], p["status"], m.get("trajectories"),
               one.get("mse"), one.get("rmse"), one.get("mae"), one.get("nrmse"), one.get("vrmse"),
               m.get("rollout_steps"), ro.get("nrmse"), ro.get("vrmse"), curve[-1],
               (p.get("timing") or {}).get("ms_per_frame"), (p.get("info") or {}).get("parameters_m"),
               (p.get("error") or "").replace(",", ";").replace("\n", " ")]
        lines.append(",".join("" if x is None else str(x) for x in row))
    return "\n".join(lines) + "\n"
