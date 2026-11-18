"""What happens during a run.

Each model is loaded once, then evaluated on every dataset:
  1. read the evaluation trajectories                       (datasets.py)
  2. one-step: predict every frame from the true frame(s) before it
  3. rollout: start from true frames, then feed predictions back in
  4. compare with the truth                                  (metrics.py)
  5. save predictions for the field viewer                   (frames.py reads them)

Progress and results are written into the run record as they happen; the page
polls that record (see runs.py).
"""
from __future__ import annotations

import threading
import time
import traceback
from collections import OrderedDict
from typing import Callable, Optional

import numpy as np

from . import metrics
from .adapters import REGISTRY, Adapter, IncompatibleData
from .catalog import DatasetCard, ModelCard
from .datasets import DatasetHandle
from .devices import pick_device
from .runs import RunSettings, RunStore, now, prediction_file, truth_file
from .sources import fetch

DATA_CACHE_BYTES = 2 * 1024**3   # keep up to 2 GB of loaded trajectories while moving between models
MAX_PAIR_BYTES = 3 * 1024**3     # refuse a model × dataset pair that would need more memory than this
LARGE_ARRAY_BYTES = 200 * 1024**2  # viewer arrays larger than this are saved as float16


class Stopped(Exception):
    """The user pressed Stop run."""


# ---------------------------------------------------------------- predictions
def predict_one_step(adapter: Adapter, X: np.ndarray, on_step: Optional[Callable[[int, int], None]] = None) -> np.ndarray:
    """Predict every frame t >= k from the k true frames before it.

    X is (n, T, F, C, D, H, W). Returns the same shape; frames that can't be
    predicted (t < k) are NaN.
    """
    k = adapter.ar_order
    out = np.full(X.shape, np.nan, dtype=np.float32)
    trajectories = np.arange(X.shape[0])
    targets = range(k, X.shape[1])
    for i, t in enumerate(targets, 1):
        out[:, t] = adapter.predict(X[:, t - k:t], trajectories)
        if on_step:
            on_step(i, len(targets))
    return out


def predict_rollout(adapter: Adapter, X: np.ndarray, start: int, steps: int,
                    on_step: Optional[Callable[[int, int], None]] = None) -> np.ndarray:
    """Give the model the true frames up to `start`, then let it predict `steps`
    frames, each from its own previous predictions. Returns X's shape, NaN
    outside start+1 .. start+steps."""
    k = adapter.ar_order
    out = np.full(X.shape, np.nan, dtype=np.float32)
    trajectories = np.arange(X.shape[0])
    window = X[:, start - k + 1:start + 1].copy()  # the k most recent frames
    for h in range(1, steps + 1):
        frame = adapter.predict(window, trajectories).astype(np.float32)
        out[:, start + h] = frame
        window = np.concatenate([window[:, 1:], frame[:, None]], axis=1)
        if on_step:
            on_step(h, steps)
    return out


def execute_run(run: dict, store: RunStore, stop: threading.Event) -> None:
    """Entry point used by RunStore."""
    Runner(run, store, stop).execute()


# ---------------------------------------------------------------- one run
class Runner:
    def __init__(self, run: dict, store: RunStore, stop: threading.Event):
        self.run = run
        self.store = store
        self.stop = stop
        self.settings = RunSettings(**run["settings"])
        self.device = pick_device(self.settings.device)
        self._data: OrderedDict[str, np.ndarray] = OrderedDict()  # dataset id -> trajectories
        self._data_errors: dict[str, str] = {}                     # dataset id -> why it can't be read
        self._last_save = 0.0

    # ---- top level
    def execute(self) -> None:
        self._set_run(status="running", started=now(), device=self.device)
        self.log(f"Running on {self.device}")
        try:
            for model_id in self.run["models"]:
                self._run_model(model_id)
            failed = any(p["status"] in ("failed", "skipped") for p in self.run["pairs"])
            status = "done_with_issues" if failed else "done"
        except Stopped:
            status = "cancelled"
            for pair in self.run["pairs"]:
                if pair["status"] in ("queued", "running"):
                    self._update(pair, status="cancelled", stage="Stopped")
            self.log("Stopped by user")
        except Exception as e:  # noqa: BLE001 - record any unexpected error instead of losing it
            status = "failed"
            self.log(f"Run failed: {e}\n{traceback.format_exc(limit=4)}")
        self._set_run(status=status, finished=now())

    def _run_model(self, model_id: str) -> None:
        card = ModelCard(**self.run["model_cards"][model_id])
        pairs = [p for p in self.run["pairs"] if p["model"] == model_id]
        self._check_stop()
        try:
            adapter, load_seconds = self._load_model(card, pairs[0])
        except Exception as e:  # noqa: BLE001 - a model that won't load fails all of its pairs
            message = str(e) or e.__class__.__name__
            self.log(f"[{card.name}] failed to load: {message}\n{traceback.format_exc(limit=4)}")
            for pair in pairs:
                self._update(pair, status="failed", stage="Model failed to load", error=message)
            self.save(force=True)
            return
        try:
            for pair in pairs:
                self._check_stop()
                self._run_pair(card, pair, adapter, load_seconds)
        finally:
            adapter.unload()

    def _run_pair(self, card: ModelCard, pair: dict, adapter: Adapter, load_seconds: float) -> None:
        label = f"[{card.name} × {pair['dataset']}]"
        try:
            self._update(pair, status="running", progress=0.0)
            self.save(force=True)
            X = self._load_dataset(pair["dataset"], pair)
            self._evaluate(pair, adapter, X, load_seconds)
            self._update(pair, status="done", stage="Done", progress=1.0)
        except Stopped:
            raise
        except IncompatibleData as e:
            self._update(pair, status="skipped", stage="Not compatible", error=str(e))
            self.log(f"{label} skipped: {e}")
        except Exception as e:  # noqa: BLE001
            self._update(pair, status="failed", stage="Failed", error=str(e) or e.__class__.__name__)
            self.log(f"{label} failed: {e}\n{traceback.format_exc(limit=4)}")
        self.save(force=True)

    # ---- loading
    def _load_model(self, card: ModelCard, first_pair: dict) -> tuple[Adapter, float]:
        cls = REGISTRY[card.adapter]
        adapter = cls(card, self.device, lambda msg: self.log(f"[{card.name}] {msg}"))
        started = time.time()
        weights = None
        if cls.needs_weights:
            self._update(first_pair, status="running", stage="Getting model weights")
            self.save(force=True)
            weights = fetch(card.source, lambda msg, frac: self._progress(first_pair, msg, frac or 0.0))
        self._update(first_pair, status="running", stage="Loading model")
        self.save(force=True)
        adapter.load(weights)
        seconds = time.time() - started
        self.log(f"Loaded {card.name} in {seconds:.1f}s")
        return adapter, seconds

    def _load_dataset(self, dataset_id: str, pair: dict) -> np.ndarray:
        """Trajectories to evaluate, read once per run and shared by all models."""
        if dataset_id in self._data_errors:  # don't retry a broken dataset for every model
            raise RuntimeError(self._data_errors[dataset_id])
        if dataset_id in self._data:
            self._data.move_to_end(dataset_id)
            return self._data[dataset_id]
        card = DatasetCard(**self.run["dataset_cards"][dataset_id])
        try:
            self._update(pair, stage="Getting dataset files")
            path = fetch(card.source, lambda msg, frac: self._update(pair, stage=msg))
            handle = DatasetHandle(card, path)
            indices = handle.eval_indices(self.settings.max_trajectories)
            self._update(pair, stage=f"Reading {len(indices)} trajectories")
            X = handle.read(indices)
        except Exception as e:  # noqa: BLE001
            self._data_errors[dataset_id] = f"Dataset '{card.name}': {e}"
            raise RuntimeError(self._data_errors[dataset_id]) from e

        with self.store.lock:
            self.run["dataset_info"][dataset_id] = handle.info() | {"eval_indices": indices}
        truth = truth_file(self.run["id"], dataset_id)
        if not truth.exists():  # written once; the viewer may have it open, and Windows locks open files
            _save_array(truth, X[:self.settings.viz_samples])
        self._data[dataset_id] = X
        while len(self._data) > 1 and sum(a.nbytes for a in self._data.values()) > DATA_CACHE_BYTES:
            self._data.popitem(last=False)  # forget the least recently used dataset
        return X

    # ---- evaluation
    def _evaluate(self, pair: dict, adapter: Adapter, X: np.ndarray, load_seconds: float) -> None:
        n, T = X.shape[:2]
        if 2 * X.nbytes > MAX_PAIR_BYTES:
            raise RuntimeError(f"{n} trajectories need {2 * X.nbytes / 1e9:.1f} GB of memory; "
                               "lower 'Trajectories per dataset'")
        adapter.check(X.shape)
        adapter.begin(X)
        k = adapter.ar_order
        first_scored = self.run["max_ar_order"]  # same scored steps for every model in the run

        # rollout start and length, shared by every model in the run
        start = max(self.settings.rollout_start, first_scored - 1)
        if start >= T - 1:
            raise IncompatibleData(f"Rollout start {start} leaves no steps to predict (dataset has {T} steps)")
        steps = T - 1 - start
        if self.settings.rollout_steps:
            steps = min(self.settings.rollout_steps, steps)

        def progress(phase: str, offset: float):
            return lambda i, total: self._progress(pair, f"{phase} {i}/{total}", offset + 0.5 * i / total)

        t0 = time.time()
        onestep = predict_one_step(adapter, X, self._stoppable(progress("One-step predictions", 0.0)))
        onestep_seconds = time.time() - t0
        t0 = time.time()
        rollout = predict_rollout(adapter, X, start, steps, self._stoppable(progress("Rollout step", 0.5)))
        rollout_seconds = time.time() - t0

        scored = slice(first_scored, T)
        rolled = slice(start + 1, start + steps + 1)
        one = metrics.summary(onestep[:, scored], X[:, scored])
        one["per_field_nrmse"] = metrics.per_field_nrmse(onestep[:, scored], X[:, scored])
        one["per_step_nrmse"] = metrics.per_step_nrmse(onestep[:, scored], X[:, scored])
        roll = metrics.summary(rollout[:, rolled], X[:, rolled])
        roll["per_step_nrmse"] = metrics.per_step_nrmse(rollout[:, rolled], X[:, rolled])

        keep = self.settings.viz_samples
        _save_array(prediction_file(self.run["id"], pair["model"], pair["dataset"], "onestep"), onestep[:keep])
        _save_array(prediction_file(self.run["id"], pair["model"], pair["dataset"], "rollout"), rollout[:keep])

        frames = n * ((T - k) + steps)
        self._update(
            pair,
            metrics={"onestep": one, "rollout": roll, "onestep_from": first_scored, "rollout_start": start,
                     "rollout_steps": steps, "trajectories": n, "ar_order": k},
            timing={"load_s": round(load_seconds, 2), "onestep_s": round(onestep_seconds, 2),
                    "rollout_s": round(rollout_seconds, 2),
                    "ms_per_frame": round(1000 * (onestep_seconds + rollout_seconds) / max(frames, 1), 2)},
            info=adapter.info(),
        )

    # ---- record updates
    def log(self, message: str) -> None:
        with self.store.lock:
            self.run["log"].append({"t": now(), "msg": message})
            del self.run["log"][:-300]  # keep the last 300 lines

    def save(self, force: bool = False) -> None:
        """Write the record to disk; at most once a second unless forced."""
        if force or time.time() - self._last_save > 1.0:
            self.store.save(self.run)
            self._last_save = time.time()

    def _update(self, pair: dict, **fields) -> None:
        with self.store.lock:
            pair.update(fields)

    def _progress(self, pair: dict, stage: str, fraction: float) -> None:
        self._update(pair, stage=stage, progress=fraction)
        self.save()

    def _set_run(self, **fields) -> None:
        with self.store.lock:
            self.run.update(fields)
        self.save(force=True)

    def _check_stop(self) -> None:
        if self.stop.is_set():
            raise Stopped()

    def _stoppable(self, callback: Callable[[int, int], None]) -> Callable[[int, int], None]:
        """Progress callback that also stops the run between steps when asked."""
        def wrapped(i: int, total: int) -> None:
            callback(i, total)
            self._check_stop()
        return wrapped


def _save_array(path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array.astype(np.float16 if array.nbytes > LARGE_ARRAY_BYTES else np.float32))
