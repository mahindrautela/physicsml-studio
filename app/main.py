"""HTTP API + static web page. Start with:  python studio.py"""
from __future__ import annotations

import importlib.util
import platform
import sys
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ValidationError

from . import catalog
from .adapters import LOAD_ERRORS, REGISTRY
from .catalog import DatasetCard, ModelCard, Source
from .config import CATALOG_DIR, HOME_DIR, STATIC_DIR
from .datasets import list_variables
from .jobs import RunManager, RunSettings, metrics_csv
from .sources import cached_path, fetch, parse_link

app = FastAPI(title="Inference Studio", version="0.1.0")
runs = RunManager()


def _err(status: int, e: Exception):
    if isinstance(e, ValidationError):
        msg = "; ".join(f"{'.'.join(str(x) for x in er['loc'])}: {er['msg']}" for er in e.errors())
    else:
        msg = str(e)
    raise HTTPException(status_code=status, detail=msg)


# ---------------------------------------------------------------- system
@app.get("/api/system")
def system():
    info = {"python": platform.python_version(), "platform": platform.platform(terse=True),
            "catalog_dir": str(CATALOG_DIR), "home_dir": str(HOME_DIR),
            "adapters": sorted(REGISTRY), "adapter_errors": LOAD_ERRORS,
            "devices": ["auto", "cpu"], "torch": None, "gpus": []}
    try:
        import torch

        info["torch"] = torch.__version__
        for i in range(torch.cuda.device_count()):
            info["devices"].append(f"cuda:{i}")
            info["gpus"].append(torch.cuda.get_device_name(i))
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            info["devices"].append("mps")
    except ImportError:
        pass
    return info


# ---------------------------------------------------------------- catalog
def _model_status(c: ModelCard) -> dict:
    if c.adapter not in REGISTRY:
        why = LOAD_ERRORS.get(c.adapter)
        return {"state": "error", "message": f"Adapter '{c.adapter}' isn't installed" + (f": {why}" if why else "")}
    missing = [r for r in c.requires if importlib.util.find_spec(r) is None]
    if missing:
        return {"state": "needs_install", "message": c.install_hint or f"pip install {' '.join(missing)}"}
    if not REGISTRY[c.adapter].needs_weights:
        return {"state": "ready", "message": "No download needed"}
    if cached_path(c.source) is not None:
        return {"state": "ready", "message": "Weights downloaded"}
    if c.source.type == "local":
        return {"state": "error", "message": f"Weights file not found: {c.source.path}"}
    return {"state": "download", "message": "Weights download on first run"}


def _dataset_status(c: DatasetCard) -> dict:
    if cached_path(c.source) is not None:
        return {"state": "ready", "message": "File available"}
    if c.source.type == "local":
        return {"state": "error", "message": f"File not found: {c.source.path}"}
    return {"state": "download", "message": "Downloads on first run"}


def _card_out(c, status: dict) -> dict:
    return c.model_dump() | {"status": status, "source_text": c.source.describe(), "source_url": c.source.web_url()}


@app.get("/api/catalog")
def get_catalog():
    models, mprob = catalog.load_all("models")
    datasets, dprob = catalog.load_all("datasets")
    adapters = {name: {"doc": (cls.__doc__ or sys.modules[cls.__module__].__doc__ or "").strip().split("\n")[0],
                       "needs_weights": cls.needs_weights, "ar_order": cls.ar_order}
                for name, cls in sorted(REGISTRY.items())}
    return {"models": [_card_out(c, _model_status(c)) for c in models],
            "datasets": [_card_out(c, _dataset_status(c)) for c in datasets],
            "problems": [{"kind": "model", **p} for p in mprob] + [{"kind": "dataset", **p} for p in dprob],
            "adapters": adapters}


@app.post("/api/models", status_code=201)
def add_model(card: dict, overwrite: bool = False):
    try:
        c = ModelCard(**card)
        catalog.save("models", c, overwrite=overwrite)
    except FileExistsError as e:
        _err(409, e)
    except (ValidationError, ValueError) as e:
        _err(422, e)
    return _card_out(c, _model_status(c))


@app.post("/api/datasets", status_code=201)
def add_dataset(card: dict, overwrite: bool = False):
    try:
        c = DatasetCard(**card)
        catalog.save("datasets", c, overwrite=overwrite)
    except FileExistsError as e:
        _err(409, e)
    except (ValidationError, ValueError) as e:
        _err(422, e)
    return _card_out(c, _dataset_status(c))


def _delete_card(kind: str, card_id: str):
    try:
        catalog.delete(kind, card_id)
    except KeyError as e:
        _err(404, e)
    return {"deleted": card_id}


@app.delete("/api/models/{card_id}")
def delete_model(card_id: str):
    return _delete_card("models", card_id)


@app.delete("/api/datasets/{card_id}")
def delete_dataset(card_id: str):
    return _delete_card("datasets", card_id)


class LinkIn(BaseModel):
    link: str


@app.post("/api/parse-link")
def api_parse_link(body: LinkIn):
    try:
        return parse_link(body.link).model_dump(exclude_none=True)
    except (ValidationError, ValueError) as e:
        _err(422, e)


class InspectIn(BaseModel):
    source: Source
    format: Literal["auto", "npy", "nc", "h5"] = "auto"


@app.post("/api/inspect")
def inspect(body: InspectIn):
    """Fetch a data file (downloading it if needed) and list the arrays inside."""
    try:
        path = fetch(body.source)
        return {"file": path.name, "variables": list_variables(path, body.format)}
    except Exception as e:  # noqa: BLE001
        _err(400, e)


# ---------------------------------------------------------------- runs
class RunIn(BaseModel):
    models: list[str]
    datasets: list[str]
    settings: RunSettings = RunSettings()
    name: Optional[str] = None


@app.post("/api/runs", status_code=201)
def start_run(body: RunIn):
    try:
        return runs.create(body.models, body.datasets, body.settings, body.name)
    except KeyError as e:
        _err(404, e)
    except ValueError as e:
        _err(422, e)


@app.get("/api/runs")
def list_runs():
    return runs.list()


@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    try:
        return runs.get(run_id)
    except KeyError:
        _err(404, KeyError(f"No run {run_id}"))


@app.post("/api/runs/{run_id}/cancel")
def cancel_run(run_id: str):
    try:
        runs.cancel(run_id)
    except KeyError:
        _err(409, RuntimeError("That run isn't running"))
    return {"cancelling": run_id}


@app.delete("/api/runs/{run_id}")
def delete_run(run_id: str):
    try:
        runs.delete(run_id)
    except RuntimeError as e:
        _err(409, e)
    except (KeyError, FileNotFoundError):
        _err(404, KeyError(f"No run {run_id}"))
    return {"deleted": run_id}


@app.get("/api/runs/{run_id}/frame")
def frame(run_id: str, dataset: str, models: str = "", sample: int = 0, t: int = 1,
          mode: Literal["onestep", "rollout"] = "rollout", field: int = 0, component: int = 0,
          axis: Literal["d", "h", "w"] = "d", index: Optional[int] = Query(None)):
    try:
        return runs.frame(run_id, dataset, [m for m in models.split(",") if m], sample, t, mode,
                          field, component, axis, index)
    except FileNotFoundError:
        _err(404, FileNotFoundError("No saved data for that dataset in this run yet"))
    except (KeyError, ValueError) as e:
        _err(400, e)


@app.get("/api/runs/{run_id}/metrics.csv", response_class=PlainTextResponse)
def run_csv(run_id: str):
    try:
        run = runs.get(run_id)
    except KeyError:
        _err(404, KeyError(f"No run {run_id}"))
    return PlainTextResponse(metrics_csv(run), headers={
        "Content-Disposition": f'attachment; filename="metrics-{run_id}.csv"'})


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
