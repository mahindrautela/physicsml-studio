"""Web server: the page in static/ and the JSON API it calls.

Routes stay thin; the work happens in the module each section names.
Start the server with:  python studio.py
"""
from __future__ import annotations

import mimetypes
import platform
import sys
from typing import Literal, NoReturn, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ValidationError

from . import catalog, frames, metrics
from .adapters import LOAD_ERRORS, REGISTRY
from .catalog import DatasetCard, ModelCard, Source
from .config import CATALOG_DIR, HOME_DIR, STATIC_DIR
from .datasets import list_variables
from .devices import available_devices, gpu_names
from .readiness import dataset_status, model_status
from .runner import execute_run
from .runs import RunSettings, RunStore
from .sources import fetch, parse_link

app = FastAPI(title="PhysicsML Studio")
runs = RunStore(execute=execute_run)


def fail(status: int, error: Exception) -> NoReturn:
    """Send an error to the page with a readable message."""
    if isinstance(error, ValidationError):
        message = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in error.errors())
    else:
        message = str(error).strip("'\"")
    raise HTTPException(status_code=status, detail=message)


# ---------------------------------------------------------------- system  (devices.py)
@app.get("/api/system")
def system():
    try:
        import torch
        torch_version = torch.__version__
    except ImportError:
        torch_version = None
    return {"python": platform.python_version(), "torch": torch_version,
            "devices": available_devices(), "gpus": gpu_names(),
            "catalog_dir": str(CATALOG_DIR), "home_dir": str(HOME_DIR),
            "adapters": sorted(REGISTRY), "adapter_errors": LOAD_ERRORS}


# ---------------------------------------------------------------- catalog  (catalog.py, readiness.py)
def _card_json(card, status: dict) -> dict:
    return card.model_dump() | {"status": status, "source_text": card.source.describe(),
                                "source_url": card.source.web_url()}


@app.get("/api/catalog")
def get_catalog():
    models, model_problems = catalog.load_all("models")
    datasets, dataset_problems = catalog.load_all("datasets")
    model_types = {
        name: {"doc": (cls.__doc__ or sys.modules[cls.__module__].__doc__ or "").strip().split("\n")[0],
               "needs_weights": cls.needs_weights, "ar_order": cls.ar_order}
        for name, cls in sorted(REGISTRY.items())
    }
    return {"models": [_card_json(c, model_status(c)) for c in models],
            "datasets": [_card_json(c, dataset_status(c)) for c in datasets],
            "problems": [{"kind": "model", **p} for p in model_problems]
                        + [{"kind": "dataset", **p} for p in dataset_problems],
            "adapters": model_types}


@app.post("/api/models", status_code=201)
def add_model(card: dict, overwrite: bool = False):
    try:
        model = ModelCard(**card)
        catalog.save("models", model, overwrite=overwrite)
    except FileExistsError as e:
        fail(409, e)
    except (ValidationError, ValueError) as e:
        fail(422, e)
    return _card_json(model, model_status(model))


@app.post("/api/datasets", status_code=201)
def add_dataset(card: dict, overwrite: bool = False):
    try:
        dataset = DatasetCard(**card)
        catalog.save("datasets", dataset, overwrite=overwrite)
    except FileExistsError as e:
        fail(409, e)
    except (ValidationError, ValueError) as e:
        fail(422, e)
    return _card_json(dataset, dataset_status(dataset))


@app.delete("/api/models/{card_id}")
def delete_model(card_id: str):
    try:
        catalog.delete("models", card_id)
    except KeyError as e:
        fail(404, e)
    return {"deleted": card_id}


@app.delete("/api/datasets/{card_id}")
def delete_dataset(card_id: str):
    try:
        catalog.delete("datasets", card_id)
    except KeyError as e:
        fail(404, e)
    return {"deleted": card_id}


class LinkIn(BaseModel):
    link: str


@app.post("/api/parse-link")
def read_link(body: LinkIn):
    """Turn a pasted Hugging Face / GitHub / URL / path into a card source."""
    try:
        return parse_link(body.link).model_dump(exclude_none=True)
    except (ValidationError, ValueError) as e:
        fail(422, e)


class InspectIn(BaseModel):
    source: Source
    format: Literal["auto", "npy", "nc", "h5"] = "auto"


@app.post("/api/inspect")
def inspect(body: InspectIn):
    """Fetch a data file (downloading it if needed) and list the arrays inside."""
    try:
        path = fetch(body.source)
        return {"file": path.name, "variables": list_variables(path, body.format)}
    except Exception as e:  # noqa: BLE001 - any problem goes back to the form
        fail(400, e)


# ---------------------------------------------------------------- runs  (runs.py, runner.py, frames.py)
class RunIn(BaseModel):
    models: list[str]
    datasets: list[str]
    settings: RunSettings = RunSettings()
    name: Optional[str] = None


@app.post("/api/runs", status_code=201)
def start_run(body: RunIn):
    try:
        return runs.start(body.models, body.datasets, body.settings, body.name)
    except KeyError as e:
        fail(404, e)
    except ValueError as e:
        fail(422, e)


@app.get("/api/runs")
def list_runs():
    return runs.list()


@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    try:
        return runs.get(run_id)
    except KeyError:
        fail(404, KeyError(f"No run {run_id}"))


@app.post("/api/runs/{run_id}/cancel")
def stop_run(run_id: str):
    try:
        runs.stop(run_id)
    except KeyError:
        fail(409, RuntimeError("That run isn't running"))
    return {"cancelling": run_id}


@app.delete("/api/runs/{run_id}")
def delete_run(run_id: str):
    try:
        runs.delete(run_id)
    except RuntimeError as e:
        fail(409, e)
    except KeyError:
        fail(404, KeyError(f"No run {run_id}"))
    return {"deleted": run_id}


@app.get("/api/runs/{run_id}/frame")
def get_frame(run_id: str, dataset: str, models: str = "", sample: int = 0, t: int = 1,
              mode: Literal["onestep", "rollout"] = "rollout", field: int = 0, component: int = 0,
              axis: Literal["d", "h", "w"] = "d", index: Optional[int] = Query(None)):
    try:
        return frames.frame(run_id, dataset, [m for m in models.split(",") if m],
                            sample, t, mode, field, component, axis, index)
    except FileNotFoundError:
        fail(404, FileNotFoundError("No saved data for that dataset in this run yet"))
    except (KeyError, ValueError) as e:
        fail(400, e)


@app.get("/api/runs/{run_id}/metrics.csv", response_class=PlainTextResponse)
def metrics_csv(run_id: str):
    try:
        run = runs.get(run_id)
    except KeyError:
        fail(404, KeyError(f"No run {run_id}"))
    return PlainTextResponse(metrics.to_csv(run), headers={
        "Content-Disposition": f'attachment; filename="metrics-{run_id}.csv"'})


# The page itself. Mounted last so it doesn't shadow the API routes.
# Windows can map .js to "text/plain" in its registry, and browsers refuse to run
# JavaScript modules served that way (the page stays blank). Set the types explicitly.
mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("text/css", ".css")
mimetypes.add_type("font/woff2", ".woff2")
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
