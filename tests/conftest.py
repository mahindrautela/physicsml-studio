"""Shared test setup: a temporary catalog and workspace with three small datasets.

Run all tests with:  python -m pytest -q
Nothing here touches your own catalog/ or workspace/ folders.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for fake_models.py

DATASETS = {
    # id: (array shape, card layout, evaluation split)
    "two-d": ((4, 6, 2, 16, 16), {"fields": 2, "height": 16, "width": 16}, {"mode": "all"}),          # N T F H W
    "chlast": ((3, 5, 8, 8, 8, 1), {"axes": "NTDHWF"}, {"mode": "all"}),                             # 3D, channels last
    "one-d": ((5, 7, 32), {"axes": "NTW"}, {"mode": "last_fraction", "fraction": 0.4}),              # 1D
}
MODELS = {  # fake model types from fake_models.py
    "copy": {"name": "Copy", "adapter": "test_copy"},
    "trend": {"name": "Trend", "adapter": "test_trend"},
    "slow": {"name": "Slow copy", "adapter": "test_copy", "config": {"delay": 0.3}},
}


@pytest.fixture(scope="session", autouse=True)
def workspace(tmp_path_factory) -> dict:
    """Write the test catalog and data, and point the app at them before any test imports it."""
    base = tmp_path_factory.mktemp("studio")
    catalog, home, data = base / "catalog", base / "home", base / "data"
    for folder in (catalog / "models", catalog / "datasets", data):
        folder.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for ds_id, (shape, layout, split) in DATASETS.items():
        np.save(data / f"{ds_id}.npy", rng.standard_normal(shape).astype(np.float32))
        card = {"id": ds_id, "name": ds_id, "source": {"type": "local", "path": str(data / f"{ds_id}.npy")},
                "layout": layout, "eval": split}
        (catalog / "datasets" / f"{ds_id}.yaml").write_text(yaml.safe_dump(card))
    for model_id, card in MODELS.items():
        (catalog / "models" / f"{model_id}.yaml").write_text(yaml.safe_dump({"id": model_id, **card}))
    os.environ["STUDIO_CATALOG"], os.environ["STUDIO_HOME"] = str(catalog), str(home)
    return {"catalog": catalog, "home": home, "data": data}


@pytest.fixture(scope="session")
def client(workspace):
    from fastapi.testclient import TestClient

    import app.main
    import fake_models  # noqa: F401 - registers the test_copy and test_trend model types

    return TestClient(app.main.app)


@pytest.fixture
def wait_for_run(client):
    """Poll a run until it stops running; return its record."""
    def wait(run_id: str, timeout: float = 120) -> dict:
        deadline = time.time() + timeout
        while time.time() < deadline:
            run = client.get(f"/api/runs/{run_id}").json()
            if run["status"] not in ("queued", "running"):
                return run
            time.sleep(0.2)
        raise TimeoutError(run_id)
    return wait
