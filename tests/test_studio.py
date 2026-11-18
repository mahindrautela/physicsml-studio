"""Backend tests. Run with:  python -m pytest -q

They use a temporary catalog and workspace, so they never touch your own files.
MORPH tests run only if morph_pde is installed (they use random weights, no download).
"""
from __future__ import annotations

import importlib
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    base = tmp_path_factory.mktemp("studio")
    cat, home = base / "catalog", base / "home"
    (cat / "models").mkdir(parents=True)
    (cat / "datasets").mkdir(parents=True)
    os.environ["STUDIO_CATALOG"], os.environ["STUDIO_HOME"] = str(cat), str(home)
    for mod in [m for m in list(sys.modules) if m == "app" or m.startswith("app.")]:
        del sys.modules[mod]
    from fastapi.testclient import TestClient

    import app.main as main

    rng = np.random.default_rng(0)
    data = home / "data"
    data.mkdir(parents=True)
    np.save(data / "two_d.npy", rng.standard_normal((4, 6, 2, 16, 16)).astype(np.float32))      # N T F H W
    np.save(data / "chlast.npy", rng.standard_normal((3, 5, 8, 8, 8, 1)).astype(np.float32))     # N T D H W F
    np.save(data / "one_d.npy", rng.standard_normal((5, 7, 32)).astype(np.float32))              # N T W
    cards = {
        "datasets/two_d.yaml": {"id": "two-d", "name": "2D", "source": {"type": "local", "path": str(data / "two_d.npy")},
                                "layout": {"fields": 2, "components": 1, "depth": 1, "height": 16, "width": 16}, "eval": {"mode": "all"}},
        "datasets/chlast.yaml": {"id": "chlast", "name": "3D channels-last", "source": {"type": "local", "path": str(data / "chlast.npy")},
                                 "layout": {"axes": "NTDHWF"}, "eval": {"mode": "all"}},
        "datasets/one_d.yaml": {"id": "one-d", "name": "1D", "source": {"type": "local", "path": str(data / "one_d.npy")},
                                "layout": {"axes": "NTW"}, "eval": {"mode": "last_fraction", "fraction": 0.4}},
        "models/persistence.yaml": {"id": "persistence", "name": "Persistence", "adapter": "persistence"},
        "models/linear.yaml": {"id": "linear", "name": "Linear", "adapter": "linear_extrapolation"},
    }
    for rel, card in cards.items():
        (cat / rel).write_text(yaml.safe_dump(card))
    c = TestClient(main.app)
    c.home = home
    c.cat = cat
    return c


def wait(client, run_id, timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        run = client.get(f"/api/runs/{run_id}").json()
        if run["status"] not in ("queued", "running"):
            return run
        time.sleep(0.2)
    raise TimeoutError(run_id)


def test_catalog_lists_cards(client):
    cat = client.get("/api/catalog").json()
    assert {m["id"] for m in cat["models"]} >= {"persistence", "linear"}
    assert all(d["status"]["state"] == "ready" for d in cat["datasets"])
    assert cat["problems"] == []


def test_parse_links(client):
    hf = client.post("/api/parse-link", json={"link": "https://huggingface.co/mahindrautela/MORPH/blob/main/models/FM/morph-Ti-FM-max_ar1_ep225.pth"}).json()
    assert hf == {"type": "huggingface", "repo_id": "mahindrautela/MORPH", "repo_type": "model",
                  "subfolder": "models/FM", "filename": "morph-Ti-FM-max_ar1_ep225.pth"}
    ds = client.post("/api/parse-link", json={"link": "https://huggingface.co/datasets/camlab-ethz/FNS-KF/blob/main/solution_0.nc"}).json()
    assert ds["repo_type"] == "dataset" and ds["filename"] == "solution_0.nc"
    gh = client.post("/api/parse-link", json={"link": "https://github.com/lanl/MORPH/blob/package-morph/docs/user-guide.md"}).json()
    assert gh == {"type": "github", "repo_id": "lanl/MORPH", "repo_type": "model", "revision": "package-morph", "filename": "docs/user-guide.md"}
    assert client.post("/api/parse-link", json={"link": "/data/x.npy"}).json()["type"] == "local"


def test_inspect_lists_arrays(client):
    res = client.post("/api/inspect", json={"source": {"type": "local", "path": str(client.home / "data" / "chlast.npy")}}).json()
    assert res["variables"][0]["shape"] == [3, 5, 8, 8, 8, 1]


def test_dataset_layouts():
    from app.catalog import DatasetCard
    from app.datasets import DatasetHandle

    home = Path(os.environ["STUDIO_HOME"]) / "data"
    h = DatasetHandle(DatasetCard(id="a", name="a", source={"type": "local", "path": str(home / "chlast.npy")},
                                  layout={"axes": "NTDHWF"}), home / "chlast.npy")
    assert h.shape == (3, 5, 1, 1, 8, 8, 8)
    raw = np.load(home / "chlast.npy")
    x = h.read([2, 0])
    assert np.allclose(x[1, 3, 0, 0], raw[2, 3, ..., 0])  # sorted read order: [0, 2]
    bad = DatasetCard(id="b", name="b", source={"type": "local", "path": str(home / "two_d.npy")},
                      layout={"fields": 3, "height": 16, "width": 16})
    with pytest.raises(ValueError, match="can't be reshaped"):
        DatasetHandle(bad, home / "two_d.npy")


def test_netcdf_and_h5_read_only_needed_rows(tmp_path):
    import h5py
    import netCDF4

    from app.catalog import DatasetCard
    from app.datasets import DatasetHandle

    arr = np.arange(5 * 3 * 2 * 8, dtype=np.float32).reshape(5, 3, 2, 8)
    with netCDF4.Dataset(tmp_path / "d.nc", "w") as ds:
        for n, s in zip("abcd", arr.shape):
            ds.createDimension(n, s)
        ds.createVariable("x", "f4", ("c",))[:] = np.arange(2)
        ds.createVariable("solution", "f4", tuple("abcd"))[:] = arr
    with h5py.File(tmp_path / "d.h5", "w") as f:
        f.create_dataset("grp/u", data=arr)
    for fn in ("d.nc", "d.h5"):
        card = DatasetCard(id="n", name="n", source={"type": "local", "path": str(tmp_path / fn)}, layout={"axes": "NTFW"})
        h = DatasetHandle(card, tmp_path / fn)
        assert h.var["name"] in ("solution", "grp/u")  # largest array picked automatically
        assert np.array_equal(h.read([4])[0, :, :, 0, 0, 0, :], arr[4])


def test_metrics_match_definitions():
    from app import metrics

    t = np.ones((2, 3, 1, 1, 1, 1, 4), np.float32)
    p = t * 1.1
    m = metrics.summary(p, t)
    assert m["mse"] == pytest.approx(0.01, rel=1e-5)
    assert m["nrmse"] == pytest.approx(0.1, rel=1e-4)
    assert len(metrics.per_step_nrmse(p, t)) == 3


def test_run_all_pairs_and_frames(client):
    r = client.post("/api/runs", json={"models": ["persistence", "linear"], "datasets": ["two-d", "chlast", "one-d"],
                                       "settings": {"max_trajectories": 3, "viz_samples": 2}})
    assert r.status_code == 201, r.text
    run = wait(client, r.json()["id"])
    assert run["status"] == "done", run["log"]
    by = {(p["model"], p["dataset"]): p for p in run["pairs"]}
    # persistence and linear share the same rollout start (linear needs 2 frames)
    assert {p["metrics"]["rollout_start"] for p in run["pairs"]} == {1}
    assert by[("persistence", "one-d")]["metrics"]["trajectories"] == 2  # 40% of 5
    # persistence one-step on random data: error equals frame-to-frame change
    assert by[("persistence", "two-d")]["metrics"]["onestep"]["nrmse"] > 0.5
    f = client.get(f"/api/runs/{run['id']}/frame", params={"dataset": "chlast", "models": "persistence,linear",
                                                           "t": 2, "axis": "h", "index": 3}).json()
    assert f["kind"] == "3d" and len(f["truth"]) == 8 and f["models"]["linear"]["pred"]
    f1 = client.get(f"/api/runs/{run['id']}/frame", params={"dataset": "one-d", "models": "linear", "t": 1, "mode": "onestep"}).json()
    assert f1["kind"] == "1d" and f1["models"]["linear"] is None  # needs two earlier frames
    csv = client.get(f"/api/runs/{run['id']}/metrics.csv").text
    assert csv.count("\n") == 7


def test_add_card_validation(client):
    bad = client.post("/api/models", json={"id": "Bad ID", "name": "x", "adapter": "persistence"})
    assert bad.status_code == 422
    ok = client.post("/api/datasets", json={"id": "extra", "name": "Extra", "source": {"type": "url", "url": "https://example.org/x.npy"}})
    assert ok.status_code == 201 and ok.json()["status"]["state"] == "download"
    assert client.post("/api/datasets", json={"id": "extra", "name": "Extra", "source": {"type": "local", "path": "x"}}).status_code == 409
    assert client.delete("/api/datasets/extra").status_code == 200


def test_missing_weights_fail_cleanly(client):
    client.post("/api/models", json={"id": "ghost", "name": "Ghost", "adapter": "pytorch",
                                     "source": {"type": "local", "path": "/nope/model.pt2"}})
    run = wait(client, client.post("/api/runs", json={"models": ["ghost"], "datasets": ["two-d"]}).json()["id"])
    assert run["status"] == "done_with_issues"
    assert "not found" in run["pairs"][0]["error"].lower()
    client.delete("/api/models/ghost")


def test_cancel(client, monkeypatch):
    from app.adapters.baselines import Persistence

    slow = Persistence.predict
    monkeypatch.setattr(Persistence, "predict", lambda self, f, t: (time.sleep(0.3), slow(self, f, t))[1])
    rid = client.post("/api/runs", json={"models": ["persistence"], "datasets": ["two-d", "one-d"]}).json()["id"]
    time.sleep(0.6)
    assert client.post(f"/api/runs/{rid}/cancel").status_code == 200
    run = wait(client, rid)
    assert run["status"] == "cancelled"
    assert client.delete(f"/api/runs/{rid}").status_code == 200


@pytest.mark.skipif(importlib.util.find_spec("morph_pde") is None, reason="morph_pde not installed")
def test_morph_adapter_with_random_weights(client, tmp_path):
    import torch
    from morph_pde import MORPH

    m = MORPH(patch_size=8, dim=256, depth=4, heads=4, heads_xa=32, mlp_dim=1024, max_components=3, conv_filter=8,
              max_ar=1, max_patches=4096, max_fields=3, dropout=0.1, emb_dropout=0.1, lora_r_attn=0, lora_r_mlp=0,
              lora_alpha=None, lora_p=0.0)
    torch.save({"model_state_dict": m.state_dict()}, tmp_path / "ti.pth")
    client.post("/api/models", json={"id": "morph-rand", "name": "MORPH rand", "adapter": "morph",
                                     "source": {"type": "local", "path": str(tmp_path / "ti.pth")}, "config": {"size": "Ti"}})
    run = wait(client, client.post("/api/runs", json={"models": ["morph-rand"], "datasets": ["two-d", "chlast", "one-d"],
                                                      "settings": {"max_trajectories": 2}}).json()["id"])
    st = {p["dataset"]: p for p in run["pairs"]}
    assert st["two-d"]["status"] == "done", st["two-d"]
    assert st["chlast"]["status"] == "done", st["chlast"]
    assert st["one-d"]["status"] == "done", st["one-d"]   # 32 points, divisible by 8
    assert st["two-d"]["info"]["parameters_m"] > 5
    client.delete("/api/models/morph-rand")
