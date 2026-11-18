"""Whole runs through the API: results, viewer frames, CSV, errors, stopping."""
import importlib.util
import time

import pytest


def test_run_every_model_on_every_dataset(client, wait_for_run):
    started = client.post("/api/runs", json={"models": ["persistence", "linear"], "datasets": ["two-d", "chlast", "one-d"],
                                             "settings": {"max_trajectories": 3, "viz_samples": 2}})
    assert started.status_code == 201, started.text
    run = wait_for_run(started.json()["id"])
    assert run["status"] == "done", run["log"]
    pairs = {(p["model"], p["dataset"]): p for p in run["pairs"]}
    # the linear baseline needs 2 frames, so every model's rollout starts after step 1
    assert {p["metrics"]["rollout_start"] for p in run["pairs"]} == {1}
    assert pairs[("persistence", "one-d")]["metrics"]["trajectories"] == 2  # 40% of 5

    frame = client.get(f"/api/runs/{run['id']}/frame", params={
        "dataset": "chlast", "models": "persistence,linear", "t": 2, "axis": "h", "index": 3}).json()
    assert frame["kind"] == "3d" and len(frame["truth"]) == 8 and frame["models"]["linear"]["pred"]
    one_d = client.get(f"/api/runs/{run['id']}/frame", params={
        "dataset": "one-d", "models": "linear", "t": 1, "mode": "onestep"}).json()
    assert one_d["kind"] == "1d" and one_d["models"]["linear"] is None  # needs two earlier frames

    csv = client.get(f"/api/runs/{run['id']}/metrics.csv").text
    assert csv.count("\n") == 1 + 6  # header + 2 models × 3 datasets


def test_missing_weights_fail_with_a_clear_message(client, wait_for_run):
    client.post("/api/models", json={"id": "ghost", "name": "Ghost", "adapter": "pytorch",
                                     "source": {"type": "local", "path": "/nope/model.pt2"}})
    run = wait_for_run(client.post("/api/runs", json={"models": ["ghost"], "datasets": ["two-d"]}).json()["id"])
    assert run["status"] == "done_with_issues"
    assert "not found" in run["pairs"][0]["error"].lower()
    client.delete("/api/models/ghost")


def test_stop_then_delete_a_run(client, wait_for_run, monkeypatch):
    from app.adapters.baselines import Persistence

    real_predict = Persistence.predict
    monkeypatch.setattr(Persistence, "predict", lambda self, f, t: (time.sleep(0.3), real_predict(self, f, t))[1])
    run_id = client.post("/api/runs", json={"models": ["persistence"], "datasets": ["two-d", "one-d"]}).json()["id"]
    time.sleep(0.6)
    assert client.post(f"/api/runs/{run_id}/cancel").status_code == 200
    assert wait_for_run(run_id)["status"] == "cancelled"
    assert client.delete(f"/api/runs/{run_id}").status_code == 200
    assert client.get(f"/api/runs/{run_id}").status_code == 404


@pytest.mark.skipif(importlib.util.find_spec("morph_pde") is None, reason="MORPH's code isn't installed")
def test_morph_runs_on_1d_2d_and_3d(client, wait_for_run, tmp_path):
    """Random weights: checks the plumbing only, no download needed."""
    import torch
    from morph_pde import MORPH

    model = MORPH(patch_size=8, dim=256, depth=4, heads=4, heads_xa=32, mlp_dim=1024, max_components=3,
                  conv_filter=8, max_ar=1, max_patches=4096, max_fields=3, dropout=0.1, emb_dropout=0.1,
                  lora_r_attn=0, lora_r_mlp=0, lora_alpha=None, lora_p=0.0)
    torch.save({"model_state_dict": model.state_dict()}, tmp_path / "ti.pth")
    client.post("/api/models", json={"id": "morph-random", "name": "MORPH random", "adapter": "morph",
                                     "source": {"type": "local", "path": str(tmp_path / "ti.pth")},
                                     "config": {"size": "Ti"}})
    run = wait_for_run(client.post("/api/runs", json={"models": ["morph-random"], "datasets": ["two-d", "chlast", "one-d"],
                                                      "settings": {"max_trajectories": 2}}).json()["id"])
    for pair in run["pairs"]:
        assert pair["status"] == "done", pair
    assert run["pairs"][0]["info"]["parameters_m"] > 5
    client.delete("/api/models/morph-random")
