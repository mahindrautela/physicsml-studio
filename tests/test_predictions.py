"""The one-step and rollout loops, and the metric definitions."""
import numpy as np
import pytest


class AddOne:
    """Tiny stand-in model: next frame = last frame + 1."""
    ar_order = 1

    def predict(self, frames, traj):
        return frames[:, -1] + 1


def test_one_step_uses_true_frames():
    from app.runner import predict_one_step

    X = np.zeros((2, 4, 1, 1, 1, 1, 3), np.float32)
    X[:, 2] = 10
    out = predict_one_step(AddOne(), X)
    assert np.isnan(out[:, 0]).all()          # nothing before the first input frame
    assert (out[:, 3] == 11).all()            # predicted from the TRUE frame 2


def test_rollout_feeds_predictions_back():
    from app.runner import predict_rollout

    X = np.zeros((2, 6, 1, 1, 1, 1, 3), np.float32)
    out = predict_rollout(AddOne(), X, start=1, steps=3)
    assert np.isnan(out[:, :2]).all() and np.isnan(out[:, 5]).all()
    assert [out[0, t, 0, 0, 0, 0, 0] for t in (2, 3, 4)] == [1, 2, 3]  # errors accumulate


def test_metric_definitions():
    from app import metrics

    truth = np.ones((2, 3, 1, 1, 1, 1, 4), np.float32)
    pred = truth * 1.1
    m = metrics.summary(pred, truth)
    assert m["mse"] == pytest.approx(0.01, rel=1e-5)
    assert m["nrmse"] == pytest.approx(0.1, rel=1e-4)
    assert len(metrics.per_step_nrmse(pred, truth)) == 3
