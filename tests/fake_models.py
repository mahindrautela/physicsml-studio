"""Tiny model types used only by the tests: no weights, no downloads, instant.

    test_copy   next frame = last frame                    (needs 1 input frame)
    test_trend  next frame = 2 * last - the one before     (needs 2 input frames)

A card can set config `delay` (seconds per prediction) to make a run slow
enough to stop it in a test.
"""
import time

from app.adapters import register
from app.adapters.base import Adapter


class _Fake(Adapter):
    needs_weights = False

    def predict(self, frames, traj):
        time.sleep(float(self.cfg.get("delay", 0)))
        return self.next_frame(frames)


@register("test_copy")
class Copy(_Fake):
    ar_order = 1

    @staticmethod
    def next_frame(frames):
        return frames[:, -1].copy()


@register("test_trend")
class Trend(_Fake):
    ar_order = 2

    @staticmethod
    def next_frame(frames):
        return 2.0 * frames[:, -1] - frames[:, -2]
