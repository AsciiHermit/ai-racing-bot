"""Phase 3 Step B: training across the full Phase 2 train set, not one
fixed track. RacingEnv takes a `track_provider` callable instead of (or
alongside) a fixed `track`; this module's RandomTrackProvider is the
simplest one -- picks uniformly at random from a fixed list on every call.

A plain callable class, not a closure/lambda: SB3's SubprocVecEnv pickles
the env (and everything it closes over) to hand to worker processes for
vectorized rollout, and a class instance survives that reliably where a
closure risks not.
"""
from __future__ import annotations

import random

from ars.core.interfaces import Track


class RandomTrackProvider:
    def __init__(self, tracks: list[Track], seed: int | None = None):
        if not tracks:
            raise ValueError("tracks must be non-empty")
        self.tracks = tracks
        self._rng = random.Random(seed)

    def __call__(self) -> Track:
        return self._rng.choice(self.tracks)
