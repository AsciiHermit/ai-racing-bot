"""Phase 5's regret-proxy curriculum (wayfinder ticket #10,
github.com/AsciiHermit/ai-racing-bot/issues/10): regret is a track's gap
to the agent's own per-track rolling-best return, not the Phase 4 MPC
gap -- oversampling is priority-weighted sampling proportional to that
regret, refreshed periodically (every `refresh_every` episodes, not every
step), not a bucket scheme.

RegretCurriculumTrackProvider samples from a fixed pool of tracks (the
domain-randomization pool, for Phase 5 Step A) rather than re-randomizing
continuous parameters fresh every single episode -- that would make
"per-track rolling-best" meaningless, since there would be no "same track"
to revisit. The pool itself is what represents the randomized distribution;
this provider decides which pool member gets used each episode.

Feeding back an episode's return: call record_episode() with the return
right after an episode ends, before the next reset() -- see
scripts/train_phase5_step_a.py's callback, which reads it off SB3's
Monitor-wrapper info dict rather than requiring any change to RacingEnv
itself.
"""
from __future__ import annotations

import random

from ars.core.interfaces import Track


class RegretCurriculumTrackProvider:
    def __init__(self, tracks: list[Track], refresh_every: int = 20, seed: int | None = None):
        if not tracks:
            raise ValueError("tracks must be non-empty")
        self.tracks = tracks
        self.refresh_every = refresh_every
        self._rng = random.Random(seed)

        n = len(tracks)
        self._rolling_best = [float("-inf")] * n
        self._last_return = [float("-inf")] * n
        self._visited = [False] * n
        self.weights = [1.0] * n  # uniform until the first refresh
        self._episodes_since_refresh = 0
        self._last_index: int | None = None

    @property
    def rolling_best(self) -> list[float]:
        return self._rolling_best

    def __call__(self) -> Track:
        self._last_index = self._rng.choices(range(len(self.tracks)), weights=self.weights)[0]
        return self.tracks[self._last_index]

    def index_of(self, track: Track) -> int:
        for i, candidate in enumerate(self.tracks):
            if candidate is track:
                return i
        raise ValueError("track is not in this provider's pool")

    def record_episode(self, episode_return: float, track_index: int | None = None) -> None:
        """track_index defaults to the most recently sampled track, which is
        only right with a single env. With several vectorized envs sharing
        this provider, pass the index of the track the finished episode
        actually ran on (see index_of)."""
        idx = track_index if track_index is not None else self._last_index
        if idx is None:
            raise RuntimeError("record_episode() called before any track was sampled")
        self._rolling_best[idx] = max(self._rolling_best[idx], episode_return)
        self._last_return[idx] = episode_return
        self._visited[idx] = True

        self._episodes_since_refresh += 1
        if self._episodes_since_refresh >= self.refresh_every:
            self._refresh_weights()
            self._episodes_since_refresh = 0

    def _refresh_weights(self) -> None:
        regrets = [
            max(0.0, self._rolling_best[i] - self._last_return[i]) if self._visited[i] else None
            for i in range(len(self.tracks))
        ]
        visited_regrets = [r for r in regrets if r is not None]
        total = sum(visited_regrets)
        # Floor keeps every track (visited or not) sampleable -- an
        # unvisited track needs a nonzero chance to ever get visited at
        # all, and a visited track that's currently at its own rolling
        # best (regret 0) shouldn't drop to zero sampling probability
        # either, since its best could still be beaten later.
        floor = 0.1 * (total / len(visited_regrets) if visited_regrets else 1.0) or 0.1
        self.weights = [(r if r is not None else 0.0) + floor for r in regrets]
