"""Dummy expert: a hand-coded (non-RL) line-following controller. Not
learned -- reads the track directly (not the flattened observation vector,
to avoid depending on sensor ordering) and steers/throttles to hold the
centerline. Exists as v1's placeholder for Step 4 "Agent" -- a believable
baseline to run in Step 5 before any real RL agent exists.
"""
from __future__ import annotations

import numpy as np

from ars.core.interfaces import Track


class DummyExpertAgent:
    def __init__(
        self,
        track: Track,
        steer_gain_offset: float = 0.06,
        steer_gain_heading: float = 1.1,
        base_throttle: float = 0.55,
        min_throttle: float = 0.25,
        curvature_slowdown: float = 6.0,
    ):
        self.track = track
        self.steer_gain_offset = steer_gain_offset
        self.steer_gain_heading = steer_gain_heading
        self.base_throttle = base_throttle
        self.min_throttle = min_throttle
        self.curvature_slowdown = curvature_slowdown

    def act(self, vehicle_state) -> np.ndarray:
        sample = self.track.query(vehicle_state.x, vehicle_state.y)
        heading_error = _wrap(vehicle_state.heading - sample.heading)

        steer = -self.steer_gain_offset * sample.lateral_offset - self.steer_gain_heading * heading_error
        steer = float(np.clip(steer, -1.0, 1.0))

        throttle = max(self.min_throttle, self.base_throttle - self.curvature_slowdown * abs(sample.curvature))
        throttle = float(np.clip(throttle, 0.0, 1.0))

        return np.array([throttle, 0.0, steer], dtype=np.float32)


def _wrap(angle: float) -> float:
    import math

    return (angle + math.pi) % (2 * math.pi) - math.pi
