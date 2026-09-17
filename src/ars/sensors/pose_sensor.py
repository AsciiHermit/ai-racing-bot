"""Track-relative pose sensor: centerline distance, heading error, progress.

This is usually the single most important observation for a line-following
racing agent, so it's the first sensor implemented in v1.
"""
from __future__ import annotations

import numpy as np

from ars.core.interfaces import Track
from ars.core.types import SensorFrame, VehicleState


class TrackPoseSensor:
    """Satisfies ars.core.interfaces.Sensor."""

    name = "track_pose"

    def read(self, state: VehicleState, track: Track) -> SensorFrame:
        sample = track.query(state.x, state.y)
        heading_error = _wrap(state.heading - sample.heading)
        reading = np.array(
            [sample.lateral_offset, heading_error, sample.curvature, sample.s],
            dtype=np.float64,
        )
        return SensorFrame(readings={self.name: reading})

    def reset(self, state: VehicleState) -> None:
        pass  # stateless -- nothing to reset


def _wrap(angle: float) -> float:
    import math

    return (angle + math.pi) % (2 * math.pi) - math.pi
