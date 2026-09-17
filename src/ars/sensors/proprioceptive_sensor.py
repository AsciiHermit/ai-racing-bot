"""Proprioceptive sensor: what the vehicle "feels" about itself, no track
knowledge required (speed, yaw rate, steer angle)."""
from __future__ import annotations

import numpy as np

from ars.core.interfaces import Track
from ars.core.types import SensorFrame, VehicleState


class ProprioceptiveSensor:
    """Satisfies ars.core.interfaces.Sensor."""

    name = "proprioceptive"

    def read(self, state: VehicleState, track: Track) -> SensorFrame:
        reading = np.array(
            [state.vx, state.vy, state.yaw_rate, state.steer_angle],
            dtype=np.float64,
        )
        return SensorFrame(readings={self.name: reading})

    def reset(self, state: VehicleState) -> None:
        pass  # stateless -- nothing to reset
