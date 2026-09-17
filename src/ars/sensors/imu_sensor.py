"""IMU sensor: accelerometer (body-frame linear acceleration) + gyroscope
(yaw rate). This is what a real IMU measures -- NOT position or velocity.
Velocity/odometry sensing (e.g. from wheel speed) is a separate sensor,
left for the physics teammate to add later alongside the real vehicle
model.

Acceleration isn't in VehicleState, so it's derived here by differencing
consecutive body-frame velocities (vx, vy) between calls -- mirrors how a
real accelerometer's reading arises from the vehicle's actual motion, and
needs no change to VehicleState or VehiclePhysics. Ideal/noiseless for v1
(no bias, drift, or vibration noise) -- a noise model is v1.1+.
"""
from __future__ import annotations

import numpy as np

from ars.core.interfaces import Track
from ars.core.types import SensorFrame, VehicleState


class ImuSensor:
    """Satisfies ars.core.interfaces.Sensor."""

    name = "imu"

    def __init__(self, dt: float = 0.02):
        self.dt = dt
        self._prev_vx: float | None = None
        self._prev_vy: float | None = None

    def read(self, state: VehicleState, track: Track) -> SensorFrame:
        if self._prev_vx is None:
            # First reading (or right after reset()) -- no prior velocity
            # to difference against, so report zero acceleration rather
            # than a spurious jump from an unrelated previous episode.
            ax, ay = 0.0, 0.0
        else:
            ax = (state.vx - self._prev_vx) / self.dt
            ay = (state.vy - self._prev_vy) / self.dt
        self._prev_vx = state.vx
        self._prev_vy = state.vy

        reading = np.array([ax, ay, state.yaw_rate], dtype=np.float64)
        return SensorFrame(readings={self.name: reading})

    def reset(self, state: VehicleState) -> None:
        self._prev_vx = None
        self._prev_vy = None
