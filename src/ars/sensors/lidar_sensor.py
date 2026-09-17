"""Lidar-style raycast sensor: distance to track boundary along N fixed
rays relative to heading. Cheap math-only implementation (ray-march along
each direction, step until off-track) -- no rendering involved.
"""
from __future__ import annotations

import math

import numpy as np

from ars.core.interfaces import Track
from ars.core.types import SensorFrame, VehicleState


class LidarSensor:
    """Satisfies ars.core.interfaces.Sensor."""

    def __init__(
        self,
        num_rays: int = 9,
        fov: float = math.pi,  # total field of view, centered on heading
        max_range: float = 40.0,
        step: float = 0.5,
    ):
        self.name = "lidar"
        self.num_rays = num_rays
        self.fov = fov
        self.max_range = max_range
        self.step = step
        if num_rays == 1:
            self._angles = np.array([0.0])
        else:
            self._angles = np.linspace(-fov / 2, fov / 2, num_rays)

    def read(self, state: VehicleState, track: Track) -> SensorFrame:
        distances = np.empty(self.num_rays, dtype=np.float64)
        for i, rel_angle in enumerate(self._angles):
            ray_heading = state.heading + rel_angle
            distances[i] = self._cast(state.x, state.y, ray_heading, track)
        return SensorFrame(readings={self.name: distances})

    def _cast(self, x: float, y: float, heading: float, track: Track) -> float:
        dx = math.cos(heading)
        dy = math.sin(heading)
        r = 0.0
        while r < self.max_range:
            r += self.step
            if not track.is_on_track(x + dx * r, y + dy * r):
                return r
        return self.max_range
