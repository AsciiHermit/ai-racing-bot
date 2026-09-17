"""GPS sensor: reports the vehicle's position as real-world latitude/
longitude (degrees), not raw simulation x/y. The track's local (0, 0)
origin is anchored to an arbitrary real-world lat/lon reference point;
local meters are converted to a lat/lon offset via the equirectangular
(local-tangent-plane) approximation, which is accurate for track-scale
distances (a few km at most -- errors from Earth's curvature/ellipsoid
shape are negligible at this scale).

Ideal/noiseless for v1 (no jitter, drift, or update-rate limiting) --
matches TrackPoseSensor/ProprioceptiveSensor. A noise model is a v1.1+
physics-realism addition, not a v1 requirement.
"""
from __future__ import annotations

import math

import numpy as np

from ars.core.interfaces import Track
from ars.core.types import SensorFrame, VehicleState

EARTH_RADIUS_M = 6_371_000.0

# Arbitrary placeholder origin (equator / prime meridian) for where the
# track's local (x=0, y=0) sits on Earth. Not a real racetrack location --
# swap for a real circuit's lat/lon to anchor the sim there instead.
DEFAULT_ORIGIN_LAT_DEG = 0.0
DEFAULT_ORIGIN_LON_DEG = 0.0


class GpsSensor:
    """Satisfies ars.core.interfaces.Sensor."""

    name = "gps"

    def __init__(
        self,
        origin_lat_deg: float = DEFAULT_ORIGIN_LAT_DEG,
        origin_lon_deg: float = DEFAULT_ORIGIN_LON_DEG,
    ):
        self.origin_lat_deg = origin_lat_deg
        self.origin_lon_deg = origin_lon_deg
        self._origin_lat_rad = math.radians(origin_lat_deg)

    def read(self, state: VehicleState, track: Track) -> SensorFrame:
        lat_deg = self.origin_lat_deg + math.degrees(state.y / EARTH_RADIUS_M)
        lon_deg = self.origin_lon_deg + math.degrees(
            state.x / (EARTH_RADIUS_M * math.cos(self._origin_lat_rad))
        )
        reading = np.array([lat_deg, lon_deg], dtype=np.float64)
        return SensorFrame(readings={self.name: reading})

    def reset(self, state: VehicleState) -> None:
        pass  # stateless -- nothing to reset
