"""Shared plain-data types passed across module boundaries.

Every cross-module interface in ARS speaks in these dataclasses (or numpy
arrays), never in a module's internal representation. This is what lets
physics, track, sensors, and env be built/tested in isolation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class VehicleState:
    """Full kinematic/dynamic state of one vehicle, in world frame."""

    x: float  # m, world frame
    y: float  # m, world frame
    heading: float  # rad, 0 = +x axis, CCW positive
    vx: float  # m/s, body-frame longitudinal velocity
    vy: float  # m/s, body-frame lateral velocity
    yaw_rate: float  # rad/s
    steer_angle: float = 0.0  # rad, current road-wheel steer angle

    def as_array(self) -> np.ndarray:
        return np.array(
            [self.x, self.y, self.heading, self.vx, self.vy, self.yaw_rate, self.steer_angle],
            dtype=np.float64,
        )


@dataclass
class VehicleAction:
    """Normalized control inputs, always in [-1, 1] (or [0, 1] where noted)."""

    throttle: float  # 0..1
    brake: float  # 0..1
    steer: float  # -1..1 (left..right)


@dataclass
class TrackSample:
    """Local track geometry at/near a given world point (from Track.query)."""

    centerline_x: float
    centerline_y: float
    heading: float  # track tangent direction, rad
    curvature: float  # 1/m, signed (positive = left turn)
    width: float  # m, full track width at this point
    s: float  # arc-length position along centerline, m
    lateral_offset: float  # signed distance of query point from centerline, m


@dataclass
class SensorFrame:
    """One timestep of sensor output for one vehicle. Producers add keys;
    consumers (agents) read only the keys they know about."""

    readings: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class StepInfo:
    """Auxiliary per-step info returned alongside Gym obs/reward/done."""

    progress_s: float = 0.0
    lap: int = 0
    off_track: bool = False
    collided: bool = False
    extra: dict = field(default_factory=dict)
