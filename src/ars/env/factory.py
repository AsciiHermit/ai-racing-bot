"""Convenience constructors for the default v1 environment configuration.

Kept separate from racing_env.py so RacingEnv itself stays free of opinions
about which concrete physics/track/sensors to use -- this module is the
one place those defaults are decided.
"""
from __future__ import annotations

from ars.core.interfaces import VehiclePhysics
from ars.env.racing_env import RacingEnv
from ars.physics import DynamicBicyclePhysics
from ars.sensors import GpsSensor, ImuSensor, LidarSensor, ProprioceptiveSensor, TrackPoseSensor
from ars.track import make_simple_oval


def make_default_env(physics: VehiclePhysics | None = None, **kwargs) -> RacingEnv:
    """v1 default: dynamic-bicycle physics (Phase 1), simple oval track,
    localization (GPS + IMU) + pose + proprioceptive + a single forward
    lidar ray -- matches the v1 spec (ars.dashboard.config.LidarConfig
    default: num_rays=1, fov=0). Pass `physics=KinematicBicyclePhysics()`
    for the older, faster/degenerate stub (e.g. later ablations); swap any
    other piece by constructing RacingEnv directly instead of using this
    factory."""
    physics = physics or DynamicBicyclePhysics()
    track = make_simple_oval()
    dt = kwargs.get("dt", 0.02)  # must match RacingEnv's dt so ImuSensor differences velocity correctly
    sensors = [
        GpsSensor(),
        ImuSensor(dt=dt),
        TrackPoseSensor(),
        ProprioceptiveSensor(),
        LidarSensor(num_rays=1, fov=0.0),
    ]
    return RacingEnv(physics=physics, track=track, sensors=sensors, **kwargs)
