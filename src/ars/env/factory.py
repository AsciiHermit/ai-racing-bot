"""Convenience constructors for the default v1 environment configuration.

Kept separate from racing_env.py so RacingEnv itself stays free of opinions
about which concrete physics/track/sensors to use -- this module is the
one place those defaults are decided.
"""
from __future__ import annotations

from ars.env.racing_env import RacingEnv
from ars.physics import KinematicBicyclePhysics
from ars.sensors import GpsSensor, ImuSensor, LidarSensor, ProprioceptiveSensor, TrackPoseSensor
from ars.track import make_simple_oval


def make_default_env(**kwargs) -> RacingEnv:
    """v1 default: kinematic-bicycle stub physics, simple oval track,
    localization (GPS + IMU) + pose + proprioceptive + a single forward
    lidar ray -- matches the v1 spec (ars.dashboard.config.LidarConfig
    default: num_rays=1, fov=0). Swap any piece by constructing RacingEnv
    directly instead of using this factory."""
    physics = KinematicBicyclePhysics()
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
