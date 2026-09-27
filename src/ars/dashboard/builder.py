"""Turns a SessionConfig (from Steps 1-4) into a runnable (env, agent)
pair for Step 5's Run button. Kept separate from ars.env.factory, which
owns the library's own opinionated default -- this is the dashboard's own
wiring of user choices to concrete objects.
"""
from __future__ import annotations

from ars.agents import DummyExpertAgent
from ars.core.interfaces import VehiclePhysics
from ars.dashboard.config import SessionConfig
from ars.env.racing_env import RacingEnv
from ars.physics import DynamicBicyclePhysics, KinematicBicyclePhysics
from ars.sensors import GpsSensor, ImuSensor, LidarSensor, ProprioceptiveSensor, TrackPoseSensor
from ars.track import make_simple_oval

_DT = 0.02  # must match the RacingEnv dt below, for ImuSensor's velocity differencing

_PHYSICS_BACKENDS: dict[str, type[VehiclePhysics]] = {
    "dynamic_bicycle": DynamicBicyclePhysics,
    "kinematic_stub": KinematicBicyclePhysics,
}


def build_env_and_agent(config: SessionConfig) -> tuple[RacingEnv, DummyExpertAgent]:
    track = make_simple_oval()
    # v1: mass/dims from config not yet consumed by physics
    physics_cls = _PHYSICS_BACKENDS[config.physics.model_name]
    physics = physics_cls()
    sensors = [
        GpsSensor(),
        ImuSensor(dt=_DT),
        TrackPoseSensor(),
        ProprioceptiveSensor(),
        LidarSensor(num_rays=1, fov=0.0, max_range=config.lidar.max_range_m),
    ]
    env = RacingEnv(physics=physics, track=track, sensors=sensors, dt=_DT, off_track_terminates=False)
    agent = DummyExpertAgent(track)
    return env, agent
