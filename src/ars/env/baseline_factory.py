"""Phase 3 baseline factory (IMPLEMENTATION_PLAN.md Phase 3's default
observation bundle): LiDAR + proprioceptive + track-relative pose only --
narrower than ars.env.factory.make_default_env's fuller bundle, which also
includes GPS and IMU. Full modality-combination ablation is deferred to
Phase 5/6, per the plan's own task list.

Step A (single fixed oval) vs. Step B (the full Phase 2 train set) is just
whether you pass `track` or `track_provider` -- see ars.env.multi_track and
ars.track.manifest.tracks_from_specs for building the Step B track list.
"""
from __future__ import annotations

from typing import Callable

from ars.core.interfaces import Track, VehiclePhysics
from ars.env.racing_env import RacingEnv
from ars.physics import DynamicBicyclePhysics
from ars.sensors import LidarSensor, ProprioceptiveSensor, TrackPoseSensor
from ars.track import make_simple_oval


def make_baseline_env(
    track: Track | None = None,
    track_provider: Callable[[], Track] | None = None,
    physics: VehiclePhysics | None = None,
    **kwargs,
) -> RacingEnv:
    physics = physics or DynamicBicyclePhysics()
    if track is None and track_provider is None:
        track = make_simple_oval()  # Step A default: the single fixed oval
    sensors = [
        LidarSensor(num_rays=1, fov=0.0),
        ProprioceptiveSensor(),
        TrackPoseSensor(),
    ]
    return RacingEnv(
        physics=physics, track=track, track_provider=track_provider, sensors=sensors, **kwargs
    )
