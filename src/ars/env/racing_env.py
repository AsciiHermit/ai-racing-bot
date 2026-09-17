"""Gymnasium-compatible single-agent racing environment.

This is the integration point (Module H): it owns nothing about physics,
track geometry, or sensors -- it only depends on the Protocols in
ars.core.interfaces, and composes whatever concrete implementations are
passed into it. Swapping the physics model or track later means changing
the objects passed to __init__, not this file.
"""
from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ars.core.interfaces import Sensor, Track, VehiclePhysics
from ars.core.types import StepInfo, VehicleAction


class RacingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        physics: VehiclePhysics,
        track: Track,
        sensors: list[Sensor],
        dt: float = 0.02,
        max_episode_steps: int = 2000,
        off_track_terminates: bool = True,
    ):
        super().__init__()
        self.physics = physics
        self.track = track
        self.sensors = sensors
        self.dt = dt
        self.max_episode_steps = max_episode_steps
        self.off_track_terminates = off_track_terminates

        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )  # [throttle, brake, steer]

        obs_dim = self._probe_obs_dim()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self._state = None
        self._steps = 0
        self._prev_s = 0.0
        self._lap = 0

    def _probe_obs_dim(self) -> int:
        start = self.track.sample_at_s(0.0)
        dummy_state = self.physics.reset(start.centerline_x, start.centerline_y, start.heading)
        obs = self._build_obs(dummy_state)
        return obs.shape[0]

    def _build_obs(self, state) -> np.ndarray:
        parts = [sensor.read(state, self.track).readings[sensor.name] for sensor in self.sensors]
        return np.concatenate(parts).astype(np.float32)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        start = self.track.sample_at_s(0.0)
        self._state = self.physics.reset(start.centerline_x, start.centerline_y, start.heading)
        self._steps = 0
        self._prev_s = 0.0
        self._lap = 0
        for sensor in self.sensors:
            sensor.reset(self._state)
        obs = self._build_obs(self._state)
        info = {"progress_s": 0.0, "lap": 0}
        return obs, info

    def step(self, action: np.ndarray):
        assert self._state is not None, "call reset() before step()"
        throttle, brake, steer = (float(a) for a in action)
        vehicle_action = VehicleAction(throttle=throttle, brake=brake, steer=steer)
        self._state = self.physics.step(self._state, vehicle_action, self.dt)
        self._steps += 1

        sample = self.track.query(self._state.x, self._state.y)
        off_track = abs(sample.lateral_offset) > sample.width / 2.0

        ds = _progress_delta(self._prev_s, sample.s, self.track.length)
        if _crossed_finish_line_forward(self._prev_s, sample.s, self.track.length):
            self._lap += 1
        self._prev_s = sample.s

        reward = ds - 0.01 * abs(sample.lateral_offset)
        if off_track:
            reward -= 1.0

        terminated = off_track and self.off_track_terminates
        truncated = self._steps >= self.max_episode_steps

        obs = self._build_obs(self._state)
        step_info = StepInfo(progress_s=sample.s, lap=self._lap, off_track=off_track)
        info = {"progress_s": step_info.progress_s, "lap": step_info.lap, "off_track": off_track}
        return obs, reward, terminated, truncated, info

    @property
    def vehicle_state(self):
        """Current VehicleState (or physics-module-specific state type),
        read-only access for renderers/loggers. None before reset()."""
        return self._state

    def render(self):
        return None


def _progress_delta(prev_s: float, new_s: float, track_length: float) -> float:
    """Signed arc-length progress, handling wraparound at the start/finish line."""
    delta = new_s - prev_s
    if delta < -track_length / 2:
        delta += track_length
    elif delta > track_length / 2:
        delta -= track_length
    return delta


def _crossed_finish_line_forward(prev_s: float, new_s: float, track_length: float) -> bool:
    """True if progress moved forward across s=track_length -> s=0 this
    step (a completed lap), i.e. raw s dropped by more than half the track
    -- as opposed to _progress_delta's return value, which is already
    wraparound-corrected into [-length/2, length/2] and so can never
    exceed that range itself."""
    return (new_s - prev_s) < -track_length / 2
