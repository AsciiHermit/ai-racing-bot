"""Placeholder physics: a kinematic bicycle model.

This exists so the env/sensors/agent stack has something real to run
against on day one. It is deliberately simple -- no tire slip, no load
transfer, no aero -- and is meant to be REPLACED by the physics teammate's
implementation (e.g. dynamic bicycle + Pacejka tire model). As long as the
replacement satisfies ars.core.interfaces.VehiclePhysics, nothing else in
the repo needs to change.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from ars.core.types import VehicleAction, VehicleState


@dataclass
class KinematicBicycleParams:
    wheelbase: float = 2.6  # m
    max_speed: float = 60.0  # m/s
    max_accel: float = 6.0  # m/s^2, from throttle=1
    max_decel: float = 9.0  # m/s^2, from brake=1
    drag_coeff: float = 0.02  # simple speed-proportional drag
    max_steer_angle: float = math.radians(28)  # rad, at steer=+-1
    steer_rate: float = math.radians(180)  # rad/s, max steer angle change


class KinematicBicyclePhysics:
    """Satisfies ars.core.interfaces.VehiclePhysics."""

    def __init__(self, params: KinematicBicycleParams | None = None):
        self.params = params or KinematicBicycleParams()

    def reset(self, x: float, y: float, heading: float) -> VehicleState:
        return VehicleState(x=x, y=y, heading=heading, vx=0.0, vy=0.0, yaw_rate=0.0, steer_angle=0.0)

    def step(self, state: VehicleState, action: VehicleAction, dt: float) -> VehicleState:
        p = self.params
        throttle = _clamp(action.throttle, 0.0, 1.0)
        brake = _clamp(action.brake, 0.0, 1.0)
        steer_cmd = _clamp(action.steer, -1.0, 1.0) * p.max_steer_angle

        max_delta = p.steer_rate * dt
        steer_angle = state.steer_angle + _clamp(steer_cmd - state.steer_angle, -max_delta, max_delta)

        speed = state.vx
        accel = throttle * p.max_accel - brake * p.max_decel - p.drag_coeff * speed
        new_speed = _clamp(speed + accel * dt, 0.0, p.max_speed)

        # Kinematic bicycle: no lateral slip, vy = 0 always.
        yaw_rate = (new_speed / p.wheelbase) * math.tan(steer_angle) if p.wheelbase else 0.0
        heading = state.heading + yaw_rate * dt
        x = state.x + new_speed * math.cos(heading) * dt
        y = state.y + new_speed * math.sin(heading) * dt

        return VehicleState(
            x=x,
            y=y,
            heading=heading,
            vx=new_speed,
            vy=0.0,
            yaw_rate=yaw_rate,
            steer_angle=steer_angle,
        )


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))
