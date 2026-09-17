"""Placeholder physics: a kinematic bicycle model.

This exists so the env/sensors/agent stack has something real to run
against on day one. It is deliberately simple -- no tire slip, no load
transfer, no aero -- and is meant to be REPLACED by the physics teammate's
implementation (e.g. dynamic bicycle + Pacejka tire model). As long as the
replacement satisfies ars.core.interfaces.VehiclePhysics, nothing else in
the repo needs to change.

Default params are tuned to real F1-scale reference values (2024-era
regs/performance): ~95 m/s top speed (~342 km/h), ~5g max braking,
~4.5g max lateral grip. A kinematic bicycle has no tire model, so the
lateral-grip cap below is a stand-in for what tire slip will eventually
enforce -- without it, the car could corner at any speed with zero slip,
which is physically impossible (see max_lateral_accel).
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from ars.core.types import VehicleAction, VehicleState

G = 9.81  # m/s^2


@dataclass
class KinematicBicycleParams:
    wheelbase: float = 2.6  # m
    max_speed: float = 95.0  # m/s (~342 km/h, real F1 top speed)
    max_accel: float = 12.0  # m/s^2 (~1.2g, from throttle=1)
    max_decel: float = 5.0 * G  # m/s^2 (~5g, real F1 max braking, from brake=1)
    drag_coeff: float = 0.02  # simple speed-proportional drag
    max_steer_angle: float = math.radians(28)  # rad, at steer=+-1
    steer_rate: float = math.radians(180)  # rad/s, max steer angle change
    max_lateral_accel: float = 4.5 * G  # m/s^2 (~4.5g, real F1 max cornering grip).
    # Stand-in for a tire model: caps speed so a_lat = v^2 * curvature
    # never exceeds this, i.e. the car can't corner faster than real grip
    # allows. curvature here comes from the vehicle's own steer angle
    # (v * yaw_rate = v^2 * tan(steer_angle) / wheelbase), not the track --
    # this stub has no track knowledge, by design (see VehiclePhysics).


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

        # Grip cap: curvature implied by the current steer angle
        # (kinematic bicycle, no track knowledge -- see class docstring).
        # a_lat = v^2 * curvature must not exceed max_lateral_accel.
        curvature = abs(math.tan(steer_angle)) / p.wheelbase if p.wheelbase else 0.0
        if curvature > 1e-9:
            max_corner_speed = math.sqrt(p.max_lateral_accel / curvature)
            new_speed = min(new_speed, max_corner_speed)

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
