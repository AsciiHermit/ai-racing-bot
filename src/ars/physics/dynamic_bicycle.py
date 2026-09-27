"""Phase 1 physics: a dynamic bicycle model with a Pacejka-style tire curve.

Replaces the kinematic stub's flat lateral-acceleration ceiling with a real
slip-based tire model -- grip now emerges from tire saturation, not an
imposed cap. Still the *bicycle* stage of PHYSICS.md's incremental path
(one virtual front wheel, one virtual rear wheel, static load split only --
no load transfer, no 4-wheel independence, no aero). Those are later steps
on that same path, not this module's job.

Uses the existing ars.core.types.VehicleState directly, not a new state
type -- the wayfinder map's "ChassisState protocol" ticket
(github.com/AsciiHermit/ai-racing-bot/issues/3) decided that formalizing a
separate ChassisState Protocol is deferred until the 4-wheel migration;
this bicycle stage fits VehicleState as-is.

Slip-angle sign convention: alpha = steer_angle - atan2(vy + a*yaw_rate, vx)
for the front axle (no steer term for the rear). This was verified
empirically during the Phase 0 spike (see PHASE0_SPIKE_NOTE.md) -- getting
this backwards makes the tire force reinforce a slide instead of opposing
it, a runaway bug, not a realistic spin. With this convention, a small-angle
expansion of the Pacejka curve below reduces to Fy = +mu*Fz*D*C*B*alpha,
matching the linear restoring-force behavior the spike confirmed is stable.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from ars.core.types import VehicleAction, VehicleState

G = 9.81  # m/s^2


@dataclass
class DynamicBicycleParams:
    mass: float = 798.0  # kg, real F1 reference (matches KinematicBicycleParams)
    izz: float = 1100.0  # kg*m^2, approximate single-seater-scale yaw inertia
    wheelbase: float = 2.6  # m
    dist_cg_to_front: float = 1.2  # m ("a"), rear-biased CG typical of an RWD race car
    dist_cg_to_rear: float = 1.4  # m ("b"); a + b == wheelbase

    # Longitudinal: forces in Newtons, not raw m/s^2 constants, so mass
    # actually affects acceleration (accel = force / mass). Calibrated so
    # a 798 kg reference vehicle gets the same ~1.2g accel / ~5g braking
    # as the kinematic stub's tuning.
    max_engine_force: float = 9_576.0  # N (= 798 kg * 12 m/s^2)
    max_brake_force: float = 39_146.0  # N (= 798 kg * 5g)
    drag_coeff: float = 0.02  # simple speed-proportional drag

    max_steer_angle: float = math.radians(28)  # rad, at steer=+-1
    steer_rate: float = math.radians(180)  # rad/s, max steer angle change
    max_speed: float = 95.0  # m/s (~342 km/h), same cap as the kinematic stub

    # Pacejka "Magic Formula" lateral tire curve, same B/C/D/E for both
    # axles -- front/rear asymmetry comes from the static load split
    # (dist_cg_to_front/rear), not from different tire-curve shapes.
    # Representative race-tire-scale values (Pacejka '96 ballpark), not
    # fit to a specific real tire.
    pacejka_b: float = 10.0  # stiffness factor, 1/rad
    pacejka_c: float = 1.4  # shape factor, dimensionless
    pacejka_d: float = 1.0  # peak factor, dimensionless (peak force = mu * Fz * D)
    pacejka_e: float = -0.2  # curvature factor, dimensionless
    mu: float = 1.6  # tire-road friction coefficient, no aero/downforce yet


class DynamicBicyclePhysics:
    """Satisfies ars.core.interfaces.VehiclePhysics."""

    def __init__(self, params: DynamicBicycleParams | None = None):
        self.params = params or DynamicBicycleParams()
        p = self.params
        self._fz_front = p.mass * G * p.dist_cg_to_rear / p.wheelbase
        self._fz_rear = p.mass * G * p.dist_cg_to_front / p.wheelbase

    def reset(self, x: float, y: float, heading: float) -> VehicleState:
        return VehicleState(x=x, y=y, heading=heading, vx=0.0, vy=0.0, yaw_rate=0.0, steer_angle=0.0)

    def step(self, state: VehicleState, action: VehicleAction, dt: float) -> VehicleState:
        p = self.params
        throttle = _clamp(action.throttle, 0.0, 1.0)
        brake = _clamp(action.brake, 0.0, 1.0)
        steer_cmd = _clamp(action.steer, -1.0, 1.0) * p.max_steer_angle
        max_delta = p.steer_rate * dt
        steer_angle = state.steer_angle + _clamp(steer_cmd - state.steer_angle, -max_delta, max_delta)

        vx = state.vx
        vy = state.vy
        r = state.yaw_rate
        # Guard only the atan2 slip-angle calc against divide-by-zero at a
        # dead stop -- the equations of motion below use the real vx, never
        # this floor (flooring vx there would inject a fake forcing term
        # once speed settles near zero; see the Phase 0 spike note).
        vx_for_slip = max(vx, 1.0)

        alpha_f = steer_angle - math.atan2(vy + p.dist_cg_to_front * r, vx_for_slip)
        alpha_r = -math.atan2(vy - p.dist_cg_to_rear * r, vx_for_slip)

        fy_f = _pacejka(alpha_f, p.mu * self._fz_front, p)
        fy_r = _pacejka(alpha_r, p.mu * self._fz_rear, p)

        fx = throttle * p.max_engine_force - brake * p.max_brake_force - p.drag_coeff * p.mass * vx

        vx_dot = fx / p.mass + vy * r
        vy_dot = (fy_f * math.cos(steer_angle) + fy_r) / p.mass - vx * r
        yaw_rate_dot = (p.dist_cg_to_front * fy_f * math.cos(steer_angle) - p.dist_cg_to_rear * fy_r) / p.izz

        new_vx = _clamp(vx + vx_dot * dt, 0.0, p.max_speed)
        new_vy = vy + vy_dot * dt
        new_yaw_rate = r + yaw_rate_dot * dt
        heading = state.heading + new_yaw_rate * dt

        x = state.x + (new_vx * math.cos(heading) - new_vy * math.sin(heading)) * dt
        y = state.y + (new_vx * math.sin(heading) + new_vy * math.cos(heading)) * dt

        return VehicleState(
            x=x, y=y, heading=heading, vx=new_vx, vy=new_vy, yaw_rate=new_yaw_rate, steer_angle=steer_angle
        )


def _pacejka(alpha: float, peak_force: float, p: DynamicBicycleParams) -> float:
    """Simplified Pacejka Magic Formula: Fy = peak_force * D * sin(C *
    arctan(B*alpha - E*(B*alpha - arctan(B*alpha)))). peak_force is mu*Fz
    for the axle in question; D further scales the curve's peak."""
    b_alpha = p.pacejka_b * alpha
    return peak_force * p.pacejka_d * math.sin(
        p.pacejka_c * math.atan(b_alpha - p.pacejka_e * (b_alpha - math.atan(b_alpha)))
    )


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))
