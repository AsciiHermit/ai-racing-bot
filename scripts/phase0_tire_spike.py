"""Phase 0 technical spike (IMPLEMENTATION_PLAN.md Phase 0): de-risk the
plan's core assumption that a slip-based tire model is hard enough to make
cross-track generalization a real learning problem, not one a simple
controller already solves.

Throwaway by design -- not production quality, not meant to be reused by
Phase 1's real dynamic_bicycle.py. It exists only to answer one question:
does the existing DummyExpertAgent (a hand-coded line-follower, already in
the repo) drive near the tire model's grip limit, or does it fall well
short of it?

Run: python scripts/phase0_tire_spike.py
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from ars.core.types import VehicleAction, VehicleState
from ars.track.loop_track import make_simple_oval
from ars.agents.dummy_expert import DummyExpertAgent

G = 9.81  # m/s^2


@dataclass
class SlipBicycleParams:
    """Throwaway, unvalidated reference numbers -- good enough to answer
    the go/no-go question, not a real vehicle calibration."""

    mass: float = 798.0  # kg, matches KinematicBicycleParams' F1 reference
    izz: float = 1200.0  # kg*m^2, rough estimate for a single-seater
    dist_cg_to_front: float = 1.2  # m ("a")
    dist_cg_to_rear: float = 1.4  # m ("b"), wheelbase 2.6 m, rear-biased CG
    cornering_stiffness_front: float = 80_000.0  # N/rad, linear tire region
    cornering_stiffness_rear: float = 90_000.0  # N/rad
    mu: float = 1.6  # tire-road friction coefficient, NO aero/downforce yet
    max_speed: float = 95.0  # m/s, same cap as the kinematic stub
    max_accel: float = 12.0  # m/s^2, throttle=1
    max_decel: float = 5.0 * G  # m/s^2, brake=1
    drag_coeff: float = 0.02
    max_steer_angle: float = math.radians(28)
    steer_rate: float = math.radians(180)


class SlipBicyclePhysics:
    """Single-track (bicycle) model with a linear tire law saturated by a
    friction-circle limit (mu * Fz per axle, static load split only -- no
    dynamic load transfer, that is Phase 1's job). Satisfies
    ars.core.interfaces.VehiclePhysics structurally."""

    def __init__(self, params: SlipBicycleParams | None = None):
        self.params = params or SlipBicycleParams()
        p = self.params
        wheelbase = p.dist_cg_to_front + p.dist_cg_to_rear
        self._fz_front = p.mass * G * p.dist_cg_to_rear / wheelbase
        self._fz_rear = p.mass * G * p.dist_cg_to_front / wheelbase

    def reset(self, x: float, y: float, heading: float) -> VehicleState:
        return VehicleState(x=x, y=y, heading=heading, vx=5.0, vy=0.0, yaw_rate=0.0, steer_angle=0.0)

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
        # dead stop -- the actual equations of motion below use the real vx,
        # never this floor (using the floored value there would inject a
        # fake forcing term once vx settles at 0, making vy run away).
        vx_for_slip = max(vx, 1.0)

        # Standard single-track slip-angle convention (e.g. Rajamani, Vehicle
        # Dynamics and Control): alpha is the angle between the wheel's
        # heading and its actual velocity vector; force opposes it via
        # Fy = -C * alpha. Getting this backwards (as an earlier version of
        # this script did) makes the tire force reinforce a slide instead of
        # arresting it -- a runaway sign bug, not a realistic spin.
        alpha_f = math.atan2(vy + p.dist_cg_to_front * r, vx_for_slip) - steer_angle
        alpha_r = math.atan2(vy - p.dist_cg_to_rear * r, vx_for_slip)

        fy_f = _clamp(-p.cornering_stiffness_front * alpha_f, -p.mu * self._fz_front, p.mu * self._fz_front)
        fy_r = _clamp(-p.cornering_stiffness_rear * alpha_r, -p.mu * self._fz_rear, p.mu * self._fz_rear)

        fx = p.mass * (throttle * p.max_accel - brake * p.max_decel) - p.drag_coeff * p.mass * vx

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


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def run_spike() -> dict:
    dt = 0.02
    max_sim_time = 60.0
    track = make_simple_oval()  # straight_length=80, turn_radius=25, width=10
    physics = SlipBicyclePhysics()
    controller = DummyExpertAgent(track)

    state = physics.reset(x=0.0, y=0.0, heading=0.0)
    t = 0.0
    max_lateral_accel = 0.0
    off_track_time = 0.0
    lap_times: list[float] = []
    last_s = 0.0
    lap_start_time = 0.0
    total_laps_started = 0

    while t < max_sim_time:
        action_arr = controller.act(state)
        action = VehicleAction(throttle=float(action_arr[0]), brake=float(action_arr[1]), steer=float(action_arr[2]))
        state = physics.step(state, action, dt)
        t += dt

        sample = track.query(state.x, state.y)
        if abs(sample.lateral_offset) > sample.width / 2.0:
            off_track_time += dt

        lateral_accel = abs(state.vx * state.yaw_rate) / G
        max_lateral_accel = max(max_lateral_accel, lateral_accel)

        # lap detection: s wraps from near track.length back to ~0
        if sample.s < last_s - track.length / 2.0:
            if total_laps_started > 0:
                lap_times.append(t - lap_start_time)
            lap_start_time = t
            total_laps_started += 1
        last_s = sample.s

    turn_radius = 25.0
    friction_circle_corner_speed = math.sqrt(physics.params.mu * G * turn_radius)  # v_max = sqrt(mu*g*r), no aero

    return {
        "sim_time_s": t,
        "off_track_time_s": off_track_time,
        "off_track_fraction": off_track_time / t if t else 0.0,
        "laps_completed": len(lap_times),
        "lap_times_s": lap_times,
        "best_lap_time_s": min(lap_times) if lap_times else None,
        "max_lateral_accel_g": max_lateral_accel,
        "friction_circle_limit_g": physics.params.mu,
        "theoretical_corner_speed_ms": friction_circle_corner_speed,
        "theoretical_corner_speed_kmh": friction_circle_corner_speed * 3.6,
    }


if __name__ == "__main__":
    results = run_spike()
    for key, value in results.items():
        print(f"{key}: {value}")
