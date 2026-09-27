import math

import pytest

from ars.core.types import VehicleAction
from ars.physics.dynamic_bicycle import DynamicBicycleParams, DynamicBicyclePhysics


def test_reset_is_at_rest():
    physics = DynamicBicyclePhysics()
    state = physics.reset(0.0, 0.0, 0.0)
    assert state.vx == 0.0
    assert state.vy == 0.0


def test_full_throttle_accelerates():
    physics = DynamicBicyclePhysics()
    state = physics.reset(0.0, 0.0, 0.0)
    action = VehicleAction(throttle=1.0, brake=0.0, steer=0.0)
    next_state = physics.step(state, action, dt=0.1)
    assert next_state.vx > state.vx


def test_brake_decelerates():
    physics = DynamicBicyclePhysics()
    state = physics.reset(0.0, 0.0, 0.0)
    state.vx = 10.0
    action = VehicleAction(throttle=0.0, brake=1.0, steer=0.0)
    next_state = physics.step(state, action, dt=0.1)
    assert next_state.vx < state.vx


def test_straight_driving_has_no_lateral_velocity():
    physics = DynamicBicyclePhysics()
    state = physics.reset(0.0, 0.0, 0.0)
    action = VehicleAction(throttle=0.5, brake=0.0, steer=0.0)
    for _ in range(50):
        state = physics.step(state, action, dt=0.02)
    assert state.vy == pytest.approx(0.0, abs=1e-9)


def test_speed_never_exceeds_max():
    physics = DynamicBicyclePhysics()
    state = physics.reset(0.0, 0.0, 0.0)
    action = VehicleAction(throttle=1.0, brake=0.0, steer=0.0)
    for _ in range(3000):
        state = physics.step(state, action, dt=0.1)
    assert state.vx <= physics.params.max_speed + 1e-6


def test_heavier_vehicle_accelerates_more_slowly():
    # Wiring mass into the dynamics (IMPLEMENTATION_PLAN.md Phase 1 task):
    # the same engine force should produce less acceleration for a heavier
    # vehicle, since accel = force / mass.
    light = DynamicBicyclePhysics(DynamicBicycleParams(mass=700.0))
    heavy = DynamicBicyclePhysics(DynamicBicycleParams(mass=1200.0))
    action = VehicleAction(throttle=1.0, brake=0.0, steer=0.0)

    light_state = light.step(light.reset(0.0, 0.0, 0.0), action, dt=0.1)
    heavy_state = heavy.step(heavy.reset(0.0, 0.0, 0.0), action, dt=0.1)

    assert heavy_state.vx < light_state.vx


def test_lateral_slip_produces_a_restoring_force_not_a_runaway():
    # Regression test for the sign-convention bug found during the Phase 0
    # spike (see PHASE0_SPIKE_NOTE.md): getting the slip-angle sign backwards
    # makes the tire force reinforce a slide instead of opposing it, so
    # lateral velocity runs away instead of settling. Perturb vy off zero
    # with the wheel pointed straight (steer=0) and confirm the tire force
    # pushes vy back toward zero, not further away.
    physics = DynamicBicyclePhysics()
    state = physics.reset(0.0, 0.0, 0.0)
    state.vx = 20.0
    state.vy = 3.0  # perturbation: sliding sideways with wheels straight
    action = VehicleAction(throttle=0.0, brake=0.0, steer=0.0)

    next_state = physics.step(state, action, dt=0.02)

    assert next_state.vy < state.vy  # restoring, not amplifying
    assert math.isfinite(next_state.vy)


def test_skid_pad_reaches_a_finite_steady_state_without_a_hard_cap():
    # IMPLEMENTATION_PLAN.md Phase 1 exit criterion: "passes the skid-pad
    # test" -- sustained cornering settles to a physically plausible,
    # finite steady state; grip emerges from tire saturation, not an
    # imposed ceiling (the hard max_lateral_accel cap is gone, see
    # test_no_hard_lateral_accel_cap_below).
    physics = DynamicBicyclePhysics()
    state = physics.reset(0.0, 0.0, 0.0)
    action = VehicleAction(throttle=0.3, brake=0.0, steer=0.3)  # sustained turn-in

    max_lateral_accel_g = 0.0
    for _ in range(3000):  # 60 s at dt=0.02
        state = physics.step(state, action, dt=0.02)
        assert math.isfinite(state.vx)
        assert math.isfinite(state.vy)
        assert math.isfinite(state.yaw_rate)
        lateral_accel_g = abs(state.vx * state.yaw_rate) / 9.81
        max_lateral_accel_g = max(max_lateral_accel_g, lateral_accel_g)

    assert 0.0 < max_lateral_accel_g < 6.0  # plausible tire-limited range, not unbounded


def test_no_hard_lateral_accel_cap_below():
    # The old kinematic stub's max_lateral_accel scalar cap is gone --
    # grip now emerges from the tire model, not an imposed ceiling.
    params = DynamicBicycleParams()
    assert not hasattr(params, "max_lateral_accel")
