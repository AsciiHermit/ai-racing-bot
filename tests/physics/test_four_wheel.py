import math

import pytest

from ars.core.types import VehicleAction
from ars.physics.four_wheel import FourWheelPhysics

G = 9.81


def _axle_loads(state):
    """(front_total, rear_total) normal load across both wheels of each axle."""
    fl, fr, rl, rr = state.wheels
    return fl.load + fr.load, rl.load + rr.load


def test_reset_is_at_rest():
    physics = FourWheelPhysics()
    state = physics.reset(0.0, 0.0, 0.0)
    assert state.vx == 0.0
    assert state.vy == 0.0


def test_reset_wheel_loads_sum_to_static_weight():
    # At rest, no aero and no load transfer -- total normal force across
    # all 4 wheels must equal the car's static weight, mass * g.
    physics = FourWheelPhysics()
    state = physics.reset(0.0, 0.0, 0.0)
    total_load = sum(w.load for w in state.wheels)
    expected = physics.params.mass_kg * G
    assert total_load == pytest.approx(expected, rel=1e-6)


def test_full_throttle_accelerates():
    physics = FourWheelPhysics()
    state = physics.reset(0.0, 0.0, 0.0)
    action = VehicleAction(throttle=1.0, brake=0.0, steer=0.0)
    next_state = physics.step(state, action, dt=0.1)
    assert next_state.vx > state.vx


def test_speed_never_exceeds_max():
    physics = FourWheelPhysics()
    state = physics.reset(0.0, 0.0, 0.0)
    action = VehicleAction(throttle=1.0, brake=0.0, steer=0.0)
    for _ in range(2000):
        state = physics.step(state, action, dt=0.1)
    speed = math.hypot(state.vx, state.vy)
    assert speed <= physics.params.max_speed + 1e-6


def test_brake_decelerates():
    physics = FourWheelPhysics()
    state = physics.reset(0.0, 0.0, 0.0)
    state.vx = 10.0
    action = VehicleAction(throttle=0.0, brake=1.0, steer=0.0)
    next_state = physics.step(state, action, dt=0.1)
    assert next_state.vx < state.vx


def test_wheel_loads_stay_nonnegative_under_hard_braking():
    physics = FourWheelPhysics()
    state = physics.reset(0.0, 0.0, 0.0)
    state.vx = 40.0
    action = VehicleAction(throttle=0.0, brake=1.0, steer=0.0)
    for _ in range(50):
        state = physics.step(state, action, dt=0.02)
        assert all(w.load >= 0.0 for w in state.wheels)


def test_acceleration_transfers_load_rearward():
    # Replaces the old stub's "no lateral velocity" test, which tested an
    # assumption (vy always 0) that no longer holds for this model.
    # Instead, test the thing that's actually new here: load transfer.
    physics = FourWheelPhysics()
    p = physics.params
    static_front = p.mass_kg * G * p.weight_dist_front
    static_rear = p.mass_kg * G * (1.0 - p.weight_dist_front)

    state = physics.reset(0.0, 0.0, 0.0)
    state.vx = 20.0  # already moving, so the accel request is unambiguous
    action = VehicleAction(throttle=1.0, brake=0.0, steer=0.0)
    next_state = physics.step(state, action, dt=0.02)

    front_total, rear_total = _axle_loads(next_state)
    assert front_total < static_front
    assert rear_total > static_rear


def test_braking_transfers_load_forward():
    physics = FourWheelPhysics()
    p = physics.params
    static_front = p.mass_kg * G * p.weight_dist_front
    static_rear = p.mass_kg * G * (1.0 - p.weight_dist_front)

    state = physics.reset(0.0, 0.0, 0.0)
    state.vx = 20.0
    action = VehicleAction(throttle=0.0, brake=1.0, steer=0.0)
    next_state = physics.step(state, action, dt=0.02)

    front_total, rear_total = _axle_loads(next_state)
    assert front_total > static_front
    assert rear_total < static_rear


def test_cornering_reduces_top_speed_below_straight_line_max():
    # Equivalent of the old stub's "grip cap limits corner speed"
    # regression test: sustained full throttle + full steer must top out
    # below the straight-line max_speed, because the friction circle
    # forces some of each tire's grip budget to go to cornering instead
    # of traction.
    physics = FourWheelPhysics()
    state = physics.reset(0.0, 0.0, 0.0)
    action = VehicleAction(throttle=1.0, brake=0.0, steer=1.0)
    for _ in range(3000):
        state = physics.step(state, action, dt=0.02)
    speed = math.hypot(state.vx, state.vy)
    assert speed < physics.params.max_speed


def test_hard_steering_reduces_achieved_acceleration():
    # Direct check on the friction-circle mechanism itself (not just its
    # downstream effect on top speed): at the same speed and throttle,
    # steering hard must produce less forward acceleration than driving
    # straight, because some of each tire's limited grip budget is being
    # spent on cornering force instead of traction.
    physics = FourWheelPhysics()

    straight = physics.reset(0.0, 0.0, 0.0)
    straight.vx = 20.0
    straight_next = physics.step(straight, VehicleAction(throttle=1.0, brake=0.0, steer=0.0), dt=0.02)

    cornering = physics.reset(0.0, 0.0, 0.0)
    cornering.vx = 20.0
    cornering_next = physics.step(cornering, VehicleAction(throttle=1.0, brake=0.0, steer=1.0), dt=0.02)

    assert (cornering_next.vx - cornering.vx) < (straight_next.vx - straight.vx)
