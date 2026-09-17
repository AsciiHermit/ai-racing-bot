import pytest

from ars.core.types import VehicleAction
from ars.physics import KinematicBicyclePhysics


def test_reset_is_at_rest():
    physics = KinematicBicyclePhysics()
    state = physics.reset(0.0, 0.0, 0.0)
    assert state.vx == 0.0
    assert state.vy == 0.0


def test_full_throttle_accelerates():
    physics = KinematicBicyclePhysics()
    state = physics.reset(0.0, 0.0, 0.0)
    action = VehicleAction(throttle=1.0, brake=0.0, steer=0.0)
    next_state = physics.step(state, action, dt=0.1)
    assert next_state.vx > state.vx


def test_speed_never_exceeds_max():
    physics = KinematicBicyclePhysics()
    state = physics.reset(0.0, 0.0, 0.0)
    action = VehicleAction(throttle=1.0, brake=0.0, steer=0.0)
    for _ in range(2000):
        state = physics.step(state, action, dt=0.1)
    assert state.vx <= physics.params.max_speed + 1e-6


def test_brake_decelerates():
    physics = KinematicBicyclePhysics()
    state = physics.reset(0.0, 0.0, 0.0)
    state.vx = 10.0
    action = VehicleAction(throttle=0.0, brake=1.0, steer=0.0)
    next_state = physics.step(state, action, dt=0.1)
    assert next_state.vx < state.vx


def test_straight_driving_has_no_lateral_velocity():
    physics = KinematicBicyclePhysics()
    state = physics.reset(0.0, 0.0, 0.0)
    action = VehicleAction(throttle=0.5, brake=0.0, steer=0.0)
    next_state = physics.step(state, action, dt=0.1)
    assert next_state.vy == 0.0


def test_lateral_acceleration_never_exceeds_grip_cap():
    # Regression test for the "infinite grip" bug: without a cap, full
    # throttle + full steer could reach top speed while cornering at max
    # steer angle, implying lateral g-forces far beyond real tire grip.
    physics = KinematicBicyclePhysics()
    state = physics.reset(0.0, 0.0, 0.0)
    action = VehicleAction(throttle=1.0, brake=0.0, steer=1.0)  # full throttle, full steer
    max_lat_accel = 0.0
    for _ in range(3000):
        state = physics.step(state, action, dt=0.02)
        lat_accel = abs(state.vx * state.yaw_rate)
        max_lat_accel = max(max_lat_accel, lat_accel)
    assert max_lat_accel <= physics.params.max_lateral_accel + 1e-6


def test_grip_cap_limits_corner_speed_below_straight_line_top_speed():
    physics = KinematicBicyclePhysics()
    state = physics.reset(0.0, 0.0, 0.0)
    action = VehicleAction(throttle=1.0, brake=0.0, steer=1.0)  # sustained max steer
    for _ in range(3000):
        state = physics.step(state, action, dt=0.02)
    assert state.vx < physics.params.max_speed
