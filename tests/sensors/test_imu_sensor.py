import pytest

from ars.core.types import VehicleState
from ars.sensors import ImuSensor
from ars.track import make_simple_oval


def _state(vx=0.0, vy=0.0, yaw_rate=0.0):
    return VehicleState(x=0.0, y=0.0, heading=0.0, vx=vx, vy=vy, yaw_rate=yaw_rate)


def test_first_read_reports_zero_acceleration():
    track = make_simple_oval()
    sensor = ImuSensor(dt=0.1)
    frame = sensor.read(_state(vx=10.0), track)
    ax, ay, yaw_rate = frame.readings["imu"]
    assert ax == 0.0
    assert ay == 0.0


def test_reports_gyroscope_yaw_rate_directly():
    track = make_simple_oval()
    sensor = ImuSensor(dt=0.1)
    frame = sensor.read(_state(yaw_rate=0.42), track)
    assert frame.readings["imu"][2] == pytest.approx(0.42)


def test_computes_acceleration_from_velocity_change():
    track = make_simple_oval()
    sensor = ImuSensor(dt=0.1)
    sensor.read(_state(vx=10.0, vy=0.0), track)  # establishes previous velocity
    frame = sensor.read(_state(vx=11.0, vy=0.0), track)
    ax, ay, _ = frame.readings["imu"]
    assert ax == pytest.approx(10.0)  # (11-10)/0.1
    assert ay == pytest.approx(0.0)


def test_reset_clears_previous_velocity():
    track = make_simple_oval()
    sensor = ImuSensor(dt=0.1)
    sensor.read(_state(vx=50.0), track)  # simulate stale velocity from a prior episode
    sensor.reset(_state(vx=0.0))
    frame = sensor.read(_state(vx=0.0), track)
    ax, ay, _ = frame.readings["imu"]
    assert ax == 0.0, "reset should prevent a spurious jump from the previous episode's velocity"


def test_racing_env_reset_calls_sensor_reset():
    # Regression test: RacingEnv.reset() must call sensor.reset() BEFORE
    # building the new episode's first observation, or the IMU reading
    # inside reset()'s own returned obs would difference against the
    # previous episode's final velocity -- a spurious spike, not zero.
    #
    # RacingEnv.reset() calls sensor.reset() then immediately sensor.read()
    # (to build the obs), so by the time this test can inspect the sensor,
    # _prev_vx is already the new episode's starting velocity (0.0) rather
    # than None -- that's expected. What actually catches the bug is the
    # ACCELERATION reported in the returned obs: without the reset() call,
    # it would compute (0.0 - <fast speed from last episode>) / dt, a large
    # negative spike, instead of 0.0 (first-ever-read behavior).
    import numpy as np

    from ars.env import make_default_env

    env = make_default_env(off_track_terminates=False, max_episode_steps=50)
    env.reset()

    action = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    for _ in range(50):
        env.step(action)  # builds up real speed before the episode ends

    obs, info = env.reset()  # new episode -- physics resets to vx=0

    imu_sensor = [s for s in env.sensors if s.name == "imu"][0]
    imu_index = env.sensors.index(imu_sensor)
    offset = sum(len(s.read(env.vehicle_state, env.track).readings[s.name]) for s in env.sensors[:imu_index])
    ax = obs[offset]
    assert ax == 0.0, "IMU accel in reset()'s own obs should be 0, not a spike from the previous episode"
