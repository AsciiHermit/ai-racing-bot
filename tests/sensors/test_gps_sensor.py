import pytest

from ars.core.types import VehicleState
from ars.sensors import GpsSensor
from ars.track import make_simple_oval


def test_gps_at_origin_reports_origin_lat_lon():
    track = make_simple_oval()
    sensor = GpsSensor(origin_lat_deg=51.5, origin_lon_deg=-0.1)
    state = VehicleState(x=0.0, y=0.0, heading=0.0, vx=0, vy=0, yaw_rate=0)
    frame = sensor.read(state, track)
    lat, lon = frame.readings["gps"]
    assert lat == pytest.approx(51.5)
    assert lon == pytest.approx(-0.1)


def test_gps_matches_known_degrees_per_meter():
    # ~111km per degree of latitude is the standard reference figure.
    track = make_simple_oval()
    sensor = GpsSensor(origin_lat_deg=0.0, origin_lon_deg=0.0)
    state = VehicleState(x=0.0, y=111_000.0, heading=0.0, vx=0, vy=0, yaw_rate=0)
    frame = sensor.read(state, track)
    lat, lon = frame.readings["gps"]
    assert lat == pytest.approx(1.0, abs=0.01)
    assert lon == pytest.approx(0.0, abs=1e-9)


def test_gps_reading_changes_with_position():
    track = make_simple_oval()
    sensor = GpsSensor()
    state_a = VehicleState(x=0.0, y=0.0, heading=0.0, vx=0, vy=0, yaw_rate=0)
    state_b = VehicleState(x=50.0, y=20.0, heading=0.0, vx=0, vy=0, yaw_rate=0)
    frame_a = sensor.read(state_a, track)
    frame_b = sensor.read(state_b, track)
    assert not (frame_a.readings["gps"] == frame_b.readings["gps"]).all()


def test_gps_is_stateless_across_reset():
    track = make_simple_oval()
    sensor = GpsSensor()
    state = VehicleState(x=10.0, y=5.0, heading=0.0, vx=0, vy=0, yaw_rate=0)
    before = sensor.read(state, track).readings["gps"].copy()
    sensor.reset(state)
    after = sensor.read(state, track).readings["gps"]
    assert (before == after).all()
