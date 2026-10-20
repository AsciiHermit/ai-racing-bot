from ars.core.interfaces import Sensor, Track, VehiclePhysics
from ars.physics import FourWheelPhysics
from ars.sensors import GpsSensor, ImuSensor, TrackPoseSensor
from ars.track import make_simple_oval


def test_four_wheel_physics_satisfies_protocol():
    assert isinstance(FourWheelPhysics(), VehiclePhysics)


def test_fixed_loop_track_satisfies_protocol():
    assert isinstance(make_simple_oval(), Track)


def test_pose_sensor_satisfies_protocol():
    assert isinstance(TrackPoseSensor(), Sensor)


def test_gps_sensor_satisfies_protocol():
    assert isinstance(GpsSensor(), Sensor)


def test_imu_sensor_satisfies_protocol():
    assert isinstance(ImuSensor(), Sensor)
