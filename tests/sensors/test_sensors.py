import math

from ars.physics import KinematicBicyclePhysics
from ars.sensors import LidarSensor, ProprioceptiveSensor, TrackPoseSensor
from ars.track import make_simple_oval


def test_pose_sensor_zero_offset_on_centerline():
    track = make_simple_oval()
    physics = KinematicBicyclePhysics()
    sample = track.sample_at_s(0.0)
    state = physics.reset(sample.centerline_x, sample.centerline_y, sample.heading)
    frame = TrackPoseSensor().read(state, track)
    lateral_offset = frame.readings["track_pose"][0]
    assert abs(lateral_offset) < 1e-2


def test_proprioceptive_sensor_shape():
    track = make_simple_oval()
    physics = KinematicBicyclePhysics()
    state = physics.reset(0.0, 0.0, 0.0)
    frame = ProprioceptiveSensor().read(state, track)
    assert frame.readings["proprioceptive"].shape == (4,)


def test_lidar_sensor_returns_num_rays():
    track = make_simple_oval()
    physics = KinematicBicyclePhysics()
    sample = track.sample_at_s(0.0)
    state = physics.reset(sample.centerline_x, sample.centerline_y, sample.heading)
    sensor = LidarSensor(num_rays=5)
    frame = sensor.read(state, track)
    assert frame.readings["lidar"].shape == (5,)


def test_lidar_sensor_detects_boundary_within_max_range():
    track = make_simple_oval(width=10.0)
    physics = KinematicBicyclePhysics()
    sample = track.sample_at_s(0.0)
    state = physics.reset(sample.centerline_x, sample.centerline_y, sample.heading)
    # 3 rays spanning fov=pi -> leftmost/rightmost point 90 degrees off
    # heading, straight at the nearby track edge, well within max_range.
    sensor = LidarSensor(num_rays=3, fov=math.pi, max_range=40.0)
    frame = sensor.read(state, track)
    assert frame.readings["lidar"][0] < 40.0
    assert frame.readings["lidar"][-1] < 40.0
