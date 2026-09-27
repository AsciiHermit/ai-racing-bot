import math

from ars.core.interfaces import Track
from ars.track.generator import generate_track
from ars.track.metrics import track_curvature_feature_vector


def test_same_seed_is_deterministic():
    a = generate_track(seed=42)
    b = generate_track(seed=42)
    assert a.control_points() == b.control_points()
    assert a.length == b.length


def test_different_seeds_produce_different_tracks():
    a = generate_track(seed=1)
    b = generate_track(seed=2)
    assert a.control_points() != b.control_points()


def test_generated_track_satisfies_protocol_and_is_closed():
    track = generate_track(seed=7)
    assert isinstance(track, Track)
    assert track.length > 0

    start = track.sample_at_s(0.0)
    end = track.sample_at_s(track.length - 1e-6)
    dx = end.centerline_x - start.centerline_x
    dy = end.centerline_y - start.centerline_y
    assert math.hypot(dx, dy) < 5.0


def test_generated_track_feature_vector_is_finite_and_nondegenerate():
    track = generate_track(seed=7)
    mean_curvature, variance, min_radius = track_curvature_feature_vector(track)
    assert math.isfinite(mean_curvature) and mean_curvature > 0.0
    assert math.isfinite(variance)
    assert math.isfinite(min_radius) and min_radius > 0.0


def test_num_control_points_is_respected():
    track = generate_track(seed=7, num_control_points=8)
    assert len(track.control_points()) == 8


def test_width_is_applied():
    track = generate_track(seed=7, width=14.0)
    assert track.width == 14.0
