import math

import pytest

from ars.core.interfaces import Track
from ars.track.spline_track import SplineTrack


def _diamond_points() -> list[tuple[float, float]]:
    # A simple closed loop: a diamond, large enough that a constant width
    # of 10 m fits comfortably without self-overlap.
    return [(100.0, 0.0), (0.0, 100.0), (-100.0, 0.0), (0.0, -100.0)]


def test_satisfies_track_protocol():
    track = SplineTrack(_diamond_points(), width=10.0)
    assert isinstance(track, Track)


def test_closed_loop_wraps_back_to_start():
    track = SplineTrack(_diamond_points(), width=10.0)
    start = track.sample_at_s(0.0)
    end = track.sample_at_s(track.length - 1e-6)
    # sample_at_s(length) should be right back near sample_at_s(0) -- same
    # closure property FixedLoopTrack enforces for its own segment tracks.
    dx = end.centerline_x - start.centerline_x
    dy = end.centerline_y - start.centerline_y
    assert math.hypot(dx, dy) < 5.0


def test_curvature_is_finite_everywhere():
    track = SplineTrack(_diamond_points(), width=10.0)
    num_samples = 200
    for i in range(num_samples):
        s = i * track.length / num_samples
        sample = track.sample_at_s(s)
        assert math.isfinite(sample.curvature)
        assert math.isfinite(sample.heading)


def test_is_on_track_true_at_centerline_false_far_away():
    track = SplineTrack(_diamond_points(), width=10.0)
    sample = track.sample_at_s(0.0)
    assert track.is_on_track(sample.centerline_x, sample.centerline_y)
    assert not track.is_on_track(sample.centerline_x + 1000.0, sample.centerline_y + 1000.0)


def test_query_returns_near_zero_offset_at_a_centerline_point():
    track = SplineTrack(_diamond_points(), width=10.0)
    reference = track.sample_at_s(track.length * 0.3)
    result = track.query(reference.centerline_x, reference.centerline_y)
    assert abs(result.lateral_offset) < 1.0  # within one coarse sampling step's slack


def test_query_offset_sign_matches_side_of_centerline():
    track = SplineTrack(_diamond_points(), width=10.0)
    sample = track.sample_at_s(0.0)
    nx = -math.sin(sample.heading)
    ny = math.cos(sample.heading)
    left_point = (sample.centerline_x + nx * 2.0, sample.centerline_y + ny * 2.0)
    right_point = (sample.centerline_x - nx * 2.0, sample.centerline_y - ny * 2.0)

    left = track.query(*left_point)
    right = track.query(*right_point)
    assert left.lateral_offset > 0
    assert right.lateral_offset < 0
