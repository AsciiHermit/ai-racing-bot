import math

import pytest

from ars.track import make_simple_oval


def test_oval_is_closed():
    track = make_simple_oval()
    assert track.length > 0


def test_sample_at_s_wraps():
    track = make_simple_oval()
    a = track.sample_at_s(0.0)
    b = track.sample_at_s(track.length)
    assert a.centerline_x == pytest.approx(b.centerline_x, abs=1e-6)
    assert a.centerline_y == pytest.approx(b.centerline_y, abs=1e-6)


def test_query_on_centerline_has_zero_offset():
    track = make_simple_oval()
    sample = track.sample_at_s(10.0)
    query = track.query(sample.centerline_x, sample.centerline_y)
    assert query.lateral_offset == pytest.approx(0.0, abs=1e-2)


def test_is_on_track_within_width():
    track = make_simple_oval(width=10.0)
    sample = track.sample_at_s(5.0)
    assert track.is_on_track(sample.centerline_x, sample.centerline_y)


def test_is_off_track_beyond_width():
    track = make_simple_oval(width=10.0)
    sample = track.sample_at_s(5.0)
    nx = -math.sin(sample.heading)
    ny = math.cos(sample.heading)
    far_x = sample.centerline_x + nx * 20.0
    far_y = sample.centerline_y + ny * 20.0
    assert not track.is_on_track(far_x, far_y)


def test_straight_has_zero_curvature():
    track = make_simple_oval()
    sample = track.sample_at_s(1.0)
    assert sample.curvature == pytest.approx(0.0)


def test_segments_lengths_sum_to_track_length():
    track = make_simple_oval()
    segments = track.segments()
    assert sum(seg.length for seg in segments) == pytest.approx(track.length)


def test_segments_report_kind_and_radius():
    track = make_simple_oval(straight_length=80.0, turn_radius=25.0)
    segments = track.segments()
    kinds = [seg.kind for seg in segments]
    assert kinds == ["straight", "arc", "straight", "arc"]
    arc_segments = [seg for seg in segments if seg.kind == "arc"]
    for seg in arc_segments:
        assert seg.radius == pytest.approx(25.0)
        assert seg.angle == pytest.approx(math.pi)
