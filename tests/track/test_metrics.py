import math

import pytest

from ars.track.loop_track import Arc, FixedLoopTrack
from ars.track.metrics import track_curvature_feature_vector


def _pure_circle(radius: float = 30.0) -> FixedLoopTrack:
    # Two half-circle arcs, same radius/direction -- a perfect circle.
    return FixedLoopTrack([Arc(radius=radius, angle=math.pi), Arc(radius=radius, angle=math.pi)])


def test_pure_circle_has_constant_curvature_and_zero_variance():
    radius = 30.0
    track = _pure_circle(radius)
    mean_curvature, variance, min_radius = track_curvature_feature_vector(track, sample_spacing_m=1.0)

    assert mean_curvature == pytest.approx(1.0 / radius, rel=1e-3)
    assert variance == pytest.approx(0.0, abs=1e-9)
    assert min_radius == pytest.approx(radius, rel=1e-3)


def test_oval_mean_curvature_is_between_zero_and_the_turn_curvature():
    from ars.track.loop_track import make_simple_oval

    turn_radius = 25.0
    track = make_simple_oval(straight_length=80.0, turn_radius=turn_radius, width=10.0)
    mean_curvature, variance, min_radius = track_curvature_feature_vector(track, sample_spacing_m=1.0)

    assert 0.0 < mean_curvature < 1.0 / turn_radius
    assert variance > 0.0  # mixes straights (curvature 0) and arcs (curvature 1/r)
    assert min_radius == pytest.approx(turn_radius, rel=1e-3)


def test_sample_spacing_does_not_change_result_much():
    track = _pure_circle(radius=20.0)
    coarse = track_curvature_feature_vector(track, sample_spacing_m=2.0)
    fine = track_curvature_feature_vector(track, sample_spacing_m=0.25)

    assert coarse[0] == pytest.approx(fine[0], rel=1e-2)
    assert coarse[2] == pytest.approx(fine[2], rel=1e-2)
