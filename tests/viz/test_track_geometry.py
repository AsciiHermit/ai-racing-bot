import pytest

from ars.track import make_simple_oval
from ars.viz.track_geometry import build_track_polylines


def test_polylines_close_the_loop():
    track = make_simple_oval()
    polylines = build_track_polylines(track)
    assert polylines.centerline[0] == polylines.centerline[-1]
    assert polylines.inner_boundary[0] == polylines.inner_boundary[-1]
    assert polylines.outer_boundary[0] == polylines.outer_boundary[-1]


def test_boundaries_are_offset_from_centerline_by_half_width():
    track = make_simple_oval(width=10.0)
    polylines = build_track_polylines(track)
    cx, cy = polylines.centerline[0]
    ix, iy = polylines.inner_boundary[0]
    ox, oy = polylines.outer_boundary[0]
    dist_inner = ((cx - ix) ** 2 + (cy - iy) ** 2) ** 0.5
    dist_outer = ((cx - ox) ** 2 + (cy - oy) ** 2) ** 0.5
    assert dist_inner == pytest.approx(5.0, abs=1e-6)
    assert dist_outer == pytest.approx(5.0, abs=1e-6)


def test_bounds_cover_all_boundary_points():
    track = make_simple_oval()
    polylines = build_track_polylines(track)
    min_x, max_x, min_y, max_y = polylines.bounds
    for x, y in polylines.outer_boundary + polylines.inner_boundary:
        assert min_x <= x <= max_x
        assert min_y <= y <= max_y
