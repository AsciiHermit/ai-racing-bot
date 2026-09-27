"""Phase 2 track: a closed-loop, uniform Catmull-Rom spline through a
sequence of control points, constant width. Satisfies the
ars.core.interfaces.Track Protocol structurally, same as FixedLoopTrack --
this is what lets the procedural generator (ars.track.generator) plug in
without changing ars/env, ars/sensors, or ars/viz.

Unlike FixedLoopTrack's closed-form arc/line projection, a spline has no
simple closed-form nearest-point projection or arc-length parameterization,
so this builds a fine table of (s, x, y, heading, curvature) samples once
at construction time and works from that: sample_at_s() interpolates
between table entries, query() does a nearest-table-entry search. Curvature
at each table entry is computed analytically from the local cubic
segment's own first/second derivatives (per wayfinder ticket #11's
resolution), not by finite-differencing the table itself.

Nearest-point search is a linear scan over the table -- fine for v1's
track sizes (~1000-2000 samples); swap for a spatial index if tracks grow
much larger, same note FixedLoopTrack's own docstring makes.
"""
from __future__ import annotations

import math

from ars.core.types import TrackSample

_SAMPLES_PER_SEGMENT = 40  # fine subdivision for the arc-length/curvature table


class SplineTrack:
    """Satisfies ars.core.interfaces.Track structurally (no inheritance
    needed -- see that module for the contract)."""

    def __init__(self, control_points: list[tuple[float, float]], width: float = 10.0):
        if len(control_points) < 3:
            raise ValueError("a closed spline track needs at least 3 control points")
        self.width = width
        self._points = control_points
        self._n = len(control_points)
        self._table = self._build_table()
        self.length = self._table[-1][0]

    def _control(self, i: int) -> tuple[float, float]:
        return self._points[i % self._n]

    def _build_table(self) -> list[tuple[float, float, float, float, float]]:
        """One (s, x, y, heading, curvature) row per fine sample, in order,
        wrapping all the way around the closed loop back to (approximately)
        the start."""
        table: list[tuple[float, float, float, float, float]] = []
        s = 0.0
        prev_x: float | None = None
        prev_y: float | None = None
        for seg in range(self._n):
            p0 = self._control(seg - 1)
            p1 = self._control(seg)
            p2 = self._control(seg + 1)
            p3 = self._control(seg + 2)
            for k in range(_SAMPLES_PER_SEGMENT):
                t = k / _SAMPLES_PER_SEGMENT
                x, y = _catmull_rom_point(p0, p1, p2, p3, t)
                dx, dy = _catmull_rom_first_derivative(p0, p1, p2, p3, t)
                ddx, ddy = _catmull_rom_second_derivative(p0, p1, p2, p3, t)
                heading = math.atan2(dy, dx)
                denom = (dx * dx + dy * dy) ** 1.5
                curvature = (dx * ddy - dy * ddx) / denom if denom > 1e-12 else 0.0
                if prev_x is not None:
                    s += math.hypot(x - prev_x, y - prev_y)
                table.append((s, x, y, heading, curvature))
                prev_x, prev_y = x, y
        # Close the loop: distance from the last sample back to the first.
        first_x, first_y = table[0][1], table[0][2]
        s += math.hypot(first_x - prev_x, first_y - prev_y)
        table.append((s, first_x, first_y, table[0][3], table[0][4]))
        return table

    def sample_at_s(self, s: float) -> TrackSample:
        s = s % self.length
        table = self._table
        lo, hi = 0, len(table) - 1
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if table[mid][0] <= s:
                lo = mid
            else:
                hi = mid
        s0, x0, y0, h0, c0 = table[lo]
        s1, x1, y1, h1, c1 = table[hi]
        span = s1 - s0
        frac = (s - s0) / span if span > 1e-12 else 0.0
        x = x0 + (x1 - x0) * frac
        y = y0 + (y1 - y0) * frac
        heading = h0 + _wrap(h1 - h0) * frac
        curvature = c0 + (c1 - c0) * frac
        return TrackSample(
            centerline_x=x, centerline_y=y, heading=_wrap(heading), curvature=curvature,
            width=self.width, s=s, lateral_offset=0.0,
        )

    def query(self, x: float, y: float) -> TrackSample:
        best_idx = 0
        best_dist2 = math.inf
        for i, (_, tx, ty, _, _) in enumerate(self._table):
            d2 = (tx - x) ** 2 + (ty - y) ** 2
            if d2 < best_dist2:
                best_dist2 = d2
                best_idx = i
        best_s = self._table[best_idx][0]
        best = self.sample_at_s(best_s)
        signed_lateral = _signed_lateral_offset(best, x, y)
        return TrackSample(
            centerline_x=best.centerline_x, centerline_y=best.centerline_y, heading=best.heading,
            curvature=best.curvature, width=best.width, s=best.s, lateral_offset=signed_lateral,
        )

    def is_on_track(self, x: float, y: float) -> bool:
        sample = self.query(x, y)
        return abs(sample.lateral_offset) <= sample.width / 2.0

    def control_points(self) -> list[tuple[float, float]]:
        """Public read-only copy of the control points this track was built
        from -- e.g. for the diversity check (comparing generated tracks by
        their raw control points) or a track visualizer. Not used
        internally; see _build_table() for the geometry the track math
        actually runs on."""
        return list(self._points)


def _catmull_rom_point(p0, p1, p2, p3, t: float) -> tuple[float, float]:
    t2, t3 = t * t, t * t * t
    x = 0.5 * (
        2 * p1[0] + (-p0[0] + p2[0]) * t + (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2
        + (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3
    )
    y = 0.5 * (
        2 * p1[1] + (-p0[1] + p2[1]) * t + (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2
        + (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3
    )
    return x, y


def _catmull_rom_first_derivative(p0, p1, p2, p3, t: float) -> tuple[float, float]:
    t2 = t * t
    dx = 0.5 * (
        (-p0[0] + p2[0]) + 2 * (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t
        + 3 * (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t2
    )
    dy = 0.5 * (
        (-p0[1] + p2[1]) + 2 * (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t
        + 3 * (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t2
    )
    return dx, dy


def _catmull_rom_second_derivative(p0, p1, p2, p3, t: float) -> tuple[float, float]:
    ddx = 0.5 * (2 * (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) + 6 * (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t)
    ddy = 0.5 * (2 * (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) + 6 * (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t)
    return ddx, ddy


def _signed_lateral_offset(sample: TrackSample, x: float, y: float) -> float:
    dx = x - sample.centerline_x
    dy = y - sample.centerline_y
    nx = -math.sin(sample.heading)
    ny = math.cos(sample.heading)
    return dx * nx + dy * ny


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi
