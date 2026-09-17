"""v1 fixed track: a closed loop built from straights and constant-radius
arcs (easy left/right turns). No banking, no elevation -- flat 2D only.

Centerline is defined as a sequence of segments, each either a Straight or
an Arc. Geometry queries (nearest point, heading, curvature, arc-length)
project analytically onto each segment (closed-form for both line and
circle) and take the closest -- O(num_segments), no arc-length sampling
loop. Segment count is small and fixed for v1's simple tracks; swap for a
spatial index (KD-tree/grid) later if tracks grow to hundreds of segments,
without changing the Track interface.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from ars.core.types import TrackSample


@dataclass
class Straight:
    length: float


@dataclass
class Arc:
    radius: float  # positive = curves left, negative = curves right
    angle: float  # total turn angle, radians, positive magnitude


Segment = Straight | Arc


@dataclass
class _ResolvedSegment:
    kind: str  # "straight" | "arc"
    s_start: float
    s_end: float
    start_x: float
    start_y: float
    start_heading: float
    length: float
    radius: float = 0.0  # signed, arcs only
    center_x: float = 0.0  # arcs only
    center_y: float = 0.0  # arcs only


class FixedLoopTrack:
    """A closed-loop track built from Straight/Arc segments, constant width.

    Satisfies the ars.core.interfaces.Track Protocol structurally (no
    inheritance needed -- see that module for the contract)."""

    def __init__(self, segments: list[Segment], width: float = 10.0):
        if not segments:
            raise ValueError("track needs at least one segment")
        self.width = width
        self._segments = self._resolve(segments)
        self.length = self._segments[-1].s_end
        self._closure_tolerance = 1e-3
        self._validate_closed()

    def _resolve(self, segments: list[Segment]) -> list[_ResolvedSegment]:
        resolved: list[_ResolvedSegment] = []
        x, y, heading, s = 0.0, 0.0, 0.0, 0.0
        for seg in segments:
            if isinstance(seg, Straight):
                resolved.append(
                    _ResolvedSegment(
                        kind="straight",
                        s_start=s,
                        s_end=s + seg.length,
                        start_x=x,
                        start_y=y,
                        start_heading=heading,
                        length=seg.length,
                    )
                )
                x += seg.length * math.cos(heading)
                y += seg.length * math.sin(heading)
                s += seg.length
            elif isinstance(seg, Arc):
                arc_len = abs(seg.radius) * seg.angle
                resolved.append(
                    _ResolvedSegment(
                        kind="arc",
                        s_start=s,
                        s_end=s + arc_len,
                        start_x=x,
                        start_y=y,
                        start_heading=heading,
                        length=arc_len,
                        radius=seg.radius,
                    )
                )
                turn = seg.angle if seg.radius > 0 else -seg.angle
                cx = x - seg.radius * math.sin(heading)
                cy = y + seg.radius * math.cos(heading)
                resolved[-1].center_x = cx
                resolved[-1].center_y = cy
                heading += turn
                x = cx + seg.radius * math.sin(heading)
                y = cy - seg.radius * math.cos(heading)
                s += arc_len
            else:
                raise TypeError(f"unknown segment type: {seg!r}")
        return resolved

    def _validate_closed(self) -> None:
        end = self.sample_at_s(self.length - 1e-9)
        start = self.sample_at_s(0.0)
        dx = end.centerline_x - start.centerline_x
        dy = end.centerline_y - start.centerline_y
        if math.hypot(dx, dy) > 1.0:
            raise ValueError(
                f"track is not closed: start/end gap = {math.hypot(dx, dy):.3f} m. "
                "Adjust segment lengths/angles so the loop returns to its origin."
            )

    def sample_at_s(self, s: float) -> TrackSample:
        s = s % self.length
        seg = self._segment_at(s)
        ds = s - seg.s_start
        if seg.kind == "straight":
            heading = seg.start_heading
            cx = seg.start_x + ds * math.cos(heading)
            cy = seg.start_y + ds * math.sin(heading)
            curvature = 0.0
        else:
            r = seg.radius
            turn = ds / r  # signed
            heading = seg.start_heading + turn
            ccx = seg.start_x - r * math.sin(seg.start_heading)
            ccy = seg.start_y + r * math.cos(seg.start_heading)
            cx = ccx + r * math.sin(heading)
            cy = ccy - r * math.cos(heading)
            curvature = 1.0 / r
        return TrackSample(
            centerline_x=cx,
            centerline_y=cy,
            heading=_wrap(heading),
            curvature=curvature,
            width=self.width,
            s=s,
            lateral_offset=0.0,
        )

    def _segment_at(self, s: float) -> _ResolvedSegment:
        for seg in self._segments:
            if seg.s_start <= s < seg.s_end:
                return seg
        return self._segments[-1]

    def query(self, x: float, y: float) -> TrackSample:
        """Nearest-point projection, O(num_segments) -- each segment is
        projected onto analytically (closed-form for both line and circle),
        not by sampling arc-length in a loop. Segment count is small and
        fixed at track-build time, so this stays cheap even called every
        sensor read on every physics step."""
        best_s = None
        best_dist2 = math.inf
        for seg in self._segments:
            s_local, dist2 = _project_onto_segment(seg, x, y)
            if dist2 < best_dist2:
                best_dist2 = dist2
                best_s = seg.s_start + s_local
        assert best_s is not None
        best = self.sample_at_s(best_s)
        signed_lateral = _signed_lateral_offset(best, x, y)
        return TrackSample(
            centerline_x=best.centerline_x,
            centerline_y=best.centerline_y,
            heading=best.heading,
            curvature=best.curvature,
            width=best.width,
            s=best.s,
            lateral_offset=signed_lateral,
        )

    def is_on_track(self, x: float, y: float) -> bool:
        sample = self.query(x, y)
        return abs(sample.lateral_offset) <= sample.width / 2.0


def _project_onto_segment(seg: _ResolvedSegment, x: float, y: float) -> tuple[float, float]:
    """Return (local arc-length along seg, squared distance to that point)
    for the closest point on seg to (x, y), clamped to the segment's
    own extent (closure handles wraparound between segments)."""
    if seg.kind == "straight":
        dirx = math.cos(seg.start_heading)
        diry = math.sin(seg.start_heading)
        dx = x - seg.start_x
        dy = y - seg.start_y
        proj = dx * dirx + dy * diry
        s_local = _clamp(proj, 0.0, seg.length)
        px = seg.start_x + dirx * s_local
        py = seg.start_y + diry * s_local
        return s_local, (px - x) ** 2 + (py - y) ** 2
    else:
        r = abs(seg.radius)
        angle_to_point = math.atan2(y - seg.center_y, x - seg.center_x)
        start_angle = math.atan2(seg.start_y - seg.center_y, seg.start_x - seg.center_x)
        sweep = seg.length / r  # unsigned angular extent
        direction = 1.0 if seg.radius > 0 else -1.0
        delta = _wrap(direction * (angle_to_point - start_angle))
        delta_clamped = _clamp(delta, 0.0, sweep)
        point_angle = start_angle + direction * delta_clamped
        px = seg.center_x + r * math.cos(point_angle)
        py = seg.center_y + r * math.sin(point_angle)
        s_local = delta_clamped * r
        return s_local, (px - x) ** 2 + (py - y) ** 2


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _signed_lateral_offset(sample: TrackSample, x: float, y: float) -> float:
    dx = x - sample.centerline_x
    dy = y - sample.centerline_y
    nx = -math.sin(sample.heading)
    ny = math.cos(sample.heading)
    return dx * nx + dy * ny


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def make_simple_oval(
    straight_length: float = 80.0, turn_radius: float = 25.0, width: float = 10.0
) -> FixedLoopTrack:
    """v1 default track: an oval -- two straights, two 180-degree turns.
    Both turns curve the same direction (both left if driven CCW, both
    right if driven CW) -- any simple closed loop's turns must sum to a
    full 2*pi, so a two-turn loop can't mix directions. Constant-radius,
    no banking. A mixed left/right track is v1.1 backlog (needs either a
    closure-solved multi-arc geometry or a waypoint/spline track definition,
    not the simple Straight/Arc segment chain used here)."""
    segments: list[Segment] = [
        Straight(length=straight_length),
        Arc(radius=turn_radius, angle=math.pi),
        Straight(length=straight_length),
        Arc(radius=turn_radius, angle=math.pi),
    ]
    return FixedLoopTrack(segments, width=width)
