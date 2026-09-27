"""Phase 2: a spline-based procedural track generator (IMPLEMENTATION_PLAN.md
Phase 2, MicroRacer-style per Asperti & Del Brutto, arXiv:2203.10494).
Places control points around a base circle, jittered in radius and angle by
a seeded RNG, then builds a closed-loop SplineTrack through them.

"Controllable curvature, turn count, and width" (Phase 2's task list) map
onto this generator's knobs as: num_control_points is the turn-count proxy
(each control point is a potential turn), radius_jitter/angle_jitter control
curvature intensity, and width is a direct per-track parameter (constant
along one track, not varying within it -- variable-width sections were
explicitly out of scope for the diversity metric, see wayfinder ticket #13).

angle_jitter is clamped below 0.5 (as a fraction of the even angular
spacing) so perturbed control points can't reorder around the loop, which
keeps the generated spline from self-intersecting for reasonable jitter
values. This is a practical safeguard, not a proven guarantee -- there is
no explicit self-intersection detector here.
"""
from __future__ import annotations

import math
import random

from ars.track.spline_track import SplineTrack

_MAX_ANGLE_JITTER_FRACTION = 0.45


def generate_track(
    seed: int,
    num_control_points: int = 12,
    base_radius: float = 150.0,
    radius_jitter: float = 0.3,
    angle_jitter: float = 0.25,
    width: float = 10.0,
) -> SplineTrack:
    rng = random.Random(seed)
    angle_jitter = min(angle_jitter, _MAX_ANGLE_JITTER_FRACTION)
    even_step = 2.0 * math.pi / num_control_points

    points: list[tuple[float, float]] = []
    for i in range(num_control_points):
        base_angle = i * even_step
        angle = base_angle + rng.uniform(-angle_jitter, angle_jitter) * even_step
        radius = base_radius * (1.0 + rng.uniform(-radius_jitter, radius_jitter))
        points.append((radius * math.cos(angle), radius * math.sin(angle)))

    return SplineTrack(points, width=width)
