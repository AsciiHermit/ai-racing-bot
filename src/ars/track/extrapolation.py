"""Extrapolation-track generation (wayfinder ticket #13,
github.com/AsciiHermit/ai-racing-bot/issues/13): two techniques, chosen
because the diversity metric (ticket #11) is curvature-only, so each
targets a different axis of that same metric rather than track width,
which the metric can't see.

- generate_non_elliptical_track: a low-frequency radius harmonic reshapes
  the whole loop (targets mean/variance -- broad curvature-profile change).
- generate_track_with_chicane: inserts a sharp local zig-zag into an
  otherwise-normal generated track (targets min-radius -- a single tight
  spike, not a broad reshaping).

Validated directly in tests/track/test_extrapolation.py against a baseline
train-set bounding box, per ticket #13's own resolution ("actually building
and validating these... is deferred to Phase 2 implementation").
"""
from __future__ import annotations

import math
import random

from ars.track.generator import _MAX_ANGLE_JITTER_FRACTION, generate_track
from ars.track.spline_track import SplineTrack


def generate_non_elliptical_track(
    seed: int,
    num_control_points: int = 12,
    base_radius: float = 150.0,
    radius_jitter: float = 0.3,
    angle_jitter: float = 0.25,
    width: float = 10.0,
    harmonic_amplitude: float = 0.8,
    harmonic_frequency: int = 2,
) -> SplineTrack:
    """Same jittered-control-point scheme as generate_track, but the base
    radius is modulated by a low-frequency sinusoid before jitter is
    applied -- a peanut/clover-like topology, not a jittered circle."""
    rng = random.Random(seed)
    angle_jitter = min(angle_jitter, _MAX_ANGLE_JITTER_FRACTION)
    even_step = 2.0 * math.pi / num_control_points

    points: list[tuple[float, float]] = []
    for i in range(num_control_points):
        base_angle = i * even_step
        angle = base_angle + rng.uniform(-angle_jitter, angle_jitter) * even_step
        harmonic = 1.0 + harmonic_amplitude * math.sin(harmonic_frequency * angle)
        radius = base_radius * harmonic * (1.0 + rng.uniform(-radius_jitter, radius_jitter))
        points.append((radius * math.cos(angle), radius * math.sin(angle)))

    return SplineTrack(points, width=width)


def generate_track_with_chicane(
    seed: int,
    num_control_points: int = 12,
    base_radius: float = 150.0,
    radius_jitter: float = 0.3,
    angle_jitter: float = 0.25,
    width: float = 10.0,
    chicane_offset: float = 25.0,
) -> SplineTrack:
    """Take an otherwise-normal generated track and insert two extra
    control points near one existing point, offset perpendicular to the
    local path in alternating directions -- a sharp S the spline must pass
    through, spiking local curvature far more than any single control
    point in the plain generator would."""
    base = generate_track(seed, num_control_points, base_radius, radius_jitter, angle_jitter, width)
    points = base.control_points()
    n = len(points)

    rng = random.Random(seed)
    i = rng.randrange(n)
    p_prev = points[(i - 1) % n]
    p_curr = points[i]
    p_next = points[(i + 1) % n]

    tangent_x = p_next[0] - p_prev[0]
    tangent_y = p_next[1] - p_prev[1]
    norm = math.hypot(tangent_x, tangent_y) or 1.0
    perp_x, perp_y = -tangent_y / norm, tangent_x / norm

    left = (p_curr[0] + perp_x * chicane_offset, p_curr[1] + perp_y * chicane_offset)
    right = (p_curr[0] - perp_x * chicane_offset, p_curr[1] - perp_y * chicane_offset)

    new_points = points[:i] + [left, right] + points[i:]
    return SplineTrack(new_points, width=width)
