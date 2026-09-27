"""Track diversity/difficulty metric (wayfinder ticket #11,
github.com/AsciiHermit/ai-racing-bot/issues/11): a 3-element feature vector
-- mean curvature, curvature variance, minimum radius -- computed from a
Track's own analytic curvature at fixed arc-length intervals along the
centerline. Feeds tickets #12 (three-tier held-out boundaries) and #13
(extrapolation-track generation).

Works against any Track implementation (FixedLoopTrack, the spline-based
generator), not just one concrete type -- it only calls sample_at_s(), the
same Protocol method every Track already implements.
"""
from __future__ import annotations

import math

from ars.core.interfaces import Track


def track_curvature_feature_vector(track: Track, sample_spacing_m: float = 1.0) -> tuple[float, float, float]:
    """Returns (mean_curvature, curvature_variance, min_radius).

    Fixed arc-length spacing (not a fixed sample count) keeps mean/variance
    per-unit-length and comparable across tracks of different total
    lengths, per ticket #11's resolution.
    """
    if sample_spacing_m <= 0.0:
        raise ValueError("sample_spacing_m must be positive")

    num_samples = max(1, round(track.length / sample_spacing_m))
    curvatures = [track.sample_at_s(i * track.length / num_samples).curvature for i in range(num_samples)]

    mean_curvature = sum(abs(c) for c in curvatures) / num_samples
    variance = sum((abs(c) - mean_curvature) ** 2 for c in curvatures) / num_samples

    max_abs_curvature = max(abs(c) for c in curvatures)
    min_radius = 1.0 / max_abs_curvature if max_abs_curvature > 1e-12 else math.inf

    return mean_curvature, variance, min_radius
