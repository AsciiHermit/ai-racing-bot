import math

from ars.track.extrapolation import generate_non_elliptical_track, generate_track_with_chicane
from ars.track.generator import generate_track
from ars.track.metrics import track_curvature_feature_vector


def _bounding_box(tracks, sample_spacing_m=1.0):
    vectors = [track_curvature_feature_vector(t, sample_spacing_m) for t in tracks]
    mins = [min(v[i] for v in vectors) for i in range(3)]
    maxs = [max(v[i] for v in vectors) for i in range(3)]
    return mins, maxs


def _is_extrapolation(vector, mins, maxs):
    return any(vector[i] < mins[i] or vector[i] > maxs[i] for i in range(3))


def test_non_elliptical_track_is_finite_and_closed():
    track = generate_non_elliptical_track(seed=3)
    assert track.length > 0
    start = track.sample_at_s(0.0)
    end = track.sample_at_s(track.length - 1e-6)
    assert math.hypot(end.centerline_x - start.centerline_x, end.centerline_y - start.centerline_y) < 5.0
    for i in range(200):
        sample = track.sample_at_s(i * track.length / 200)
        assert math.isfinite(sample.curvature)


def test_chicane_track_is_finite_and_closed():
    track = generate_track_with_chicane(seed=3)
    assert track.length > 0
    start = track.sample_at_s(0.0)
    end = track.sample_at_s(track.length - 1e-6)
    assert math.hypot(end.centerline_x - start.centerline_x, end.centerline_y - start.centerline_y) < 5.0
    for i in range(200):
        sample = track.sample_at_s(i * track.length / 200)
        assert math.isfinite(sample.curvature)


def test_chicane_pushes_min_radius_outside_the_baseline_train_range():
    train = [generate_track(seed=s) for s in range(20)]
    mins, maxs = _bounding_box(train)

    outside_count = 0
    for s in range(100, 115):
        candidate = generate_track_with_chicane(seed=s)
        vector = track_curvature_feature_vector(candidate)
        if _is_extrapolation(vector, mins, maxs):
            outside_count += 1
    # Not every seed needs to land outside, but the technique should work
    # for most of them -- this is the actual validation ticket #13 deferred
    # to Phase 2 implementation.
    assert outside_count >= 10


def test_non_elliptical_pushes_mean_or_variance_outside_the_baseline_train_range():
    train = [generate_track(seed=s) for s in range(20)]
    mins, maxs = _bounding_box(train)

    outside_count = 0
    for s in range(200, 215):
        candidate = generate_non_elliptical_track(seed=s)
        vector = track_curvature_feature_vector(candidate)
        if _is_extrapolation(vector, mins, maxs):
            outside_count += 1
    assert outside_count >= 10
