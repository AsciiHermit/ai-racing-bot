"""Phase 2's checked-in seed manifest: a reproducible train set plus a
30-track held-out set (15 interpolation + 15 extrapolation, per wayfinder
ticket #12), classified by the bounding-box rule against the train set's
own ticket-#11 feature vectors. Extrapolation tracks alternate between
ticket #13's two techniques (chicane insertion, non-elliptical topology).

A "spec" is a small, serializable dict (generator name + seed + params) --
tracks are regenerated from specs via build_track(), never stored as raw
control points. This is what scripts/generate_track_manifest.py writes to
data/tracks/manifest.json; that file is the actual checked-in manifest,
this module is how it's built and read back.
"""
from __future__ import annotations

from ars.track.extrapolation import generate_non_elliptical_track, generate_track_with_chicane
from ars.track.generator import generate_track
from ars.track.metrics import track_curvature_feature_vector
from ars.track.spline_track import SplineTrack

_GENERATORS = {
    "generate_track": generate_track,
    "generate_non_elliptical_track": generate_non_elliptical_track,
    "generate_track_with_chicane": generate_track_with_chicane,
}

_EXTRAPOLATION_GENERATOR_CYCLE = ["generate_track_with_chicane", "generate_non_elliptical_track"]


def build_track(spec: dict) -> SplineTrack:
    fn = _GENERATORS[spec["generator"]]
    return fn(seed=spec["seed"], **spec.get("params", {}))


def _make_spec(id_: str, generator: str, seed: int, tier: str, feature_vector: tuple[float, float, float]) -> dict:
    return {
        "id": id_,
        "generator": generator,
        "seed": seed,
        "params": {},
        "tier": tier,
        "feature_vector": list(feature_vector),
    }


def build_manifest(
    num_train: int = 80,
    num_interpolation: int = 15,
    num_extrapolation: int = 15,
    train_seed_start: int = 0,
    interpolation_seed_start: int = 10_000,
    extrapolation_seed_start: int = 20_000,
) -> dict:
    train_specs = []
    for i in range(num_train):
        seed = train_seed_start + i
        vector = track_curvature_feature_vector(generate_track(seed=seed))
        train_specs.append(_make_spec(f"train_{i:03d}", "generate_track", seed, "train", vector))

    mins = [min(spec["feature_vector"][k] for spec in train_specs) for k in range(3)]
    maxs = [max(spec["feature_vector"][k] for spec in train_specs) for k in range(3)]

    interpolation_specs = []
    seed = interpolation_seed_start
    while len(interpolation_specs) < num_interpolation:
        vector = track_curvature_feature_vector(generate_track(seed=seed))
        if all(mins[k] <= vector[k] <= maxs[k] for k in range(3)):
            interpolation_specs.append(
                _make_spec(f"interp_{len(interpolation_specs):03d}", "generate_track", seed, "interpolation", vector)
            )
        seed += 1

    extrapolation_specs = []
    seed = extrapolation_seed_start
    cycle_idx = 0
    while len(extrapolation_specs) < num_extrapolation:
        generator_name = _EXTRAPOLATION_GENERATOR_CYCLE[cycle_idx % len(_EXTRAPOLATION_GENERATOR_CYCLE)]
        vector = track_curvature_feature_vector(_GENERATORS[generator_name](seed=seed))
        if any(vector[k] < mins[k] or vector[k] > maxs[k] for k in range(3)):
            extrapolation_specs.append(
                _make_spec(f"extrap_{len(extrapolation_specs):03d}", generator_name, seed, "extrapolation", vector)
            )
        seed += 1
        cycle_idx += 1

    all_specs = train_specs + interpolation_specs + extrapolation_specs
    diversity_check = _check_diversity(all_specs, mins, maxs)

    return {
        "bounding_box": {"mins": mins, "maxs": maxs},
        "train": train_specs,
        "held_out": {"interpolation": interpolation_specs, "extrapolation": extrapolation_specs},
        "diversity_check": diversity_check,
    }


def _check_diversity(specs: list[dict], mins: list[float], maxs: list[float], min_distance: float = 1e-3) -> dict:
    """Basic diversity check (Phase 2's exit criterion): no two tracks in
    the whole set (train + held-out) have near-identical feature vectors.
    Distances are normalized per-axis by the train set's own range, since
    the three axes have very different natural scales."""
    ranges = [(maxs[k] - mins[k]) or 1.0 for k in range(3)]

    def normalized(vector: list[float]) -> tuple[float, float, float]:
        return tuple((vector[k] - mins[k]) / ranges[k] for k in range(3))

    normalized_vectors = [normalized(spec["feature_vector"]) for spec in specs]
    min_pairwise_distance = float("inf")
    for i in range(len(normalized_vectors)):
        for j in range(i + 1, len(normalized_vectors)):
            a, b = normalized_vectors[i], normalized_vectors[j]
            distance = sum((a[k] - b[k]) ** 2 for k in range(3)) ** 0.5
            min_pairwise_distance = min(min_pairwise_distance, distance)

    return {
        "passed": min_pairwise_distance > min_distance,
        "min_pairwise_distance": min_pairwise_distance,
    }
