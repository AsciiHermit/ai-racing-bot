from ars.track.manifest import build_manifest, build_track


def test_build_manifest_produces_requested_counts():
    manifest = build_manifest(num_train=5, num_interpolation=2, num_extrapolation=2)
    assert len(manifest["train"]) == 5
    assert len(manifest["held_out"]["interpolation"]) == 2
    assert len(manifest["held_out"]["extrapolation"]) == 2


def test_interpolation_tracks_fall_inside_the_train_bounding_box():
    manifest = build_manifest(num_train=8, num_interpolation=3, num_extrapolation=2)
    mins = manifest["bounding_box"]["mins"]
    maxs = manifest["bounding_box"]["maxs"]
    for spec in manifest["held_out"]["interpolation"]:
        vector = spec["feature_vector"]
        assert all(mins[i] <= vector[i] <= maxs[i] for i in range(3))


def test_extrapolation_tracks_fall_outside_the_train_bounding_box_on_some_axis():
    manifest = build_manifest(num_train=8, num_interpolation=2, num_extrapolation=3)
    mins = manifest["bounding_box"]["mins"]
    maxs = manifest["bounding_box"]["maxs"]
    for spec in manifest["held_out"]["extrapolation"]:
        vector = spec["feature_vector"]
        assert any(vector[i] < mins[i] or vector[i] > maxs[i] for i in range(3))


def test_diversity_check_passes_with_distinct_seeds():
    manifest = build_manifest(num_train=8, num_interpolation=3, num_extrapolation=3)
    assert manifest["diversity_check"]["passed"] is True
    assert manifest["diversity_check"]["min_pairwise_distance"] > 0.0


def test_every_spec_round_trips_through_build_track():
    manifest = build_manifest(num_train=3, num_interpolation=1, num_extrapolation=1)
    all_specs = (
        manifest["train"] + manifest["held_out"]["interpolation"] + manifest["held_out"]["extrapolation"]
    )
    for spec in all_specs:
        track = build_track(spec)
        assert track.length > 0


def test_manifest_is_deterministic():
    a = build_manifest(num_train=5, num_interpolation=2, num_extrapolation=2)
    b = build_manifest(num_train=5, num_interpolation=2, num_extrapolation=2)
    assert a["train"] == b["train"]
    assert a["held_out"] == b["held_out"]
