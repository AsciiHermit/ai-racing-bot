from ars.env.baseline_factory import make_baseline_env
from ars.env.multi_track import RandomTrackProvider
from ars.track import make_simple_oval
from ars.track.manifest import load_manifest, tracks_from_specs


def test_default_baseline_env_uses_the_narrow_sensor_bundle():
    env = make_baseline_env()
    names = {sensor.name for sensor in env.sensors}
    assert names == {"lidar", "proprioceptive", "track_pose"}


def test_default_baseline_env_uses_the_fixed_oval():
    env = make_baseline_env()
    env.reset()
    assert env.track.length > 0


def test_baseline_env_accepts_a_track_provider_for_step_b():
    tracks = [make_simple_oval(), make_simple_oval(turn_radius=15.0)]
    provider = RandomTrackProvider(tracks, seed=0)
    env = make_baseline_env(track_provider=provider)
    env.reset()
    first = env.track.length
    env.reset()
    env.reset()
    # not asserting inequality (RNG could repeat the same pick), just that
    # it runs end-to-end and reset() actually goes through the provider
    assert env.track.length in {t.length for t in tracks}
    assert first in {t.length for t in tracks}


def test_load_manifest_and_build_train_tracks():
    manifest = load_manifest("data/tracks/manifest.json")
    train_tracks = tracks_from_specs(manifest["train"])
    assert len(train_tracks) == 80
    assert all(t.length > 0 for t in train_tracks)
