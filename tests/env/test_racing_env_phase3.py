import numpy as np

from ars.env.factory import make_default_env
from ars.physics import KinematicBicyclePhysics
from ars.track import make_simple_oval


def _far_off_track_env(**kwargs):
    # Kinematic stub + zero steer drives straight ahead, quickly clearing
    # both the ordinary off-track margin and the wall margin.
    return make_default_env(physics=KinematicBicyclePhysics(), off_track_terminates=False, **kwargs)


def test_mild_off_track_is_not_a_collision():
    env = _far_off_track_env(wall_margin=50.0)
    env.reset()
    action = np.array([1.0, 0.0, 1.0], dtype=np.float32)  # hard turn, drifts off centerline
    collided = False
    for _ in range(50):
        _, _, _, _, info = env.step(action)
        collided = collided or info["collided"]
    assert not collided


def test_far_off_track_is_a_collision_with_extra_penalty():
    env = _far_off_track_env(wall_margin=1.0)
    env.reset()
    straight_action = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    for _ in range(30):
        env.step(straight_action)
    turn_action = np.array([1.0, 0.0, 1.0], dtype=np.float32)
    collided = False
    min_reward = 0.0
    for _ in range(100):
        _, reward, _, _, info = env.step(turn_action)
        collided = collided or info["collided"]
        min_reward = min(min_reward, reward)
    assert collided
    assert min_reward < -1.0  # harsher than the plain off-track penalty alone


def test_track_provider_swaps_track_on_reset():
    oval_a = make_simple_oval(straight_length=80.0, turn_radius=25.0)
    oval_b = make_simple_oval(straight_length=40.0, turn_radius=15.0)
    tracks = [oval_a, oval_b]
    calls = {"n": 0}

    def provider():
        track = tracks[calls["n"] % len(tracks)]
        calls["n"] += 1
        return track

    env = make_default_env(track_provider=provider)
    env.reset()
    first_length = env.track.length
    env.reset()
    second_length = env.track.length
    assert first_length != second_length


def test_track_provider_is_picklable_for_vectorized_rollout():
    import pickle

    from ars.env.multi_track import RandomTrackProvider

    provider = RandomTrackProvider([make_simple_oval(), make_simple_oval(turn_radius=15.0)], seed=0)
    restored = pickle.loads(pickle.dumps(provider))
    assert restored() is not None
