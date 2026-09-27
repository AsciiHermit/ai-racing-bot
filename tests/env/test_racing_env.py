import numpy as np

from ars.env import make_default_env
from ars.physics import KinematicBicyclePhysics


def test_reset_returns_obs_matching_space():
    env = make_default_env()
    obs, info = env.reset()
    assert env.observation_space.contains(obs)


def test_step_returns_expected_tuple_shapes():
    env = make_default_env()
    env.reset()
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert env.observation_space.contains(obs)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "progress_s" in info


def test_full_throttle_straight_makes_progress():
    env = make_default_env(off_track_terminates=False)
    env.reset()
    action = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    total_progress = 0.0
    for _ in range(100):
        _, reward, terminated, truncated, info = env.step(action)
        total_progress = info["progress_s"]
        if terminated or truncated:
            break
    assert total_progress > 0.0


def test_episode_truncates_at_max_steps():
    env = make_default_env(max_episode_steps=5, off_track_terminates=False)
    env.reset()
    action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    truncated = False
    for _ in range(10):
        _, _, terminated, truncated, _ = env.step(action)
        if truncated:
            break
    assert truncated


def test_lap_increments_on_wraparound():
    # Regression test: _progress_delta clamps to [-length/2, length/2], so
    # comparing its already-clamped output against length/2 (the original
    # bug) can never trigger -- lap count silently stayed 0 forever.
    # Pinned to the kinematic stub: this test is about the lap-counting
    # logic, not tire physics, and its fixed action was calibrated against
    # the stub's grip cap -- Phase 1's dynamic-bicycle model is intentionally
    # harder and this same fixed action no longer completes a lap under it
    # (see PHASE0_SPIKE_NOTE.md).
    env = make_default_env(
        physics=KinematicBicyclePhysics(), off_track_terminates=False, max_episode_steps=5000
    )
    env.reset()
    action = np.array([1.0, 0.0, 0.05], dtype=np.float32)  # mild turn, stays near centerline
    info = {}
    for _ in range(5000):
        _, _, _, _, info = env.step(action)
    assert info["lap"] >= 1


def test_gym_registration():
    import gymnasium as gym

    import ars.env  # noqa: F401 -- triggers registration

    env = gym.make("ARS-Racing-v0")
    obs, _ = env.reset()
    assert obs is not None
