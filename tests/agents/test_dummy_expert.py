import numpy as np

from ars.agents import DummyExpertAgent
from ars.env import make_default_env


def test_expert_stays_on_track_for_a_lap():
    env = make_default_env(off_track_terminates=False, max_episode_steps=5000)
    env.reset()
    expert = DummyExpertAgent(env.track)

    max_offset = 0.0
    for _ in range(5000):
        action = expert.act(env.vehicle_state)
        env.step(action)
        sample = env.track.query(env.vehicle_state.x, env.vehicle_state.y)
        max_offset = max(max_offset, abs(sample.lateral_offset))

    assert max_offset < env.track.width / 2


def test_expert_completes_at_least_one_lap():
    env = make_default_env(off_track_terminates=False, max_episode_steps=5000)
    env.reset()
    expert = DummyExpertAgent(env.track)

    info = {}
    for _ in range(5000):
        action = expert.act(env.vehicle_state)
        _, _, _, _, info = env.step(action)

    assert info["lap"] >= 1


def test_expert_action_is_within_action_space():
    env = make_default_env()
    env.reset()
    expert = DummyExpertAgent(env.track)
    action = expert.act(env.vehicle_state)
    assert env.action_space.contains(action.astype(np.float32))
