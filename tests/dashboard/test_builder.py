import numpy as np

from ars.dashboard.builder import build_env_and_agent
from ars.dashboard.config import SessionConfig


def test_build_env_and_agent_runs_one_step():
    config = SessionConfig()
    env, agent = build_env_and_agent(config)
    env.reset()
    action = agent.act(env.vehicle_state)
    obs, reward, terminated, truncated, info = env.step(action)
    assert env.observation_space.contains(obs)


def test_lidar_range_from_config_is_applied():
    config = SessionConfig()
    config.lidar.max_range_m = 15.0
    env, _ = build_env_and_agent(config)
    lidar_sensor = [s for s in env.sensors if s.name == "lidar"][0]
    assert lidar_sensor.max_range == 15.0
