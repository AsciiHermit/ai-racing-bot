"""Manual end-to-end smoke test: random agent drives the default env for
one episode and prints a summary. Not a pytest test -- run directly to
eyeball that the full stack (physics + track + sensors + env) behaves.

    python scripts/smoke_test.py
"""
from __future__ import annotations

from ars.agents import RandomAgent
from ars.env import make_default_env


def main() -> None:
    env = make_default_env(max_episode_steps=500)
    agent = RandomAgent(env.action_space, seed=0)

    obs, info = env.reset()
    print(f"obs shape: {obs.shape}, action shape: {env.action_space.shape}")

    total_reward = 0.0
    steps = 0
    for steps in range(1, env.max_episode_steps + 1):
        action = agent.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break

    print(f"episode ended after {steps} steps")
    print(f"total_reward={total_reward:.2f}")
    print(f"final progress_s={info['progress_s']:.2f} / track_length={env.track.length:.2f}")
    print(f"laps={info['lap']}, off_track={info['off_track']}")


if __name__ == "__main__":
    main()
