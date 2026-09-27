"""Phase 3 task: verify ARS-Racing-v0 (via the baseline factory) wraps
cleanly for vectorized rollout with SB3's SubprocVecEnv -- this needs the
env, physics, track, and sensors to all be picklable, since SubprocVecEnv
sends env-constructor closures to worker processes.

Run: python scripts/verify_vectorized_rollout.py
"""
from __future__ import annotations

import numpy as np
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from ars.env.baseline_factory import make_baseline_env


def main() -> None:
    num_envs = 4
    vec_env = make_vec_env(make_baseline_env, n_envs=num_envs, vec_env_cls=SubprocVecEnv)

    obs = vec_env.reset()
    print(f"reset obs shape: {obs.shape}  (expected ({num_envs}, obs_dim))")

    actions = np.array([vec_env.action_space.sample() for _ in range(num_envs)])
    obs, rewards, dones, infos = vec_env.step(actions)
    print(f"step obs shape: {obs.shape}, rewards shape: {rewards.shape}, dones shape: {dones.shape}")
    print(f"infos[0] keys: {sorted(infos[0].keys())}")

    for _ in range(50):
        actions = np.array([vec_env.action_space.sample() for _ in range(num_envs)])
        obs, rewards, dones, infos = vec_env.step(actions)

    vec_env.close()
    print("SubprocVecEnv rollout OK: reset/step/close all completed across worker processes.")


if __name__ == "__main__":
    main()
