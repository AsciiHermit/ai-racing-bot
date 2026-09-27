"""Evaluate a saved Phase 3 baseline model: does it actually complete laps?

Run: python scripts/evaluate_baseline.py --model runs/ppo_stepA_final.zip --algo ppo
"""
from __future__ import annotations

import argparse

from stable_baselines3 import PPO, SAC, TD3

from ars.env.baseline_factory import make_baseline_env
from ars.env.multi_track import RandomTrackProvider
from ars.track.manifest import load_manifest, tracks_from_specs

ALGOS = {"ppo": PPO, "sac": SAC, "td3": TD3}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--algo", choices=sorted(ALGOS), required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--track-set", choices=["oval", "train"], default="oval")
    parser.add_argument("--manifest", default="data/tracks/manifest.json")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    model = ALGOS[args.algo].load(args.model)
    if args.track_set == "oval":
        env = make_baseline_env(off_track_terminates=False, max_episode_steps=args.max_steps)
    else:
        manifest = load_manifest(args.manifest)
        tracks = tracks_from_specs(manifest["train"])
        provider = RandomTrackProvider(tracks, seed=args.seed)
        env = make_baseline_env(
            track_provider=provider, off_track_terminates=False, max_episode_steps=args.max_steps
        )

    for ep in range(args.episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        max_lap = 0
        off_track_steps = 0
        collided_steps = 0
        for _ in range(args.max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            max_lap = max(max_lap, info["lap"])
            off_track_steps += int(info["off_track"])
            collided_steps += int(info["collided"])
            if terminated or truncated:
                break
        print(
            f"episode {ep}: track_length={env.track.length:.1f} total_reward={total_reward:.1f} "
            f"laps={max_lap} off_track_steps={off_track_steps} collided_steps={collided_steps}"
        )


if __name__ == "__main__":
    main()
