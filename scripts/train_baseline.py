"""Phase 3: train a PPO/SAC/TD3 baseline (IMPLEMENTATION_PLAN.md Phase 3).

Step A -- single-track sanity check, the fixed oval only:
    python scripts/train_baseline.py --algo ppo --step A --timesteps 100000

Step B -- in-distribution training across the full Phase 2 train set
(data/tracks/manifest.json's 80 train tracks, not per-track):
    python scripts/train_baseline.py --algo sac --step B --timesteps 100000

Logs to TensorBoard under runs/<algo>_step<step>/ (`tensorboard --logdir runs`).
Saves the final model to runs/<algo>_step<step>_final.zip.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from ars.env.baseline_factory import make_baseline_env
from ars.env.multi_track import RandomTrackProvider
from ars.track.manifest import load_manifest, tracks_from_specs

ALGOS = {"ppo": PPO, "sac": SAC, "td3": TD3}


class _StepAEnvFn:
    """Picklable env constructor for Step A (fixed oval) -- a plain
    closure risks not surviving SubprocVecEnv's worker-process handoff on
    every platform, a class instance reliably does (same reasoning as
    ars.env.multi_track.RandomTrackProvider)."""

    def __call__(self):
        return make_baseline_env(off_track_terminates=False)


class _StepBEnvFn:
    """Picklable env constructor for Step B (the full train set)."""

    def __init__(self, manifest_path: str, seed: int):
        self.manifest_path = manifest_path
        self.seed = seed

    def __call__(self):
        manifest = load_manifest(self.manifest_path)
        tracks = tracks_from_specs(manifest["train"])
        provider = RandomTrackProvider(tracks, seed=self.seed)
        return make_baseline_env(track_provider=provider, off_track_terminates=False)


def make_env_fn(step: str, seed: int, manifest_path: str):
    return _StepAEnvFn() if step == "A" else _StepBEnvFn(manifest_path, seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=sorted(ALGOS), required=True)
    parser.add_argument("--step", choices=["A", "B"], required=True)
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--manifest", default="data/tracks/manifest.json")
    parser.add_argument("--log-root", default="runs")
    args = parser.parse_args()

    env_fn = make_env_fn(args.step, args.seed, args.manifest)
    vec_env = make_vec_env(env_fn, n_envs=args.n_envs, seed=args.seed, vec_env_cls=SubprocVecEnv)

    run_name = f"{args.algo}_step{args.step}"
    log_root = Path(args.log_root)
    log_root.mkdir(parents=True, exist_ok=True)

    algo_cls = ALGOS[args.algo]
    model = algo_cls("MlpPolicy", vec_env, verbose=1, tensorboard_log=str(log_root), seed=args.seed)
    model.learn(total_timesteps=args.timesteps, tb_log_name=run_name)

    out_path = log_root / f"{run_name}_final"
    model.save(str(out_path))
    print(f"saved model to {out_path}.zip")

    vec_env.close()


if __name__ == "__main__":
    main()
