"""Phase 5 Step A (IMPLEMENTATION_PLAN.md Phase 5, "lower risk, do first"):
domain-randomized model-free RL. Extends Phase 3's baseline training with
a pool of freshly-generated, randomized tracks (distinct from the Phase 2
manifest's fixed train set) and wayfinder ticket #10's regret-proxy
curriculum -- oversampling tracks the agent is currently doing worse on
relative to its own rolling-best, not a fixed or uniform distribution.

Runs single-process (n_envs=1, DummyVecEnv), not vectorized like Phase 3's
training: the regret curriculum needs a live per-track rolling-best/regret
state that's simplest to keep as one in-process Python object feeding
episode returns back into itself via SB3's Monitor wrapper + a callback,
rather than coordinating that state across SubprocVecEnv's worker
processes. A real scale-up would need that coordination; this is the
"lower risk, do first" version, not the eventual production one.

Run: python scripts/train_phase5_step_a.py --timesteps 300000
"""
from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env

from ars.env.baseline_factory import make_baseline_env
from ars.env.curriculum import RegretCurriculumTrackProvider
from ars.track.generator import generate_track

# Domain-randomization pool: fresh tracks, disjoint seed range from the
# Phase 2 manifest's train set (seeds 0-79) and held-out set (10000+,
# 20000+), so this pool is genuinely separate track instances, not a
# re-sample of the manifest.
_POOL_SEED_START = 30_000
_POOL_SIZE = 50


class _RecordEpisodeReturnCallback(BaseCallback):
    """Feeds each finished episode's return into the curriculum provider,
    reading it off the info dict SB3's Monitor wrapper already populates
    (info["episode"]["r"]) -- no change to RacingEnv needed."""

    def __init__(self, provider: RegretCurriculumTrackProvider):
        super().__init__()
        self.provider = provider

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            episode = info.get("episode")
            if episode is not None:
                self.provider.record_episode(float(episode["r"]))
        return True


def _make_pool() -> list:
    return [generate_track(seed=_POOL_SEED_START + i) for i in range(_POOL_SIZE)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--refresh-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-root", default="runs")
    args = parser.parse_args()

    pool = _make_pool()
    provider = RegretCurriculumTrackProvider(pool, refresh_every=args.refresh_every, seed=args.seed)

    env_fn = lambda: make_baseline_env(track_provider=provider, off_track_terminates=False)
    vec_env = make_vec_env(env_fn, n_envs=1, seed=args.seed)

    log_root = Path(args.log_root)
    log_root.mkdir(parents=True, exist_ok=True)
    run_name = "ppo_phase5_stepA"

    model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log=str(log_root), seed=args.seed)
    model.learn(total_timesteps=args.timesteps, tb_log_name=run_name, callback=_RecordEpisodeReturnCallback(provider))

    out_path = log_root / f"{run_name}_final"
    model.save(str(out_path))
    print(f"saved model to {out_path}.zip")
    print(f"final curriculum weights: {provider.weights}")

    vec_env.close()


if __name__ == "__main__":
    main()
