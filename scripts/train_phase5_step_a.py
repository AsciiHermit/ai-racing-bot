"""Phase 5 Step A (IMPLEMENTATION_PLAN.md Phase 5, "lower risk, do first"):
domain-randomized model-free RL. Extends Phase 3's baseline training with
a pool of freshly-generated, randomized tracks (distinct from the Phase 2
manifest's fixed train set) and wayfinder ticket #10's regret-proxy
curriculum -- oversampling tracks the agent is currently doing worse on
relative to its own rolling-best, not a fixed or uniform distribution.

Runs in one process (DummyVecEnv, any --n-envs), not SubprocVecEnv like
Phase 3's training: the regret curriculum needs a live per-track
rolling-best/regret state that's simplest to keep as one in-process Python
object shared by reference across the envs, fed by SB3's Monitor wrapper +
a callback, rather than coordinated across worker processes. A real
scale-up would need that coordination; this is the "lower risk, do first"
version, not the eventual production one.

Reward-scale controls (--reward-norm, --terminate-off-track) exist to test
whether the unbounded, unnormalized return (an agent stuck off-track earns
about -6/step for the rest of a 2000-step episode) is what makes PPO
diverge here; they only affect this script, not RacingEnv's defaults.

Run: python scripts/train_phase5_step_a.py --timesteps 300000
"""
from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from ars.env.baseline_factory import make_baseline_env
from ars.env.curriculum import RegretCurriculumTrackProvider
from ars.env.multi_track import RandomTrackProvider
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
    (info["episode"]["r"]) -- no change to RacingEnv needed.

    Each env's current track is tracked here, because with several envs
    sharing one provider the provider's "last sampled" track belongs to
    whichever env reset most recently. By the time this callback sees a
    finished episode the env has already auto-reset onto its next track, so
    the finished episode's index is the one remembered from the previous
    step, and the env's new track is read back afterwards."""

    def __init__(self, provider: RegretCurriculumTrackProvider):
        super().__init__()
        self.provider = provider
        self._current_index: list[int] = []

    def _on_training_start(self) -> None:
        self._current_index = [self.provider.index_of(t) for t in self.training_env.get_attr("track")]

    def _on_step(self) -> bool:
        for env_i, info in enumerate(self.locals["infos"]):
            episode = info.get("episode")
            if episode is not None:
                self.provider.record_episode(float(episode["r"]), track_index=self._current_index[env_i])
                new_track = self.training_env.get_attr("track", indices=[env_i])[0]
                self._current_index[env_i] = self.provider.index_of(new_track)
        return True


def _make_pool() -> list:
    return [generate_track(seed=_POOL_SEED_START + i) for i in range(_POOL_SIZE)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--refresh-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-root", default="runs")
    parser.add_argument(
        "--curriculum",
        choices=["regret", "uniform"],
        default="regret",
        help="'uniform' is an ablation baseline (plain random sampling, no regret weighting)",
    )
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--n-envs",
        type=int,
        default=1,
        help="DummyVecEnv (single-process, default) -- multiple envs still share one "
        "curriculum object by reference, no cross-process coordination needed",
    )
    parser.add_argument(
        "--reward-norm",
        action="store_true",
        help="VecNormalize on rewards only (observations stay raw, so evaluation scripts need no changes)",
    )
    parser.add_argument(
        "--terminate-off-track",
        action="store_true",
        help="end the episode on the first off-track step instead of letting the penalty accumulate",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="save a checkpoint every N total timesteps (0 = off) so an interrupted run stays evaluable",
    )
    args = parser.parse_args()

    pool = _make_pool()
    callbacks: list[BaseCallback] = []
    if args.curriculum == "regret":
        provider = RegretCurriculumTrackProvider(pool, refresh_every=args.refresh_every, seed=args.seed)
        callbacks.append(_RecordEpisodeReturnCallback(provider))
    else:
        provider = RandomTrackProvider(pool, seed=args.seed)

    env_fn = lambda: make_baseline_env(track_provider=provider, off_track_terminates=args.terminate_off_track)
    vec_env = make_vec_env(env_fn, n_envs=args.n_envs, seed=args.seed)
    if args.reward_norm:
        # Episode returns for the curriculum come from the Monitor wrapper
        # inside each env, so they stay in raw reward units.
        vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True)

    log_root = Path(args.log_root)
    log_root.mkdir(parents=True, exist_ok=True)
    run_name = args.run_name or f"ppo_phase5_stepA_{args.curriculum}"

    if args.checkpoint_every > 0:
        # save_freq counts calls per env, i.e. total timesteps / n_envs.
        callbacks.append(
            CheckpointCallback(
                save_freq=max(1, args.checkpoint_every // args.n_envs),
                save_path=str(log_root),
                name_prefix=run_name,
            )
        )
    callback = CallbackList(callbacks) if callbacks else None

    model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log=str(log_root), seed=args.seed)
    model.learn(total_timesteps=args.timesteps, tb_log_name=run_name, callback=callback)

    out_path = log_root / f"{run_name}_final"
    model.save(str(out_path))
    print(f"saved model to {out_path}.zip")
    if args.curriculum == "regret":
        print(f"final curriculum weights: {provider.weights}")

    vec_env.close()


if __name__ == "__main__":
    main()
