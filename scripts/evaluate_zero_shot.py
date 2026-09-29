"""Phase 5's zero-shot evaluation: run a trained model, deterministically,
against every track in a held-out tier of the manifest, and report lap
completion rate plus the gap to Phase 4's per-track reference lap time.

Run: python scripts/evaluate_zero_shot.py --model runs/ppo_phase5_stepA_final.zip --algo ppo --tier interpolation
"""
from __future__ import annotations

import argparse
import statistics

from stable_baselines3 import PPO, SAC, TD3

from ars.env.baseline_factory import make_baseline_env
from ars.track.manifest import build_track, load_manifest

ALGOS = {"ppo": PPO, "sac": SAC, "td3": TD3}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--algo", choices=sorted(ALGOS), required=True)
    parser.add_argument("--tier", choices=["train", "interpolation", "extrapolation"], default="interpolation")
    parser.add_argument("--manifest", default="data/tracks/manifest.json")
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument(
        "--terminate-off-track",
        action="store_true",
        help="stop each episode at the first off-track step, so progress = distance driven before leaving the track",
    )
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    specs = manifest["train"] if args.tier == "train" else manifest["held_out"][args.tier]
    model = ALGOS[args.algo].load(args.model)

    completed = 0
    lap_times = []
    gaps = []
    progress_fractions = []
    for spec in specs:
        track = build_track(spec)
        env = make_baseline_env(
            track=track, off_track_terminates=args.terminate_off_track, max_episode_steps=args.max_steps
        )
        obs, _ = env.reset()
        lap_time = None
        info = {"lap": 0, "progress_s": 0.0}
        for step in range(args.max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            if info["lap"] >= 1 and lap_time is None:
                lap_time = (step + 1) * env.dt
            if terminated or truncated:
                break
        progress_fractions.append((info["lap"] * track.length + info["progress_s"]) / track.length)
        if lap_time is not None:
            completed += 1
            lap_times.append(lap_time)
            reference = spec.get("reference_lap_time_s")
            if reference:
                gaps.append((lap_time - reference) / reference)

    n = len(specs)
    print(f"tier={args.tier} completed={completed}/{n} ({100 * completed / n:.0f}%)")
    print(
        f"progress at episode end: mean={statistics.mean(progress_fractions) * 100:.1f}% of a lap "
        f"(min={min(progress_fractions) * 100:.1f}%, max={max(progress_fractions) * 100:.1f}%)"
    )
    if lap_times:
        print(
            f"lap_time: mean={statistics.mean(lap_times):.2f}s "
            f"min={min(lap_times):.2f}s max={max(lap_times):.2f}s"
        )
    if gaps:
        print(
            f"generalization gap vs Phase 4 reference: mean={statistics.mean(gaps) * 100:.1f}% "
            "(positive = slower than the classical optimum)"
        )
    else:
        print("no completed episodes had a reference lap time to compare against" if lap_times else "")


if __name__ == "__main__":
    main()
