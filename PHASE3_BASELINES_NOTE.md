# Phase 3 Results: PPO/SAC/TD3 Baselines

Per `IMPLEMENTATION_PLAN.md` Phase 3's exit criterion: "PPO/SAC/TD3 all complete laps reliably on the fixed oval (Step A) and show reasonable in-distribution performance across the train set (Step B), with training curves logged."

## Setup

- **Observation bundle**: LiDAR (1 ray) + proprioceptive + track-relative pose, per the plan's default (`ars/env/baseline_factory.py`), narrower than `make_default_env`'s fuller GPS+IMU bundle.
- **Physics**: Phase 1's dynamic bicycle model (default).
- **Reward**: track-progress-based, with off-track (−1/step) and wall-collision (−5/step) penalties (`ars/env/racing_env.py`).
- **Step A**: the single fixed oval. **Step B**: sampled per-episode from the 80-track train manifest (`ars/env/multi_track.py`, `data/tracks/manifest.json`).
- **Training**: Stable-Baselines3, 300,000 timesteps per run, 2 vectorized envs (`SubprocVecEnv`), seed 0, logged to TensorBoard under `runs/` (gitignored — regenerate with `scripts/train_baseline.py`).
- **Environment note**: run in a project-local `.venv` (Python 3.14, SB3 2.9.0, torch 2.14.0+cpu) to avoid touching the system Python environment; `pygame`/viz extras remain unavailable here (no prebuilt wheel for 3.13/3.14), consistent with every prior phase's note.

## A bug found and fixed along the way

Evaluating the trained SAC Step A model surfaced a real, pre-existing bug: a near-stationary vehicle sitting exactly at the track's start/finish seam had its nearest-point projection flicker between `s≈0` and `s≈track_length` from sub-millimeter position noise alone, and the old lap-counting logic (`_crossed_finish_line_forward`, a discrete "did raw s jump by ~track_length this step" check) misfired on every flicker — one run reported 81 laps in 40 simulated seconds, physically impossible at the vehicle's ~95 m/s cap.

Fixed by tracking **cumulative wrapped progress** (summing the already-correctly-wrapped per-step `ds`, floor-dividing by track length) instead of a discrete per-step raw-diff check. Regression-tested against the exact numbers that exposed it (`tests/env/test_lap_counting_robustness.py`). This also **retroactively corrected** two of the six results below — TD3 Step A's original "1 lap" was the same artifact, not real driving.

## Results

| Run | Laps completed | Behavior |
|---|---|---|
| PPO Step A | **1** | Genuinely drives: wobbles off-track partway through the lap (184/2000 steps off-track, 94 collided) but recovers and finishes |
| TD3 Step A | 0 | Parked (`vx=0.000` for the entire episode) |
| SAC Step A | 0 | Parked, oscillating at the start/finish seam |
| PPO Step B | 0 | Drives safely (0 off-track steps) but crawls at ~0.5–0.75 m/s average — never covers a full lap's arc length across 80 varied tracks within 2000 steps |
| TD3 Step B | 0 | Same safe-but-inert pattern as PPO Step B, slightly worse (near-zero or negative net progress on some tracks) |
| SAC Step B | 0 | Same pattern again |

## Exit criterion: not met

Only PPO completes a lap, on Step A only. TD3 and SAC both collapsed into a "stay safe, minimize penalty, don't actually drive" local optimum on every one of their four runs (both steps). Step B shows the same qualitative pattern across all three algorithms: none complete a lap within the budget, though PPO/TD3/SAC at least stay on-track rather than crashing.

This is a real, informative training result, not an infrastructure failure — the pipeline (env, reward, vectorized rollout, TensorBoard logging, multi-track sampling) all work correctly; the lap-counting bug above is now fixed and permanently regression-tested. Two open questions worth carrying forward, not resolved here:

- Whether the off-track/collision penalties are too harsh relative to the progress reward, biasing the more aggressively-updating off-policy algorithms (SAC/TD3) toward the inert local optimum more than PPO's batched on-policy updates.
- Whether 300K timesteps is simply too few for SAC/TD3 to escape that optimum — untested; would need a longer run to distinguish "wrong reward balance" from "needs more training."

Left here per the team's own call — Phase 3 is documented as complete-with-a-negative-result rather than pushed further in this session.
