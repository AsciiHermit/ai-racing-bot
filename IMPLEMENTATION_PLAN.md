# Implementation Plan: Cross-Track Generalization in Racing RL

*Execution plan for ai-racing-bot Chapter 1 — companion to the [Problem Definition](PROBLEM_DEFINITION.md) doc*

## 1. Overview

The Problem Definition doc sets a single question: *can an RL policy trained on ai-racing-bot generalize, zero-shot, to unseen track geometries, approaching a classical optimal-control reference lap time, with better sample efficiency than per-track PPO/SAC/TD3?*

This document is the execution plan for answering that question. It is organized as seven sequential phases (Phase 0–7), each with concrete tasks, file/module targets in the existing repo, and an exit criterion. Phases are ordered so that each one only depends on what the phase before it produced — you cannot meaningfully run Phase 3 (baselines) on physics that still has an artificial grip cap, and you cannot compute a "generalization gap" (Phase 6) without the classical reference from Phase 4. §11 lays out the full dependency chain and gate criteria explicitly.

## 2. Current Progress — Baseline Audit

*(Based on an automated read of the repo's README, PHYSICS.md, and directory listing on 2026-09-26 — not a full clone. Re-verify against the live repo before treating percentages as exact.)*

| Module | What exists today | Readiness for this plan |
|---|---|---|
| Interfaces (`ars/core/interfaces.py`) | `VehiclePhysics`, `Track`, `Sensor` Protocols, fully defined | **Done.** This is the extension point every later phase plugs into — no rework needed. |
| Physics (`ars/physics`) | `KinematicBicyclePhysics`: integrates position/heading from speed+steer, hard 4.5g lateral-acceleration cap, no slip angle/ratio, no tire curve, lateral velocity hardcoded to 0, mass unused | **Not started for this plan's purposes.** This is Phase 1's target — the existing stub is a placeholder, explicitly flagged as such in the repo's own `PHYSICS.md`. |
| Track (`ars/track`) | One fixed oval loop | **Not started.** Phase 2's target. |
| Sensors (`ars/sensors`) | GPS (equirectangular), IMU (accel + yaw rate via velocity differentiation), forward lidar, proprioceptive, track-relative pose | **Largely done.** Sufficient sensor variety already exists to run the observation-modality ablation in Phase 5/6 without new sensor work, though the IMU's differentiation-based approach should be revisited once Phase 1 lands real accelerations. |
| Environment (`ars/env`) | Gymnasium env registered as `ARS-Racing-v0` | **Done, but untested at scale.** Needs verification it works cleanly with vectorized/parallel rollout (SB3's `SubprocVecEnv`) — part of Phase 3. |
| Agents (`ars/agents`) | Random agent, dummy line-following "expert" | **Not started.** No PPO/SAC/TD3 baseline exists yet — Phase 3's target. |
| Visualization/Tooling (`ars/viz`, `ars/dashboard`) | Pygame live viewer, 5-step setup dashboard | **Done and ahead of the research substance.** Useful for Phase 0/1 validation (visually sanity-checking the new tire model) but not on this plan's critical path. |
| Evaluation harness | None | **Not started.** Phase 6's target. |
| Classical reference solver | None | **Not started.** Phase 4's target. |

**Net position:** the project has solid scaffolding (interfaces, sensors, Gym registration, tooling) but none of the four things the problem statement actually depends on — real vehicle dynamics, a track distribution, trained baselines, and a reference optimum — exist yet. This plan starts effectively from zero on the research substance, with a well-built foundation to build it on.

## 3. Phase 0 — Validate Assumptions (Technical Spike)

**Goal:** de-risk the plan's core assumption before investing in it — that a dynamic-bicycle + Pacejka model is hard enough to make cross-track generalization a real learning problem, not one a simple controller already solves.

**Tasks:**

- Implement a minimal, throwaway version of a slip-based tire model (does not need to be production-quality).
- Run a hand-tuned PID or pure-pursuit controller against it on the existing fixed oval.
- If the PID controller already achieves near-optimal lap times → the model isn't demanding enough yet; flag for Phase 1 to add tire wear or basic load transfer earlier than planned.
- If the PID controller struggles or is clearly sub-optimal relative to a quick manual/optimized line → confirms the task is non-trivial; proceed as planned.
- Decide the RL library: **recommend Stable-Baselines3** over RLlib for this phase — simpler API, sufficient for single-machine PPO/SAC/TD3, lower ramp-up cost for a small team. Revisit RLlib only if parallel-rollout throughput becomes a bottleneck.
- Decide the track-generation approach: spline-based (MicroRacer-style cubic splines) is recommended over a Box2D physical mesh (CarRacing-style) — simpler to control curvature/difficulty parametrically, which Phase 2 and the generalization study depend on.

**Exit criterion:** a short (1–2 page) internal note stating the go/no-go decision on physics difficulty, plus the two library/approach decisions above. **This phase blocks Phase 1 and Phase 2 — do not start either until this note exists.**

## 4. Phase 1 — Physics Upgrade (Module A core)

**Goal:** replace the kinematic-bicycle-with-hard-cap stub with a dynamic bicycle model whose grip limits emerge from tire behavior, not an imposed ceiling.

**Tasks:**

- Create `ars/physics/dynamic_bicycle.py` implementing the `VehiclePhysics` Protocol (`reset()`, `step()`), following the repo's own `PHYSICS.md` handoff guidance (an enriched state type such as `DynamicBicycleState` carrying slip angle, alongside the existing chassis-level fields for duck-typing compatibility with the rest of the stack).
- Compute front/rear slip angles from the vehicle's velocity vector, yaw rate, and steering angle.
- Implement a Pacejka "Magic Formula"-style lateral tire force curve (`F_y = μ·F_z·D·sin(C·arctan(Bα − E(Bα − arctan(Bα))))`), calibrated against the repo's existing reference constants (798 kg mass, 2.6 m wheelbase, ~4.5g prior cap as an upper sanity bound, not a hard limit going forward).
- **Remove the hard 4.5g lateral-acceleration ceiling** — grip limits should now emerge from tire saturation, not be imposed externally.
- Wire mass into the dynamics (currently unused) so fuel/payload effects are possible in later chapters without rework.
- Add a skid-pad regression test: confirm steady-state maximum lateral acceleration under the new model lands in a physically plausible range without instability or NaNs at the grip limit.
- Update `ars/env/factory.py` and `ars/dashboard/builder.py` to make the physics backend selectable (kinematic stub vs. dynamic bicycle), preserving the old stub as a fast/degenerate option for later ablations.

**Exit criterion:** the new physics model passes the skid-pad test, integrates cleanly into the existing Gym env, and a quick visual check in the pygame viewer shows plausible slip/oversteer behavior near the grip limit. **Blocks Phase 3 and everything downstream** — training baselines on physics that still confounds "fast line" with "artificial cap" would invalidate the whole benchmark.

## 5. Phase 2 — Procedural Track Generator (Module B)

**Goal:** define "unseen track" rigorously and reproducibly.

**Tasks:**

- Implement a spline-based generator (cubic/Catmull-Rom splines defining inner/outer track borders, per the reference approach in MicroRacer — Asperti & Del Brutto, arXiv:2203.10494) with controllable curvature, turn count, and width, all seeded for reproducibility.
- Implement the `Track` Protocol (`query()`, `sample_at_s()`, `is_on_track()`) for the generated tracks, matching the existing fixed-oval implementation's interface.
- Define a track-difficulty/diversity metric (e.g., mean curvature, curvature variance, minimum radius) so "held-out" tracks can be characterized, not just counted.
- Generate and **check into the repo a fixed seed manifest**: a reproducible train set (recommend starting at ~50–100 tracks) and a held-out set (~20 tracks), so every later experiment references the same splits.
- Sanity-check visually via the existing pygame viewer (no new tooling needed here — reuse what exists).

**Exit criterion:** train/held-out track sets are generated, seeded, checked in, and pass a basic diversity check (no near-duplicate tracks across held-out and train sets). **Can run in parallel with Phase 1** — no dependency between them, both only depend on Phase 0's go/no-go.

## 6. Phase 3 — Baseline RL Agents (Module D, baselines only)

**Goal:** get PPO/SAC/TD3 trained and behaving sanely before touching anything novel.

**Tasks:**

- Integrate **Stable-Baselines3** rather than hand-rolling algorithms (the existing `ars/agents` module currently only has random and dummy-expert agents).
- Wrap `ARS-Racing-v0` for vectorized rollout (`SubprocVecEnv` or equivalent) — verify the existing Gym registration supports this cleanly; fix if not.
- Define the observation space from the existing sensor stack: LiDAR + proprioceptive + track-relative pose as the default bundle (full ablation of modality combinations is deferred to Phase 5/6).
- Define the reward function: track-progress-based (not raw speed, to avoid reward hacking), with off-track and wall-collision penalties.
- **Step A — single-track sanity check:** train PPO, SAC, TD3 on the original fixed oval only. This reproduces the "solved" single-track case from §2 of the Problem Definition and validates the training harness before scaling up.
- **Step B — in-distribution multi-track training:** train the same three algorithms across the full Phase 2 train set (not per-track). This is the "in-distribution" performance baseline the generalization gap (Phase 6) will be measured against.
- Log everything via TensorBoard or W&B from the start — retrofitting logging after runs are underway loses data.

**Exit criterion:** PPO/SAC/TD3 all complete laps reliably on the fixed oval (Step A) and show reasonable in-distribution performance across the train set (Step B), with training curves logged. **Depends on Phase 1 (physics) and Phase 2 (tracks) both being complete.**

## 7. Phase 4 — Classical Optimal-Control Reference

**Goal:** produce the per-track optimal lap time that the "generalization gap" metric (Phase 6) is measured against — without this, there is no way to tell whether a generalizing policy's lap times are actually good or just self-consistent.

**Tasks:**

- Implement or adapt a minimum-curvature or MPC-based trajectory optimizer that consumes the same track-spline representation Phase 2 produces.
- Evaluate adapting an existing open implementation (e.g., TUM's `global_racetrajectory_optimization` — *its exact repository name, license, and current maintenance status have not been re-verified in this session; confirm directly before adopting it*) versus a from-scratch minimal solver, given the team's compute/time budget.
- Run the solver against every track in both the Phase 2 train and held-out sets, producing a reference optimal lap time per track.
- Store these references alongside the track manifest so Phase 6 can look them up directly.

**Exit criterion:** every track in both sets has a reference optimal lap time. **Can run in parallel with Phase 3** — both depend on Phase 2's track set, not on each other.

## 8. Phase 5 — Generalization-Oriented Method

**Goal:** the actual research contribution — a method that generalizes zero-shot to held-out tracks better than, or more sample-efficiently than, the Phase 3 baselines.

- **Step A (lower risk, do first): domain-randomized model-free RL.** Extend the Phase 3 PPO/SAC training with randomized track curvature/width parameters sampled per-episode (not just a fixed train set), evaluate zero-shot on the Phase 2 held-out set. This gets an end-to-end generalization result with minimal new implementation risk.
- **Step B (higher payoff, do second): model-based RL.** Implement or adapt a Dreamer-style latent world-model agent (e.g., adapting an existing implementation such as `dreamerv3-torch`) trained across the Phase 2 train set, evaluated zero-shot on held-out tracks. This follows Brunnbauer et al. (arXiv:2103.04909), who showed this approach improves both sample efficiency and zero-shot transfer over model-free RL on F1TENTH — but that result has not been shown on this environment and should be verified here, not assumed.
- **Step C (ablation, once A and/or B work): observation-modality sweep.** Re-run the best-performing method with LiDAR-only, LiDAR+proprioceptive, and full track-relative-pose observation bundles, to test whether generalization robustness depends on observation choice here the way Brunnbauer et al. found it did in their setting.

**Exit criterion:** at least one generalization-oriented method has a zero-shot held-out evaluation result, comparable against the Phase 3 baselines under the Phase 6 harness. **Depends on Phase 3 (baselines to compare against) and Phase 4 (reference lap times) both being complete.**

## 9. Phase 6 — Evaluation Harness & Benchmark Run

**Goal:** turn individual training runs into a single, statistically defensible comparison.

**Tasks:**

- Build an `ars/eval` module computing, per run: lap time, completion rate, off-track/wall-collision rate, generalization gap (held-out performance minus in-distribution performance, and separately vs. the Phase 4 classical reference), and sample efficiency (environment steps to reach a fixed fraction of asymptotic reward).
- Run every condition (Phase 3 baselines × Phase 5 methods) across **at least 3–5 random seeds each** — RL training is noisy enough that single-seed comparisons are not conclusive; report mean and variance, not point estimates.
- Apply a significance test (e.g., a bootstrapped confidence interval or Welch's t-test across seeds) before claiming one method beats another.
- Produce the final comparison table/plots: lap time and generalization gap per method, with error bars.
- **Report negative or mixed results as-is.** If the generalization-oriented method from Phase 5 doesn't beat the baselines, that is still a valid, reportable outcome for this chapter — do not reframe it as success.

**Exit criterion:** a complete results table covering all conditions, ready to write up. **Depends on Phase 3, Phase 4, and Phase 5 all being complete.**

## 10. Phase 7 — Documentation & Open-Source Release

**Goal:** make the result usable and citable by others, not just true internally.

**Tasks:**

- Write up the Phase 6 results as a technical report / paper-style write-up, positioned explicitly against the related work identified in the Problem Definition's §2 (F1TENTH, MicroRacer, CarRacing, Brunnbauer et al.).
- Package a reproducible release: the checked-in track manifest (Phase 2), reference lap times (Phase 4), and baseline model checkpoints (Phase 3/5), with instructions to reproduce every number in the results table.
- Add a `CONTRIBUTING.md`, confirm a license is set, and version the benchmark (e.g., a `v1.0` tag) so later chapters (multi-agent, strategy) can build on a stable base rather than a moving target.
- Update the repo's README to reflect the new physics/track/agent modules, replacing the "kinematic stub / fixed oval" description that is currently accurate but will be outdated.

**Exit criterion:** an external user can clone the repo, follow the instructions, and reproduce the headline result. **Depends on Phase 6 being complete.**

## 11. Milestones, Sequencing & Dependency Gates

```
Phase 0 (spike, go/no-go)
 ├──→ Phase 1 (physics) ───┐
 └──→ Phase 2 (tracks) ────┤
                            │
      ├──→ Phase 3 (baselines) ─┐
      │      └──→ Phase 5 (generalization method) ─┐
      └──→ Phase 4 (classical reference) ┘          │
                                                     │
                            └──→ Phase 6 (evaluation) ──→ Phase 7 (release)
```

**Parallelizable pairs** (if the team can split work): Phase 1 ∥ Phase 2 (after Phase 0); Phase 3 ∥ Phase 4 (after Phase 1 and Phase 2 are both done).

**Hard gates — do not skip ahead:**

1. Phase 0's go/no-go note must exist before Phase 1 or Phase 2 starts.
2. Phase 1 and Phase 2 must both pass their exit criteria before Phase 3 starts — training on the old physics or the single oval wastes the run.
3. Phase 3 and Phase 4 must both be complete before Phase 5 starts — without baselines and a reference, a "generalization result" has nothing to be compared against.
4. Phase 6 requires Phase 3, 4, and 5 all complete — partial results produce a misleading comparison table.

Suggested first milestone for the team: treat **Phase 0 + Phase 1 exit criteria met** as the first checkpoint to review together, since Phase 1 is both the highest-priority and highest-risk item (per the Problem Definition's §3).

## 12. Risk Register

| Risk | Phase affected | Mitigation |
|---|---|---|
| Dynamic-bicycle + Pacejka model turns out not hard enough (a simple controller still nails it) | 1 | Caught early by Phase 0's spike, before the full model is built |
| Procedural generator's difficulty/diversity dominates results more than algorithm choice does | 2, 6 | Measure and report track-diversity metrics explicitly; don't treat track count alone as sufficient |
| Compute budget: multi-seed × multi-track × multi-algorithm training adds up even in a lightweight sim | 3, 5, 6 | Agree a compute ceiling before Phase 3 begins; Step A of Phase 5 (domain randomization) is the fallback if Step B (model-based RL) proves too expensive |
| Classical reference solver (Phase 4) is harder to adapt than expected (license/compatibility issues with existing implementations) | 4 | Flagged as unverified in the Problem Definition; budget time to evaluate a from-scratch minimal solver as a fallback |
| Model-based RL implementation risk is high; team may not finish Step B of Phase 5 in time | 5 | Step A (domain randomization) is designed as a complete, publishable result on its own if Step B slips |
| Negative/mixed generalization result | 6 | Explicitly treated as a valid, reportable outcome in this plan — not a failure condition for the chapter |

## 13. Sources

- Betz, J., Zheng, H., Liniger, A., Rosolia, U., Karle, P., Behl, M., Krovi, V., & Mangharam, R. (2022). Autonomous Vehicles on the Edge: A Survey on Autonomous Vehicle Racing. *IEEE Open Journal of Intelligent Transportation Systems*, 3, 458–488. https://doi.org/10.1109/OJITS.2022.3181510 (also arXiv:2202.07008)
- Brunnbauer, A., et al. (2021). Model-based versus Model-free Deep Reinforcement Learning for Autonomous Racing Cars. arXiv:2103.04909.
- O'Kelly, M., Zheng, H., Karthik, D., & Mangharam, R. (2020). F1TENTH: An Open-source Evaluation Environment for Continuous Control and Reinforcement Learning. *Proceedings of Machine Learning Research* (NeurIPS 2019 Competition and Demonstration Track), 123.
- Asperti, A., & Del Brutto, M. (2022). MicroRacer: a didactic environment for Deep Reinforcement Learning. arXiv:2203.10494 (also LOD 2022 proceedings).
- TUM `global_racetrajectory_optimization` (GitHub) — reference minimum-curvature trajectory optimizer. *Not re-verified in this session (exact name/license/maintenance status); confirm before adopting.*
- Stable-Baselines3 documentation — https://stable-baselines3.readthedocs.io/
- Repository under discussion: https://github.com/AsciiHermit/ai-racing-bot *(state as read on 2026-09-26 via an automated fetch of its README and PHYSICS.md, not a full clone — verify against the live repo)*
- See the companion [Problem Definition](PROBLEM_DEFINITION.md) doc for the full source list behind the gap claim this plan executes against.
