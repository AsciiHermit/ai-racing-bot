# Phase 5 Step A Results: Domain-Randomized RL with a Regret-Proxy Curriculum

Per `IMPLEMENTATION_PLAN.md` Phase 5's exit criterion: "at least one generalization-oriented method has a zero-shot held-out evaluation result, comparable against the Phase 3 baselines."

## Approach

Extended Phase 3's PPO baseline with per-episode track randomization: a 50-track pool generated fresh via Phase 2's generator (seeds 30000-30049, disjoint from the 110-track manifest), sampled each episode via `RegretCurriculumTrackProvider` (wayfinder ticket #10) — priority-weighted toward whichever pool track the agent is currently doing worst on relative to its own rolling-best return there, refreshed every 20 episodes. `off_track_terminates=True` (see below) and 4 parallel `DummyVecEnv` workers sharing the curriculum object by reference.

## Training divergence, and what actually fixed it

Every run at the Phase 3 reward settings (`off_track_terminates=False`, meaning an agent that leaves the track sits there accumulating roughly -6/step for up to 2000 steps) diverged, regardless of curriculum (regret or uniform) or batch size (`n_envs` 1 or 4) — `ep_rew_mean` fell monotonically into the thousands-negative range, with the agent driving fast but off-track most of the episode, not parked. An `n_envs=1`-starves-PPO hypothesis was tested and refuted: `n_envs=4` collapsed at the same point.

Three short (150-200K step) diagnostic runs isolated the actual cause: reward normalization alone (`VecNormalize`, rewards only) delayed but did not prevent collapse; off-track termination alone (ending the episode on the first off-track step, instead of letting the penalty accumulate) fully prevented it, with reward climbing steadily past +150; combining both was no better than termination alone. **Off-track termination was adopted as the fix**, unbounded per-step penalty accumulation being the root cause, not batch size or reward scale per se.

A genuine multi-env bug was found and fixed alongside this: `RegretCurriculumTrackProvider.record_episode()` implicitly attributed a finished episode to "whichever track was most recently sampled," which is wrong once more than one environment shares the provider (a different env's reset can happen in between). Fixed with an explicit `track_index` parameter and `index_of()`, with the training callback tracking each env's current track via `training_env.get_attr("track")`.

Two full training runs (regret curriculum; uniform sampling as an ablation) were separately interrupted by the same background-process memory-pressure safeguard when launched concurrently; both completed after being relaunched one at a time, with periodic checkpointing (`--checkpoint-every`) added specifically so a future interruption would not lose an entire run's progress.

## Result

Both models trained for the full 300,000 timesteps at the matched configuration (4 envs, off-track termination, same 50-track pool, same seed), then evaluated zero-shot via `scripts/evaluate_zero_shot.py --terminate-off-track` against the own training pool and all three manifest tiers, reporting progress (fraction of a lap driven before leaving the track):

| Tier | Regret curriculum | Uniform sampling (ablation) |
|---|---|---|
| Own training pool | 15.5% | 11.4% |
| Manifest train tier | 15.6% | 10.9% |
| Interpolation (held-out) | 11.9% | 7.7% |
| Extrapolation (held-out) | 11.0% | 7.3% |

Final `ep_rew_mean`: regret 160+, uniform 70.1.

The regret curriculum outperforms plain uniform sampling by roughly 35-45% relative, consistently across every tier — not just on the pool it was tuned against. This isolates the curriculum's own contribution from the termination fix, since both runs share it. Interpolation and extrapolation perform almost identically to each other under both curricula, meaning genuinely novel track topology was not found to be harder than merely-unseen-but-familiar-style topology at this stage. Manifest train tier and own training pool are both, strictly, zero-shot relative to training (disjoint seed ranges from the domain-randomization pool), and score almost identically to each other, suggesting the pool-vs-manifest distinction itself is not adding a meaningful confound.

Neither model completes a lap on any tier within the 300K-step budget.

**Exit criterion met**: a generalization-oriented method (domain-randomized PPO + regret curriculum) has a zero-shot held-out evaluation result, directly comparable against a matched-budget ablation and against Phase 3's baselines (which completed zero laps on the multi-track manifold at the same budget). The result is a real, measured, modest improvement, not a fully solved task: full lap completion under this reward/observation/budget combination remains open, and is a natural candidate for Phase 5 Step B (model-based RL, `r2dreamer`) rather than further Step A tuning.
