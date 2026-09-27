# Wayfinder ticket: rliable statistical protocol

**Issue:** [AsciiHermit/ai-racing-bot#14](https://github.com/AsciiHermit/ai-racing-bot/issues/14)
**Trigger:** `reports/Racing RL research directions.md` recommended replacing the Phase 6 evaluation harness's point-estimate-across-3–5-seeds design with `rliable`-style interval statistics, citing Agarwal et al. (NeurIPS 2021 Outstanding Paper, arXiv:2108.13264) and the `rliable` library.
**Scope of this note:** read-only research against primary sources only. No code changes, no git/gh actions taken.

**Primary sources consulted:**
- Agarwal, Schwarzer, Castro, Courville, Bellemare, "Deep Reinforcement Learning at the Edge of the Statistical Precipice," NeurIPS 2021 (arXiv:2108.13264; HTML full text at arxiv.org/html/2108.13264v4).
- `google-research/rliable` GitHub repo: `README.md`, `rliable/metrics.py`, `rliable/library.py` (fetched at their `master` HEAD via raw.githubusercontent.com).
- Mathieu, Della Vecchia, Shilova, Centa, Kohler, Maillard, Preux, "AdaStop: adaptive statistical testing for sound comparisons of Deep RL agents," TMLR 2024 (arXiv:2306.10882), and its reference implementation at `github.com/TimotheeMathieu/adastop`.

---

## Cited findings

### 1. Exact interval-estimate metrics and their robustness

`rliable` (`rliable/metrics.py`) provides four aggregate-score functions, each consuming a score matrix and reducing it to one number per algorithm:

- **`aggregate_mean(scores)`** — mean of per-task sample means. Not robust: a single anomalously high or low run on one task can move it.
- **`aggregate_median(scores)`** — median of per-task sample means. Robust to outliers but, per the paper, statistically inefficient — it needs many more runs to get a tight confidence interval than IQM does (paper §4.3, Fig. 2 right: "IQM requires fewer runs than median for small uncertainty").
- **`aggregate_iqm(scores)` — Interquartile Mean.** Implemented via `scipy.stats.trim_mean(scores, proportiontocut=0.25)`. Definition, verbatim from the paper (§4.3): IQM "discards the bottom and top 25% of the runs and calculates the mean score of the remaining 50% runs." Concretely, given the combined pool of `N` runs × `M` tasks scores, sort them, drop the bottom quartile and top quartile, and take the mean of the middle 50% (⌊NM/2⌋ scores). This is the paper's and library's **recommended default aggregate metric** for exactly this reason: it discards the same outlier-driving tail mass as the median (so it shares the median's robustness to a single catastrophic or lucky run) but, because it still averages over half the data rather than picking one central point, it converges with a narrower CI for a given run budget — i.e. it is more statistically efficient than the median while remaining far more robust to outliers than the mean.
- **`aggregate_optimality_gap(scores, gamma=1.0)`** — per the paper (§4.3, Fig. 8 caption): "the amount by which the algorithm fails to meet a minimum score of γ=1.0." Computed as `gamma - mean(min(scores, gamma))`, i.e. clip every run's score at the threshold `gamma` and report the shortfall of the mean from that ceiling. This is specifically sensitive to failures below a target/threshold performance level (e.g., "did the agent finish the lap at all," or "did it beat the classical reference") rather than to the full score distribution — a different robustness profile from the other three, by design.

`rliable` also provides `metrics.probability_of_improvement(scores_x, scores_y)`: a per-task Mann-Whitney U statistic normalized by `(num_runs_x × num_runs_y)`, averaged across tasks, giving `P(X_run > Y_run)` — a direct, distribution-free head-to-head comparison metric that sidesteps aggregation entirely.

### 2. Bootstrapping method and resample count

Both the paper and the library use the **percentile bootstrap**, applied in a *stratified* way. From the paper (§4.1): the procedure re-samples "runs with replacement independently for each task to construct an empirical bootstrap sample with N runs each for M tasks" — i.e., resampling is stratified by task so that each bootstrap replicate still has the same task composition, only the runs within each task are resampled.

The library's `library.get_interval_estimates(score_dict, func, method='percentile', task_bootstrap=False, reps=50000, confidence_interval_size=0.95, ...)` operationalizes this with a `task_bootstrap` flag:
- `task_bootstrap=False` (the default): resample only over runs (within each task) — this "captures the statistical uncertainty in the aggregate performance if the experiment is repeated using a different set of runs (e.g., changing seeds) on the same set of tasks."
- `task_bootstrap=True`: resample over both runs *and* tasks jointly — appropriate when the task set itself is also treated as a sample (relevant to this project — see §4/recommendation below).

**Default number of resamples: `reps=50000`** (the parameter's default value in `library.py`'s signature, also the value used in every worked example in the paper and README). The library separately shows `reps=2000` in the probability-of-improvement examples in the README, a smaller default because that computation is per-pair rather than per-algorithm.

### 3. Minimum seed-count guidance

The paper does **not** give a fixed "use N seeds" rule — it deliberately avoids prescribing one. What it does give:

- An empirical coverage study (§4.1) on percentile CIs, finding they "provide good interval estimates for as few as N=10 runs" for IQM/median-type aggregate scores — but this result is for aggregating **across a large task set** (the Atari-100k/Atari-200M benchmarks, dozens of games), not for a single task with few runs. The paper explicitly separates aggregate-CI coverage from per-task CI coverage, and for the latter states coverage remains below the nominal 95% "even with 30 runs per game" (Appendix A.5, Fig. A.18) — per-task/per-condition uncertainty is harder to pin down than benchmark-wide aggregate uncertainty.
- A demonstration (§3, Fig. 6) that comparisons based on median scores only become "statistically defensible" at ℓ=25 runs (or ℓ=10 for a coarser claim) — used to argue point estimates from small run counts are unreliable, not to hand down a target run count for every experiment.
- The paper's actual recommended decision procedure is **not a number, it's a workflow**: report interval estimates (CI width) alongside the point estimate for whatever run budget you have, and let the CI width itself tell you whether the comparison is decisive — a wide, overlapping CI means "inconclusive; you don't yet have enough runs to claim a difference," not a fixed run count you must reach beforehand.

This is precisely the gap **AdaStop** (arXiv:2306.10882, TMLR 2024) targets directly, and its abstract is unambiguous about the field's current default being inadequate: *"Researchers in Deep RL often use less than 5 independent executions to compare algorithms: we claim that this is not enough in general."* See §5 below.

### 4. Practical usage

Confirmed API shape from `rliable/library.py` and the README example:

```python
# score_dict: Dict[str, np.ndarray], each array shape = (num_runs, num_tasks)
algorithms = ['DQN (Nature)', 'DQN (Adam)', 'C51', ...]
aggregate_func = lambda x: np.array([
    metrics.aggregate_median(x),
    metrics.aggregate_iqm(x),
    metrics.aggregate_mean(x),
    metrics.aggregate_optimality_gap(x)])
aggregate_scores, aggregate_score_cis = rly.get_interval_estimates(
    score_dict, aggregate_func, reps=50000)
```

`get_interval_estimates` returns `(point_estimates_dict, cis_dict)`, each keyed by algorithm name, ready to feed straight into `plot_utils.plot_interval_estimates(...)`. Other plotting helpers: `plot_utils.plot_probability_of_improvement`, `plot_utils.plot_performance_profiles`, `plot_utils.plot_sample_efficiency_curve`; the performance-profile path uses `rly.create_performance_profile(score_dict, tau_list)`, which reports "how many scores are above tau, averaged across all runs" for a swept set of thresholds `tau`, with its own CI.

**Mapping onto this project's data:** the library's native shape — `num_runs × num_tasks` — maps cleanly onto this project's design if **held-out tracks are treated as the "task" axis and seeds as the "run" axis**. Concretely, for each algorithm/method under test, build one `(num_seeds, num_held_out_tracks)` matrix per metric (lap time, completion rate, generalization gap), and pass that matrix (or a dict of one array per algorithm) into `metrics.aggregate_iqm` / `aggregate_optimality_gap` / `rly.get_interval_estimates` exactly as in the README example above. This also means `task_bootstrap=True` is the more faithful setting for this project specifically, since the held-out track *set itself* (Phase 2's ~20-track manifest) is a finite sample the team wants the conclusion to generalize beyond — `task_bootstrap=False` would only capture "what if we reran with different seeds on this exact track set," understating the true uncertainty.

### 5. Adaptive stopping rules for seed count (AdaStop)

Verified against a primary source: **arXiv:2306.10882**, "AdaStop: adaptive statistical testing for sound comparisons of Deep RL agents" (Mathieu, Della Vecchia, Shilova, Centa, Kohler, Maillard, Preux), published in TMLR 2024, with a maintained reference implementation at `github.com/TimotheeMathieu/adastop`.

Core idea, in the authors' own words (abstract): AdaStop is "a new statistical test based on multiple group sequential tests. When used to compare algorithms, AdaStop adapts the number of executions to stop as early as possible while ensuring that enough information has been collected to distinguish algorithms that have different score distributions." It comes with a theoretical guarantee on the family-wise error rate when comparing more than two algorithms at once (correcting for multiple-comparisons inflation, which a naive per-pair test would not). In short: rather than fixing a seed count up front, you run a small batch, test, and either stop (a significant or clearly-non-significant difference has emerged) or run another batch of seeds — repeating until the group-sequential test triggers a stop, at a controlled false-positive rate.

This is a different tool from `rliable`: `rliable` gives you the *reporting* statistics (CIs, IQM, etc.) for whatever run budget you already spent; AdaStop gives you a *stopping rule* for how many runs to spend in the first place. They are complementary, not competing — this is a genuine finding of this research pass, not carried over unverified from the earlier report.

---

## Gaps

- Neither the paper nor the library gives a closed-form "N seeds → X% CI width" formula usable without first collecting pilot data — any minimum-seed number this project adopts will be a heuristic/pilot-based decision, not something citable directly from the source.
- The paper's N=10-runs coverage result (§4.1) was validated on Atari-scale benchmarks with dozens of tasks; it was not verified in this pass whether that coverage result holds up at the much smaller task count this project would use (a handful of held-out tracks, not ~60 games) — the paper does note per-task/small-M coverage is generally worse, so this is a caution, not a green light to assume N=10 seeds suffices here.
- AdaStop's guarantees and worked examples are validated on MuJoCo control benchmarks in the paper; this pass did not verify a racing-RL or track-generalization use case, nor did it inspect the `adastop` library's actual API surface (only confirmed the repo exists and the paper's claims) — treat AdaStop as a promising fit to evaluate hands-on, not a drop-in verified for this project's exact metrics.
- `rliable`'s performance-profile and probability-of-improvement functions were confirmed to exist and were summarized from the README/source but not exercised against a worked numeric example in this pass.

---

## Recommended minimum seed count / protocol for this project's Phase 6

Replace the current placeholder ("3–5 seeds," "a bootstrapped CI or Welch's t-test") in `IMPLEMENTATION_PLAN.md` §9 with a concrete, sourced protocol:

1. **Metric:** report IQM (`metrics.aggregate_iqm`) as the primary aggregate statistic for lap time, completion rate, and generalization gap, alongside the optimality gap against the Phase 4 classical reference (γ = the reference lap time or a normalized 1.0, per the paper's convention) — not a bare mean or a single point estimate.
2. **Interval:** compute 95% stratified percentile bootstrap CIs via `rly.get_interval_estimates(score_dict, aggregate_func, reps=50000, task_bootstrap=True)`, with `task_bootstrap=True` because the held-out track manifest is itself a finite sample this project wants conclusions to generalize past (see §4 above). Build score matrices as `(num_seeds, num_held_out_tracks)` per algorithm.
3. **Seed count:** do not hard-code a fixed number. Set a floor of **at least 10 seeds per algorithm/condition** (informed by the paper's own N=10 coverage result for aggregate CIs, applied conservatively given this project's much smaller track set — treat 10 as a floor, not a target), then keep the CI width itself as the stopping signal: keep adding seeds until the IQM's 95% CI half-width is small enough that the CIs for the methods being compared (e.g. Phase 5's method vs. a Phase 3 baseline) either clearly separate or clearly and stably overlap — either outcome is a reportable, decisive result; a wide, ambiguous CI is not. Pre-register the CI-width threshold that counts as "tight enough" before running the comparison, to avoid post-hoc rationalizing extra seeds.
4. **If the team wants a formal, automatic version of step 3** rather than an eyeballed CI-width rule, evaluate `AdaStop` (`github.com/TimotheeMathieu/adastop`) as a Phase 6 tooling spike: it directly answers "how many more seeds do we need" via group sequential testing with a family-wise error guarantee, which matters once Phase 6 is comparing more than two conditions (multiple baselines × multiple Phase 5 methods) at once. This should be scoped as a small follow-up spike, not assumed to drop in without verification (see Gaps).
5. Keep Welch's t-test as an optional secondary sanity check if desired, but the bootstrap/IQM path above should be the reported result, per the paper's central finding that point estimates and naive parametric tests are exactly what produced the "false precipice" of unreliable comparisons in prior deep-RL literature.
