# MPC-Teacher Source Papers: Concrete Parameters

Research for GitHub issue [AsciiHermit/ai-racing-bot#5](https://github.com/AsciiHermit/ai-racing-bot/issues/5),
"Extract concrete parameters from the MPC-teacher source papers." This supersedes the paragraph-level
summaries of these three papers in `reports/Racing RL research directions.md`.

Method: all three papers were fetched and read directly from arXiv (abstract pages plus full PDF text,
extracted with `pdftotext -layout` where needed) — no secondary summaries were relied on for the figures
quoted below.

---

## 1. arXiv:2608.12063 — "Learning Loco-Manipulation From SMPC Demonstrations With Sparse Offline-to-Online RL"

Schuck, Sorokin, Manni, Ta, Schoellig, Hutter, Le Cleac'h, Brüdigam (RAI Institute / TU Munich / ETH Zurich).
Posted 12 Aug 2026. Not an F1-style racing paper — it is loco-manipulation for a Spot quadruped and a G1
humanoid — but it is the source for the "sampling-based MPC expert seeding a TD3 replay buffer with a
phase-out curriculum" pattern.

### Cited findings

**Expert-transition mixing ratio**
- "During the initial phases of learning, we replace 50% of the transitions in the replay buffer with
  pre-collected expert data (detailed in Section 3.3)." — Section 3.2, "Training Off-Policy RL with Sparse
  Rewards."
- Confirmed in the ablations: "In our experiments, we have fixed the ratio of expert data before
  phase-outs to 50%, following the recommendations of Ball et al. [10]." — Appendix A, "Percentage of
  Expert Data." Ball et al. [10] = P. J. Ball, L. Smith, I. Kostrikov, S. Levine, "Efficient online
  reinforcement learning with offline data," ICML 2023.
- Table 1 ("Hyperparameters for the sparse TD3 training") lists this again as a fixed hyperparameter (see
  full hyperparameter reconstruction below).
- The 50% ratio is *not* universally load-bearing: "For reaching, box pushing, and tire uprighting, we
  find that training runs are almost unaffected by the choice of the expert ratio... The tire-rolling
  task, however, is sensitive to the parameter and fails to converge when the amount of expert data is
  reduced [below 50%]." — Appendix A, referencing Figure 9 (ablated ratios shown are 0.1, 0.25, 0.5, 0.75,
  0.9 based on the figure's y-axis labels as extracted).

**Curriculum / phase-out schedule**
- Mechanism: "To prevent the policy from continually relying on these demonstrations, we use a curriculum
  that phases out the expert data once the agent achieves a sufficient empirical success rate, shifting to
  pure online learning." — Section 3.2.
- Exact trigger: "Phase-out threshold: 0.1" in Table 1, and confirmed narratively in Appendix A,
  "Phase-Out Threshold": "we therefore introduce a curriculum that phases out the data once runs achieve a
  10% success rate."
- Mechanism is a **hard cutoff, not a gradual decay**: once the empirical success rate crosses 10%, expert
  transitions stop being injected into the buffer at all — there is no described ramp/anneal.
- Sensitivity: Appendix A, "Phase-Out Threshold" (Figure 11) ablates the threshold itself across Spot
  tasks and finds the 10% default is close to optimal: "Keeping the offline SMPC data in the buffer results
  in significant performance degradation across all tasks except reach... if the data is not removed,
  training progress slows down noticeably, affecting the thresholds above 25%... Runs that keep the data
  indefinitely continue to make only slow progress." Their explanation: "the SMPC data is necessary to
  guide initial exploration, [but] it is too far from the policy to achieve the fine-grained improvements
  required by the tasks, and it becomes harmful once the policy is narrowed to a successful strategy with
  a shifted data distribution."

**Architecture deltas vs. a non-demonstration baseline**
- There is **no network-architecture difference** between the demo-augmented agent and the ablation without
  SMPC data — the comparison in Figure 4 ("w. SMPC data" vs. "w/o SMPC data," Spot Reach panel) uses the
  identical network; only the data/curriculum pipeline differs. Without SMPC data, "learning complex
  loco-manipulation policies from sparse rewards fails to learn anything" (Figure 4 caption).
- The paper does describe architecture modifications relative to a generic/vanilla TD3, but these apply
  uniformly to *all* runs (demo and non-demo alike), not specifically to the demo condition:
  - "a modified FastTD3 architecture" (Section 2, Related Work; FastTD3 = Seo et al. 2025, arXiv:2505.22642).
  - **Bounded critics**: "because our sparse reward formulation defines strictly known maximum and minimum
    possible returns, we structurally bound the critic network outputs to these limits... bounded critics
    have been successfully applied in advanced algorithms [CrossQ]." (Section 3.2). Mechanism detailed in
    Appendix B.1: a tanh activation after the critic's final linear layer, shifted/scaled to
    `[Q_min, 0]` where `Q_min = -γ/(1-γ)` (γ=0.99), plus a constant bias pushing the initial output toward
    zero.
  - **Asymmetric actor-critic**: "the actor receives noisy observations, while the critic receives exact
    observations from the simulation" (Appendix B).
  - Reward scaling to keep value predictions near 0 (Appendix B; scale = 0.01 per Table 1).
  - Action-delta parameterization (outputs are deltas to current velocity/pose targets, not absolutes) to
    bound achievable accelerations (Section 3.2) — again, not demo-specific.

**Reconstructed Table 1 hyperparameters** (sparse TD3 training; the PDF-to-text extraction badly scrambled
this table's two columns, so the parameter/value pairing below is inferred by matching the ordered list of
12 parameter names to the ordered list of 12 numeric values as they appear in the extracted text — flagged
as inferred, not a direct quote):
- Buffer size B = 16,777,216 (= 2²⁴)
- Gradient steps per update N = 8
- Phase-out threshold = 0.1
- Batch size Nb = 8192
- Discount factor γ = 0.99
- Polyak factor = 0.005
- Actor learning rate = 3×10⁻⁴
- Critic learning rate = 1×10⁻⁴
- Policy delay = 2
- Smoothing noise σs = 0.05
- Action regularization = 5×10⁻⁴
- Reward scale = 0.01

**Other quantitative results worth carrying over**
- Data generation rate: "generate high-quality demonstration datasets at a rate of one million samples
  per hour" on a single GPU, bootstrapping the agent "within four hours" total (Section 3.3).
- Dataset sizes actually used per task, per Figure 6/Q2: hardest task (tire rolling) needed "four million
  samples collected within 4 GPU hours to converge"; easier tasks (reach) converge with far less (down to
  250K in the ablation).
- Policy vs. teacher performance (Q1, Figure 5): "our sparse-reward approach consistently outperforms
  SMPC. Some tasks see improvements above 50%... a 11-45% reduction in standard deviation of task
  duration." SMPC statistics computed "over the complete dataset of 4M samples," policy statistics "over
  50k simulated episodes without exploration noise."
- SMPC expert itself runs at "≈0.5× real-time performance" on a single RTX 5090 (Section 3.3).
- Multimodality matters: mixing multiple SMPC solution modes (e.g., kicking vs. pushing the tire) in the
  demo data causes the downstream policy to "completely fail to learn, even though the demonstration
  success rate is slightly higher" (Q4, Figure 8) — uni-modal demonstrations are required.

### Gaps
- The exact set of expert-ratio values ablated in Figure 9 (Appendix A) could only be read off the
  y-axis labels in the PDF text extraction (0.1, 0.25, 0.5, 0.75, 0.9-ish); the figure itself (a plot) was
  not visually inspected, so treat the exact set of tested ratios as approximate.
- Table 1's parameter–value pairing is a confident reconstruction, not a verbatim table read — the PDF
  extraction interleaved the two columns. If exact hyperparameter values matter for implementation, the
  PDF's Table 1 should be re-checked visually (e.g., open the PDF and look at the actual table) rather than
  relying on this reconstruction alone.
- This is not a racing paper — no lap-time, waypoint, or track-specific parameters exist in it at all. Any
  transfer of its ratio/threshold numbers to the MPC-teacher racing use case is an analogy, not a direct
  precedent.
- No comparison is given against a non-TD3 baseline algorithm, nor against a non-sparse (hand-shaped dense
  reward) baseline — the paper's only ablation axis relevant to "does the mechanism help" is
  with-vs-without SMPC data (Figure 4), not with-vs-without curriculum phase-out as an isolated variable
  (that's covered, but only as a threshold-value sweep, in Appendix A).

---

## 2. arXiv:2408.04198 — "F1tenth Autonomous Racing With Offline Reinforcement Learning Methods"

Koirala & Fleming (Iowa State University). Posted 8 Aug 2024.

### Cited findings

**Offline dataset generation**
- Controller type: a waypoint/lookahead-point controller, *not* Pure Pursuit exactly but "a method similar
  to pure pursuit" (Section II, Related Works): "The error is the difference between the current heading
  angle and the angle made with the lookahead point. The final error angle is derived as a weighted mean of
  the errors with the lookahead points, wherein the contribution of distant points is mitigated by a
  designated discount factor. The steering command is proportional to this weighted mean error... The motor
  command is proportional to the difference between this reference speed and the current speed of the
  vehicle." (Section III-A, "Data Collection & Preparation," illustrated in Figure 2). It does not use
  lidar — only map, odometry, and waypoints.
- Scale: "enabling the collection of comprehensive data comprising observations, actions, reward feedback,
  and instances of termination/truncation across **100 episodes**" on the Austria track (Section III-A).
  No exact transition/timestep count is given (only "100 episodes").
- For the multi-track generalization experiment (Section IV-C): the dataset is "100 'expert' demonstrations
  in each of the racetracks – Austria, Barcelona, and Treitlstrasse" (i.e., 300 episodes total, 100 per
  track).
- Simulator: F1TENTH Gym API / `racecar_gym`, built on PyBullet (Section I, citing [14],[15],[16]).
- Action space: continuous motor thrust and steering rate in [-1, 1] (Section III-B.2). Reward: progress-
  based, `r(t) = -c` (c=2) on collision (episode terminates), else `|p_t - p_{t-1}|` (Section III-B.3,
  Eq. 2); a clean lap accumulates total reward ≈ 100%.

**Offline RL algorithm(s) used**
- Their own primary method: **Return-Conditioned Decision Tree Policy (RCDTP)**, a deterministic ensemble
  of XGBoost-style gradient-boosted weak learners conditioned on state, return-to-go, and timestep
  (Section III-C.1, Eqs. 3–9), introduced in their earlier paper (Koirala & Fleming, arXiv:2401.11630).
- Comparison offline methods, all implemented via the **d3rlpy** library except RCDTP (Section I, Figure 1):
  **Decision Transformer (DT)**, **Diffusion Policy (DP)**, and four Q-learning-variant FCNN baselines:
  **IQL, CQL, AWAC, PLAS (with an added perturbation layer), TD3+BC** (Section III-C.4 gives one-paragraph
  descriptions of each).
- Online RL "non-baselines" evaluated for comparison: **SAC, PPO, PPO-LSTM** via Stable-Baselines3
  (Section IV-A, Figure 6).

**Quantitative offline-vs-online comparison**
- Online RL result: "attempts to train an agent online in the Austria environment proved unsuccessful, as
  evident in the training log (Figure 6) where all the agents fail to surpass the **35% progress mark**."
  (Section IV-A). "SAC showed the most stable online training" among the three (Section V, Conclusions).
- Offline RL result on the same track: "Despite being exposed to only 100 episodes of expert demonstration
  only on the Austria racetrack, the offline models — especially RCDTP — demonstrate competent performance
  on other unseen racetracks as well" (Section IV-B). Figure 7a shows offline agents reaching total-reward
  (≈progress) scores well above the ~35% online ceiling on several tracks (RCDTP reaches up to ~100% on
  Columbia, ~70% on Austria itself in the 1-track-training setting per the bar chart in Figure 7a — read
  from the plotted values, not a table).
- **No numeric lap-time table comparing offline algorithms to online RL is given anywhere in the paper.**
  The only quantitative comparison is: (a) the online training curves plateauing at ~35% progress
  (Figure 6, a training-step-vs-progress line plot, not a table) vs. (b) offline agents' post-training
  total-reward bar charts (Figure 7a/b/c, again a bar chart, not a table). Table I in the paper is a
  **training-time** comparison only (CPU/GPU seconds by algorithm and track), not a performance comparison.
- Additional context: Austria racetrack performance for all methods (online and offline) is throttled by
  "a challenging right turn when the racecar attains approximately 33-35% of the maximum progress" —
  suggesting the online ~35% ceiling and part of the offline models' Austria-specific plateau share a
  common track-geometry cause, not purely an algorithmic one (Section IV-B).

### Gaps
- No transition/frame count for the 100-episode dataset(s) is stated anywhere — only episode counts. If
  the wiring-mechanism ticket needs a transitions-per-second or total-frame figure, it isn't in this paper
  and would have to be estimated from simulator step rate × episode length (not given precisely either).
- No numeric lap-time or completion-percentage table exists comparing offline algorithms to online RL —
  this is confirmed absent, not merely hard to extract, contrary to a possible hope that a table exists.
  The comparison is only visual (Figures 6 and 7).
- The exact reward/progress values underlying Figure 7's bars were not transcribed here (they are bar-chart
  pixel values, not a printed table), so any specific "RCDTP scores X% vs. online's 35%" figure beyond what
  is stated in prose above should be treated as approximate/visual, not a verbatim number from text.

---

## 3. arXiv:2501.17311 — "RLPP: A Residual Method for Zero-Shot Real-World Autonomous Racing on Scaled Platforms"

Ghignone, Baumann, Hu, Wang, Xie, Carron, Magno (ETH Zürich / Zhejiang University). v2, 6 Feb 2025. Accepted
ICRA 2025.

### Cited findings

**Residual-policy formulation**
- Core equation, Section II-B ("RLPP Architecture"), Eq. (1): **u = u_PP + u_RL** — simple additive
  combination of the baseline Pure Pursuit action and the RL residual action, both in the
  `[steering angle δ, velocity v]` action space.
- PP component: standard geometric Pure Pursuit law, Eq. (2): `δ = arctan(2·l_wb·p_x / d_la²)`, with
  velocity feed-forwarded from the minimum-curvature-optimizer's reference profile, scaled by a tunable
  gain: `v_PP = α_v · v_ref`.
- RL component: `u_RL = α_RL · u_NN`, where `u_NN` is the raw two-dimensional output of the trained policy
  network and `α_RL` is a scalar tuning factor (Section II-B). **`α_RL` is the single parameter re-tuned
  for sim-to-real transfer**, without retraining: "the final value of α_RL is set to 0.55. Such amount is
  the larger amount that allows for 10 uninterrupted laps while improving the lap time when compared to
  the baseline PP algorithm... RLPP allows for simple tuning of one single parameter to alleviate the
  Sim-to-Real gap, as opposed to complete retraining with different physics parameters." (Section III-A).
- Observation space, Eq. (18), Section II-C.3: `o_RL = [d, Δψ, v_x, v_y, r, o_traj]`, a 5+6N-dimensional
  vector (N=20 lookahead waypoints × [ref point, left-bound point, right-bound point], each 2D) — lateral
  deviation, heading error, body-frame velocities, yaw rate, plus track-boundary/reference-line geometry
  ahead of the car.
- Vehicle model in the training simulator: single-track bicycle model with Pacejka tire model (Eqs. 3–11,
  Table II gives all tire/vehicle constants: m=3.56 kg, I_z=0.0627 kg·m², l_f=0.174 m, l_r=0.151 m,
  μ=0.5, Pacejka B/C/D/E front and rear).

**Training details**
- RL algorithm: **Soft Actor-Critic (SAC)** via Stable-Baselines3 (Section II-C.4).
- Network: MLP, **2 hidden layers of 256 neurons each**; learning rate **3×10⁻⁴**; replay buffer size
  **1×10⁶**; all other SAC hyperparameters left at Stable-Baselines3 defaults.
- Training length: **2×10⁶ simulated steps**, "corresponding to shortly less than 4h on a workstation
  equipped with an Intel i7-13700K CPU and NVIDIA RTX 3090 GPU" (Section II-C.4).
- Domain randomization: tire friction coefficient μ perturbed with additive Gaussian noise
  `N(0, 0.15)` per episode (Section II-C.1).
- Curriculum on initial velocity (explicitly called out as "taking inspiration from Curriculum RL"):
  "the initial velocity for the agent is randomly sampled at the beginning of every training episode
  around the average velocity of the previous episode... a Gaussian centered around the average velocity
  and with a 0.5 m/s standard deviation." (Section II-C.4).
- Reward (Section II-C.2, Eqs. 12–17): positive terms `r_adv` (progress-normalized) + `r_speed`
  (speed-normalized); negative terms for lateral deviation (`r_dev`, thresholded past `τ_dev=0.1`,
  weight `α_dev=1`) and heading error (`r_heading`, threshold `τ_ψ=0`, weight `α_heading=0.25`, Table III),
  plus a collision penalty; deviation/heading penalties are additionally scaled by the *positive* reward
  term `r_pos` each step (Eq. 17), specifically to avoid over-penalizing early exploration.
- Real-car tuning: final shared PP parameters `d_la = 1.2`, `α_v = 0.8` (real); simulator-side PP velocity
  gain reduced to `α_v = 0.75` for stable initial training (Section III-A).

**Headline numbers — exact source**
- **+6.37% lap-time improvement**: Table IV, Section III-B. This is specifically the **minimum**-lap-time
  (`t_min`) comparison: PP `t_min = 14.28 s` vs. RLPP `t_min = 13.37 s` → 6.37% improvement. (The
  mean-lap-time improvement is smaller and separately stated: "comparing the proposed RLPP to the baseline
  PP controller our method improves lap time by 5.23%, going from 14.35 s to 13.60 s" — also Table IV /
  Section III-B, mean `t̄`.) The paper explicitly argues `t_min` is "arguably an even more important metric
  in AR compared to average lap time," which is why the abstract headlines the 6.37% figure rather than
  5.23%.
- **>52% SOTA-gap closure**: Section III-B, prose immediately following Table IV: "RLPP closes the gap by
  **52.90%** against MAP and by **53.50%** against MPC" — computed on the `t_min` metric, i.e., how much of
  the (PP → tire-model-aware SOTA) lap-time gap RLPP recovers. MAP `t_min=12.56 s`, MPC `t_min=12.58 s`
  (also Table IV). The abstract's ">52%" is the more conservative of the two (the MAP figure).
- **8x sim-to-real gap reduction**: Table V, Section III-C ("Sim-to-Real Gap"). Sim-to-real gap is defined
  as "the percentual difference of the simulation time with respect to the real-time." Values: TC-Driver
  17.511%, PP 4.207%, **RLPP 2.115%**. 17.511 / 2.115 ≈ 8.28×, matching "reducing the performance gap from
  simulation to reality by more than 8-fold when compared to the baseline RL controller" (TC-Driver is the
  "baseline RL controller" referred to — it is the prior trajectory-conditioned RL agent used as the RL
  point of comparison throughout the paper, see Section II-D.I).
- Full lap-time table (Table IV, 10 consecutive real-world laps per controller): TC-Driver `t̄=22.43 s`
  (only 4 laps completed, could not do 10), PP `t̄=14.35 s`, RLPP `t̄=13.60 s`, MAP `t̄=12.60 s`,
  MPC `t̄=12.75 s`. Compute times (`t̄_CPU`): PP 1.68 ms, RLPP 8.38 ms, MAP 1.82 ms, MPC 11.23 ms — all well
  under the 40 Hz (25 ms) control-loop budget.

### Gaps
- None of the three headline numbers were unverifiable — all three were traced to specific tables/sections
  as above and the arithmetic checks out from the raw table values, so there is nothing outstanding here.
- The paper does not report training wall-clock variance across seeds, nor how many random seeds were used
  for the single reported SAC training run — if the wiring-mechanism ticket wants variance/robustness data
  for RLPP's training procedure, it isn't in this paper.

---

## Implications for the MPC-teacher wiring-mechanism ticket

Only one of the three papers (2608.12063) is a direct precedent for "MPC-generated demonstrations seeding
an off-policy replay buffer with a phase-out curriculum," and its concrete numbers translate cleanly: a
flat 50%-of-batch expert-transition replacement rate held constant until a binary, success-rate-triggered
cutoff (10% empirical success), with no architecture change required on the RL side — the entire mechanism
lives in the data pipeline (buffer composition + curriculum), not the network. That is directly actionable
for wiring a Phase 4 MPC/min-curvature solver into a TD3-style (or similar off-policy) trainer here: mix
expert transitions in the replay buffer at a fixed ratio during early training and cut them off hard once
the agent clears a chosen success threshold, rather than designing any bespoke demo-aware network head.
The F1TENTH offline-RL paper (2408.04198) is a *weaker* precedent than the original report implied — it is
pure offline RL (train once on a static waypoint-controller dataset, no online fine-tuning, no replay-buffer
mixing), and its offline-vs-online comparison is qualitative (progress plateaus/bar charts) with no lap-time
table, so it mainly supports the general claim "a suboptimal classical controller's rollouts are enough to
bootstrap a competent racing policy," not the mixing-ratio/curriculum mechanics needed for the wiring
ticket. RLPP (2501.17311) is architecturally the cleanest and most racing-relevant of the three but
describes a different mechanism entirely — a residual policy added at *inference* time to a running
classical controller, not a teacher used to bootstrap an off-policy learner's replay buffer — so it is a
candidate alternative design (residual-on-top-of-MPC) rather than evidence for the specific "MPC as
replay-buffer teacher" mechanism the ticket is scoping; its one directly reusable number for this repo is
the single-parameter (`α_RL`) sim-to-real retuning trick, which could be a fallback plan if the ratio/
curriculum approach from 2608.12063 proves too data-hungry for this project's simulator.
