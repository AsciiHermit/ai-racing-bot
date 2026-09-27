# Problem Definition: Cross-Track Generalization in Multi-Agent Racing RL

*Chapter 1 of the ai-racing-bot open research foundation*

## 1. Use Case

We are building **ai-racing-bot**: a lightweight, open-source, 2D, physics-based racing simulator, exposed as a Gymnasium-compatible RL environment, with a deliberately modular interface (separate `VehiclePhysics`, `Track`, and `Sensor` contracts so pieces can be swapped independently). It is meant to be a small, self-contained chapter inside a larger open-source RL-research foundation — not a monolithic product, but a focused, citable piece of infrastructure plus the research result it enables.

As it stands today, the repo already has the scaffolding: the modular interfaces, a Gymnasium environment (`ARS-Racing-v0`), a working sensor stack (GPS, IMU, lidar, proprioceptive, track-relative pose), a pygame visualizer, and a setup dashboard. The vehicle physics is currently a simplified kinematic-bicycle stub with a hard cornering cap, and there is one fixed oval track — both explicitly flagged in the repo's own documentation as placeholders. *(Repo state read via an automated fetch of its README and PHYSICS.md on 2026-09-26, not a full clone; verify against the live repo.)*

The target capability we want this chapter to demonstrate is narrow and specific: **a driving policy, trained with reinforcement learning on this lightweight simulator, that performs well on race tracks it has never seen — without being retrained for each new track.** That single capability, not a general-purpose racing platform, is the use case this problem definition is scoped to.

## 2. The Gap in the Field

An earlier research pass across the autonomous-racing-RL literature (simulators, algorithms, vehicle dynamics, sensors, strategy, multi-agent work, and competitions) converged on one specific, well-evidenced gap that matches this use case. It is worth stating precisely what *is* already solved, so the gap isn't overstated:

- **Finding the best line on one known track is already solved, cheaply, without RL.** Open, lightweight, Gymnasium-compatible environments — F1TENTH gym (O'Kelly, Zheng, Karthik & Mangharam, *F1TENTH: An Open-source Evaluation Environment for Continuous Control and Reinforcement Learning*, NeurIPS 2019 Competition Track, PMLR 123, 2020), MicroRacer (Asperti & Del Brutto, arXiv:2203.10494, LOD 2022), and Gymnasium's CarRacing-v3 — already provide fast, RL-ready racing tasks with published PPO/SAC/TD3 baselines. And for a single track whose geometry and vehicle model are known in advance, classical trajectory optimization (minimum-curvature or MPC solvers) computes the time-optimal line directly, in seconds, with no training data at all. RL adds cost, not capability, on that narrow task.
- **What is not solved is generalization to tracks the policy has never seen.** The field's own synthesizing survey — Betz, Zheng, Liniger, Rosolia, Karle, Behl, Krovi & Mangharam, *Autonomous Vehicles on the Edge: A Survey on Autonomous Vehicle Racing*, IEEE Open Journal of Intelligent Transportation Systems, 3:458–488, 2022 (DOI 10.1109/OJITS.2022.3181510; arXiv:2202.07008) — names low generalizability to unseen tracks, and the large data volumes end-to-end/RL methods need, as core open weaknesses of the field, alongside the difficulty of learning nonlinear vehicle and tire dynamics from data.
- **The leading mitigation — model-based RL — is an active research direction, not a closed one.** Brunnbauer et al., *Model-based versus Model-free Deep Reinforcement Learning for Autonomous Racing Cars* (arXiv:2103.04909), show that a Dreamer-style, latent-imagination agent improves both sample efficiency and zero-shot transfer to unseen tracks over model-free RL, specifically on F1TENTH — but the authors frame this as an open, ongoing direction, and note explicitly that the degree of generalization depends strongly on which observation model is used. That sensitivity is itself unresolved.

**In short: RL solving a single known track is a solved benchmark problem. RL generalizing across unseen track geometries — reliably, cheaply, in a lightweight open environment — is not.** That is the gap this use case targets, and only that gap; it is stated as a single, focused claim rather than a bundle of loosely related open problems.

## 3. Correlation: Why This Use Case Sits in This Gap

The connection is not incidental — the specific design choices already made in this project are exactly the properties a generalization study needs, and the gap is exactly what a project with these properties is positioned to address:

| Property of the gap (from §2) | Property of this use case |
|---|---|
| Needs many distinct track geometries, not one fixed track | The environment already generates tracks procedurally rather than shipping one hand-built map |
| Needs the vehicle-dynamics fidelity to be a controlled, swappable variable (so a generalization result isn't confounded by an artificial physics ceiling) | Physics is already isolated behind a `VehiclePhysics` protocol, separate from the environment and track modules |
| Needs the observation model to be a controlled, swappable variable (per Brunnbauer et al.'s finding that generalization is sensitive to it) | Sensors are already isolated behind a `Sensor` protocol, with several modalities (lidar, GPS, IMU, proprioceptive) already implemented independently |
| Needs to be cheap enough to run many track × physics × algorithm × seed combinations | The project is explicitly scoped as 2D and lightweight, not a 3D or photorealistic simulator |
| Needs the result to be reproducible and comparable against a known-optimal reference, not just "the agent looks like it's driving well" | The project already targets a Gymnasium-standard interface, compatible with standard RL libraries and evaluation tooling |

The existing modular architecture was originally motivated by team ownership (each module built by a different person), but it happens to be precisely the structure a rigorous generalization benchmark requires: track distribution, physics fidelity, and observation modality all need to vary independently and be swapped without rebuilding the rest of the stack. That correlation — between a division-of-labor architecture and a controlled-ablation research design — is what makes this a natural, not forced, fit for the gap identified in §2.

## 4. Problem Statement

> **Given a lightweight, physics-credible 2D racing environment, can a reinforcement-learning-trained driving policy generalize — without per-track retraining — to previously unseen track geometries, approaching the lap-time performance of a per-track classical trajectory optimizer, while requiring substantially less training data than standard model-free baselines (PPO/SAC/TD3) need to reach comparable in-distribution performance?**

This is the single question this chapter of the foundation exists to answer. It is deliberately narrower than the original project vision (multi-agent racing, race strategy, full 4-wheel vehicle dynamics) — not because those are uninteresting, but because §2 and §3 show this is the one question where a genuine, well-evidenced gap exists *and* where the project's current design already correlates directly with what's needed to answer it.

## 5. Boundaries of This Chapter

This document defines the problem only — it does not claim to solve multi-agent racing tactics (overtaking, blocking, defending; cf. Sony AI's GT Sophy, Wurman et al., *Nature* 602:223–228, 2022) or race strategy (fuel, tire wear, pit-stop timing). Both are real, separately-evidenced gaps in the same literature, but they are distinct problems from cross-track generalization and are left for later chapters of the same foundation, built once this one has a clear answer. Folding them in here would dilute a single, falsifiable problem statement into an unfocused one.

## 6. Sources

- Betz, J., Zheng, H., Liniger, A., Rosolia, U., Karle, P., Behl, M., Krovi, V., & Mangharam, R. (2022). Autonomous Vehicles on the Edge: A Survey on Autonomous Vehicle Racing. *IEEE Open Journal of Intelligent Transportation Systems*, 3, 458–488. https://doi.org/10.1109/OJITS.2022.3181510 (also arXiv:2202.07008)
- Brunnbauer, A., et al. (2021). Model-based versus Model-free Deep Reinforcement Learning for Autonomous Racing Cars. arXiv:2103.04909. *(A related/shortened version reportedly appeared as "Latent Imagination Facilitates Zero-Shot Transfer in Autonomous Racing" at ICRA 2022 — not independently re-verified in this session; confirm before citing formally.)*
- O'Kelly, M., Zheng, H., Karthik, D., & Mangharam, R. (2020). F1TENTH: An Open-source Evaluation Environment for Continuous Control and Reinforcement Learning. *Proceedings of Machine Learning Research* (NeurIPS 2019 Competition and Demonstration Track), 123.
- Asperti, A., & Del Brutto, M. (2022). MicroRacer: a didactic environment for Deep Reinforcement Learning. arXiv:2203.10494 (also LOD 2022 proceedings).
- Gymnasium CarRacing documentation: https://gymnasium.farama.org/environments/box2d/car_racing/
- Wurman, P. R., et al. (2022). Outracing champion Gran Turismo drivers with deep reinforcement learning. *Nature*, 602, 223–228. https://doi.org/10.1038/s41586-021-04357-7 *(cited in §5 only, to acknowledge the multi-agent-tactics gap that this chapter explicitly excludes)*
- Repository under discussion: https://github.com/AsciiHermit/ai-racing-bot *(state as read on 2026-09-26 via an automated fetch of its README and PHYSICS.md, not a full clone — verify against the live repo, which will have changed by the time this is read)*
