# Phase 0 Spike Note: Physics Go/No-Go

Per `IMPLEMENTATION_PLAN.md` Phase 0's exit criterion: a short internal note stating the go/no-go decision on physics difficulty, plus the two library/approach decisions.

## Method

- **Throwaway tire model**: `scripts/phase0_tire_spike.py`'s `SlipBicyclePhysics`, a single-track (bicycle) model with a linear tire law saturated by a friction-circle limit (`mu * Fz` per axle, static load split only, no dynamic load transfer, no aero). Explicitly throwaway, not reused by Phase 1's real `dynamic_bicycle.py`.
- **Controller**: the existing `DummyExpertAgent` (`ars/agents/dummy_expert.py`), unmodified. A reactive P-controller: steers toward the centerline based on current lateral offset and heading error, reduces throttle based on the *current* point's curvature. No anticipatory braking before a corner.
- **Track**: the existing `make_simple_oval()` fixed oval (two 80m straights, two 25m-radius 180-degree turns).
- Simulation: 60s at `dt=0.02s`, matching the codebase's existing convention (`ars/env/racing_env.py`).

## Bug found and fixed during the spike

The first version of the throwaway tire model had a slip-angle sign convention backwards (`steer_angle - atan2(...)` instead of the standard `atan2(...) - steer_angle`; see Rajamani, *Vehicle Dynamics and Control*), which made tire force reinforce a slide instead of opposing it: lateral velocity ran away toward hundreds of m/s once the car went off-line, a pure sign bug, not a realistic spin. Fixed to the standard convention; confirmed against a manual low-speed, safe-entry trajectory that the model now stays bounded and behaves sensibly.

**Worth carrying into Phase 1**: get the slip-angle sign right from the start, and never use a divide-by-zero guard value (needed only inside `atan2` at near-zero speed) inside the actual equations of motion, doing so injects a fake forcing term once speed settles near zero.

## Result

- Theoretical grip-limited cornering speed at the oval's 25m-radius turn (friction circle only, no aero): `sqrt(mu * g * r)` ≈ **19.8 m/s** (~71 km/h).
- The controller enters the turn at ≈**32 m/s**, about **61% over** the available grip.
- Outcome: understeers straight off the track within about half a second of turn-in, lateral offset grows unbounded (never recovers within the 60s window), **0 laps completed**, off-track for **92% of the simulated time**.

## Interpretation

A naive, purely reactive controller (no preview of upcoming curvature) clearly fails against this tire model: it cannot complete a single lap. This is well past "sub-optimal" and into "fails outright" (per the plan's own decision tree, either result confirms go: "if the PID controller struggles... confirms the task is non-trivial"). The physics is demanding enough that cross-track generalization is a real learning problem, not one a trivial controller already solves.

## Go/No-Go: **GO**

Proceed to Phase 1 (dynamic bicycle + Pacejka tire model) and Phase 2 (procedural track generator) as planned. No change to the plan's scope or sequencing.

## Library/approach decisions (confirmed, not re-opened)

- **RL library**: Stable-Baselines3, per `IMPLEMENTATION_PLAN.md`'s existing recommendation.
- **Track generation**: spline-based (MicroRacer-style), per `IMPLEMENTATION_PLAN.md`'s existing recommendation.

## What this spike deliberately does not cover

No load transfer, no aero, static weight split only, a linear (not Pacejka) tire curve. These stay explicitly deferred to Phase 1's real implementation, per the incremental path `PHYSICS.md` already lays out.
