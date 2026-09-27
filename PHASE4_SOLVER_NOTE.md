# Phase 4 Results: Classical Optimal-Control Reference

Per `IMPLEMENTATION_PLAN.md` Phase 4's exit criterion: "every track in both sets has a reference optimal lap time."

## Approach

Per wayfinder ticket #16, built a minimal from-scratch solver (`ars/solver/min_curvature.py`) rather than adapting TUM's `global_racetrajectory_optimization` (found unmaintained with unresolved dependency breakage in ticket #15's research):

- **Minimum-curvature line**: the racing line is the centerline offset laterally at N fixed arc-length samples, bounded by the track corridor (width, minus a safety margin). The discrete second-difference of the offset path is affine in the offsets, so minimizing its squared magnitude subject to the corridor bounds is a bounded linear least-squares problem — solved directly with `scipy.optimize.lsq_linear`, no general QP library needed.
- **Lap time**: curvature-limited cornering speed per point (friction circle, `mu=1.6`, no aero — same assumption Phase 1's tire model makes), refined by two rounds of forward (acceleration-limited) and backward (braking-limited) passes around the closed loop — the standard quasi-steady-state technique. `max_accel`/`max_decel` reuse Phase 1's `DynamicBicycleParams` force constants (`/mass`), so the reference reflects the same vehicle the RL agents drive.

## A real modeling limitation found during testing, not a bug

On a track with *uniform* curvature (a pure circle), the discrete second-difference objective has a genuine degeneracy: it pushes the line uniformly toward whichever edge of the corridor the proxy favors, and that proxy does not track literal geometric curvature (`1/radius`) in a simple monotonic way once the offset is a large fraction of the radius — confirmed by direct calculation, not assumed. This is an inherent property of this simplified proxy on constant-curvature tracks, not a coding error; it does not affect tracks with actually-varying curvature (the entire manifest), where the optimizer both reduces its own objective *and* reduces true discrete curvature through the bulk of each corner (see `tests/solver/test_min_curvature.py` for both properties, tested separately since they aren't always the same thing).

## Result

Ran the solver against all 110 manifest tracks (80 train + 15 interpolation + 15 extrapolation) via `scripts/generate_lap_time_references.py`. Every track got a finite, positive `reference_lap_time_s`, now stored directly on its spec in `data/tracks/manifest.json` for Phase 6 to read without re-solving. Range: 20.15s–37.20s, mean 27.39s, consistent with curvature-limited average speeds well below the 95 m/s top-speed cap (as expected — these are tracks with real turns, not ovals with long straights).

**Exit criterion met.**
