"""Phase 4: a minimal, from-scratch minimum-curvature racing-line solver
and quasi-steady-state lap-time estimator (wayfinder ticket #16,
github.com/AsciiHermit/ai-racing-bot/issues/16 -- adapting TUM's
global_racetrajectory_optimization was rejected there in favor of building
this directly against the project's own spline Track representation, since
TUM's dependency breakage was found unresolved upstream).

Minimum-curvature formulation: the racing line is the centerline offset
laterally by n_i (bounded by the track corridor) at each of N fixed
arc-length samples. Writing each point as P_i = C_i + n_i * N_i (C =
centerline point, N = unit normal), the discrete second-difference
P_{i+1} - 2*P_i + P_{i-1} -- a standard curvature proxy when samples are
roughly evenly spaced -- is affine in the vector of unknowns n, so
minimizing the sum of its squared magnitude subject to box bounds on each
n_i is a bounded linear least-squares problem, solved directly with
scipy.optimize.lsq_linear. No general QP library needed.

Lap time: curvature-limited cornering speed per point (friction circle, no
aero -- same assumption as PHYSICS.md's tuning), refined by a couple of
forward (acceleration-limited) and backward (braking-limited) passes
around the closed loop -- the standard quasi-steady-state technique,
without claiming full multi-pass convergence (this is explicitly the
"minimal" solver ticket #16 chose over adapting a full-featured existing
one).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np
from scipy.optimize import lsq_linear

from ars.core.interfaces import Track

G = 9.81  # m/s^2

# Physically-grounded defaults, matching Phase 1's DynamicBicycleParams so
# the reference lap time reflects the same vehicle the RL agents drive.
DEFAULT_MU = 1.6
DEFAULT_TOP_SPEED = 95.0  # m/s
DEFAULT_MAX_ACCEL = 9_576.0 / 798.0  # m/s^2 (DynamicBicycleParams.max_engine_force / mass)
DEFAULT_MAX_DECEL = 39_146.0 / 798.0  # m/s^2 (DynamicBicycleParams.max_brake_force / mass)


@dataclass
class RacingLine:
    s_values: list[float]  # arc-length position on the original centerline parameterization
    lateral_offsets: list[float]  # n_i, optimized offset from centerline
    x: list[float] = field(repr=False)
    y: list[float] = field(repr=False)
    curvatures: list[float] = field(repr=False)  # curvature of the OPTIMIZED line, not the centerline


def _sample_centerline(track: Track, sample_spacing_m: float):
    num_samples = max(4, round(track.length / sample_spacing_m))
    s_values = [i * track.length / num_samples for i in range(num_samples)]
    samples = [track.sample_at_s(s) for s in s_values]
    cx = np.array([s.centerline_x for s in samples])
    cy = np.array([s.centerline_y for s in samples])
    heading = np.array([s.heading for s in samples])
    half_width = np.array([s.width / 2.0 for s in samples])
    nx = -np.sin(heading)
    ny = np.cos(heading)
    return s_values, cx, cy, nx, ny, half_width


def optimize_racing_line(track: Track, margin: float = 0.5, sample_spacing_m: float = 2.0) -> RacingLine:
    s_values, cx, cy, nx, ny, half_width = _sample_centerline(track, sample_spacing_m)
    n_points = len(s_values)

    # Second-difference operator D (n_points x n_points, circulant for a
    # closed loop): (D @ n)_i = n_{i+1} - 2*n_i + n_{i-1}.
    D = -2.0 * np.eye(n_points) + np.eye(n_points, k=1) + np.eye(n_points, k=-1)
    D[0, -1] = 1.0  # wrap: i=0's "i-1" is the last point
    D[-1, 0] = 1.0  # wrap: i=N-1's "i+1" is the first point

    # residual = A @ n + b, stacked for x and y components. Column j of D
    # is nonzero only at rows j-1, j, j+1; scaling column-wise by nx (numpy
    # broadcasts nx as a row against D's columns) gives exactly
    # a_x[i, j] = D[i, j] * nx[j], i.e. row i's coefficient on n_j.
    a_x = D * nx
    a_y = D * ny
    b_x = D @ cx
    b_y = D @ cy

    A = np.vstack([a_x, a_y])
    b = np.concatenate([b_x, b_y])

    lb = -half_width + margin
    ub = half_width - margin
    # Degenerate/very narrow track: collapse the corridor to a single point
    # (the centerline) rather than handing lsq_linear an infeasible lb>ub.
    lb, ub = np.minimum(lb, ub), np.maximum(lb, ub)

    result = lsq_linear(A, -b, bounds=(lb, ub))
    n_opt = result.x

    opt_x = cx + n_opt * nx
    opt_y = cy + n_opt * ny
    curvatures = _discrete_curvature(opt_x, opt_y)

    return RacingLine(
        s_values=list(s_values),
        lateral_offsets=list(n_opt),
        x=list(opt_x),
        y=list(opt_y),
        curvatures=list(curvatures),
    )


def _second_difference_cost(x: np.ndarray, y: np.ndarray) -> float:
    """Sum of squared second-differences -- exactly what optimize_racing_line
    minimizes. Exposed for testing: what's guaranteed by construction is
    that the optimizer's own objective improves, not any other curvature
    proxy evaluated after the fact (see test_min_curvature.py)."""
    n = len(x)
    x_prev, x_next = np.roll(x, 1), np.roll(x, -1)
    y_prev, y_next = np.roll(y, 1), np.roll(y, -1)
    rx = x_next - 2.0 * x + x_prev
    ry = y_next - 2.0 * y + y_prev
    return float(np.sum(rx**2 + ry**2))


def _discrete_curvature(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Menger curvature at each point of a closed polyline: kappa_i =
    2 * cross(P_i-P_{i-1}, P_{i+1}-P_i) / (|P_i-P_{i-1}| * |P_{i+1}-P_i| * |P_{i+1}-P_{i-1}|).
    Signed; magnitude is what matters for cornering-speed limits."""
    x_prev, y_prev = np.roll(x, 1), np.roll(y, 1)
    x_next, y_next = np.roll(x, -1), np.roll(y, -1)

    d1x, d1y = x - x_prev, y - y_prev
    d2x, d2y = x_next - x, y_next - y
    d3x, d3y = x_next - x_prev, y_next - y_prev

    cross = d1x * d2y - d1y * d2x
    len1 = np.hypot(d1x, d1y)
    len2 = np.hypot(d2x, d2y)
    len3 = np.hypot(d3x, d3y)
    denom = len1 * len2 * len3
    with np.errstate(divide="ignore", invalid="ignore"):
        curvature = np.where(denom > 1e-9, 2.0 * cross / denom, 0.0)
    return curvature


def estimate_lap_time(
    track: Track,
    mu: float = DEFAULT_MU,
    top_speed: float = DEFAULT_TOP_SPEED,
    max_accel: float = DEFAULT_MAX_ACCEL,
    max_decel: float = DEFAULT_MAX_DECEL,
    margin: float = 0.5,
    sample_spacing_m: float = 2.0,
    smoothing_passes: int = 2,
) -> float:
    line = optimize_racing_line(track, margin=margin, sample_spacing_m=sample_spacing_m)
    x = np.array(line.x)
    y = np.array(line.y)
    kappa = np.abs(np.array(line.curvatures))
    n = len(x)

    ds = np.hypot(np.roll(x, -1) - x, np.roll(y, -1) - y)  # segment length i -> i+1

    with np.errstate(divide="ignore"):
        v_limit = np.minimum(top_speed, np.sqrt(mu * G / np.maximum(kappa, 1e-9)))

    v = v_limit.copy()
    for _ in range(smoothing_passes):
        # Forward pass: acceleration-limited ramp-up, wrapping around the
        # closed loop so entry speed into point 0 respects the previous
        # point's speed too.
        for i in range(n):
            prev = (i - 1) % n
            v[i] = min(v[i], math.sqrt(v[prev] ** 2 + 2 * max_accel * ds[prev]))
        # Backward pass: braking-limited ramp-down before corners.
        for i in reversed(range(n)):
            nxt = (i + 1) % n
            v[i] = min(v[i], math.sqrt(v[nxt] ** 2 + 2 * max_decel * ds[i]))

    v_avg = 0.5 * (v + np.roll(v, -1))
    lap_time = float(np.sum(ds / np.maximum(v_avg, 1e-6)))
    return lap_time
