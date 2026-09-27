import numpy as np

from ars.env.baseline_factory import make_baseline_env
from ars.env.racing_env import _progress_delta

# Regression test for a real bug found evaluating a trained SAC Step A
# baseline (see the Phase 3 training session): a near-stationary vehicle
# sitting right at the start/finish seam (x oscillating in the
# sub-millimeter range across x=0) flips which FixedLoopTrack segment
# Track.query() projects onto, producing a raw arc-length jump from ~0 to
# ~track_length and back -- even though the wrapped progress delta
# (_progress_delta, what the reward actually uses) is correctly tiny both
# times. The old lap-crossing check re-derived "did we cross the finish
# line" from the *raw*, unwrapped s difference instead of reusing that
# already-correct wrapped value, and misfired: one run reported 81 laps in
# 40 seconds of sim time, physically impossible at the vehicle's max speed.
#
# Exact reproducing sequence, captured from the real trained model's
# trajectory (see PHASE0_SPIKE_NOTE.md-style debugging in this session):
_SEAM_FLICKER_S_SEQUENCE = [0.00030171809692687226, 6.811754545073037e-05, 317.0794177476622, 0.004313756028467306]
_TRACK_LENGTH = 317.0794177476622


def test_seam_flicker_sequence_does_not_accumulate_a_lap():
    # Direct unit test of the fixed formula (cumulative wrapped progress,
    # floor-divided by track length) against the exact numbers that broke
    # the old per-step raw-diff check -- the precise regression test for
    # this bug, independent of reproducing the full RL trajectory that
    # originally surfaced it.
    cumulative_progress = 0.0
    lap = 0
    prev_s = 0.0
    for s in _SEAM_FLICKER_S_SEQUENCE:
        ds = _progress_delta(prev_s, s, _TRACK_LENGTH)
        cumulative_progress += ds
        lap = max(0, int(cumulative_progress // _TRACK_LENGTH))
        prev_s = s
    assert lap == 0


def test_stationary_noise_at_the_seam_does_not_accumulate_laps():
    # General robustness sanity check (not a precise repro of the seam-
    # flicker bug above -- this specific action pattern doesn't happen to
    # trigger it, but a near-stationary agent shouldn't accumulate laps
    # under any noise pattern).
    env = make_baseline_env(off_track_terminates=False, max_episode_steps=300)
    env.reset()
    max_lap = 0
    for i in range(300):
        steer = 1.0 if i % 2 == 0 else -1.0
        action = np.array([0.05, 0.0, steer], dtype=np.float32)
        _, _, _, _, info = env.step(action)
        max_lap = max(max_lap, info["lap"])
    assert max_lap == 0


def test_genuine_forward_driving_still_completes_laps():
    # Same track as the existing test_lap_increments_on_wraparound
    # regression test in tests/env/test_racing_env.py -- confirms the fix
    # doesn't regress real lap completion, just noise-driven miscounting.
    from ars.env.factory import make_default_env
    from ars.physics import KinematicBicyclePhysics

    env = make_default_env(
        physics=KinematicBicyclePhysics(), off_track_terminates=False, max_episode_steps=5000
    )
    env.reset()
    action = np.array([1.0, 0.0, 0.05], dtype=np.float32)
    info = {}
    for _ in range(5000):
        _, _, _, _, info = env.step(action)
    assert info["lap"] >= 1
