"""Headless render smoke test: uses SDL's dummy video driver so this runs
without an actual display (CI, no monitor). Verifies the viewer builds and
draw() doesn't crash -- not a pixel-level visual test."""
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from ars.env import make_default_env
from ars.viz import LiveViewer, ViewerConfig


def test_viewer_builds_and_draws_frame():
    env = make_default_env()
    env.reset()
    viewer = LiveViewer(env.track, config=ViewerConfig(width=320, height=240, fps=0))
    try:
        result = viewer.draw(env.vehicle_state)
        assert result is True
    finally:
        viewer.close()


def test_viewer_draws_multiple_frames_while_stepping():
    env = make_default_env(off_track_terminates=False)
    env.reset()
    viewer = LiveViewer(env.track, config=ViewerConfig(width=320, height=240, fps=0))
    try:
        import numpy as np

        action = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        for _ in range(20):
            env.step(action)
            assert viewer.draw(env.vehicle_state) is True
    finally:
        viewer.close()


def test_viewer_draws_with_wide_fov_sensor_overlay():
    env = make_default_env()
    env.reset()
    config = ViewerConfig(width=320, height=240, fps=0, lidar_range=25.0, lidar_fov=1.57)
    viewer = LiveViewer(env.track, config=config)
    try:
        assert viewer.draw(env.vehicle_state) is True
    finally:
        viewer.close()


def test_viewer_draws_with_sensor_overlay_disabled():
    env = make_default_env()
    env.reset()
    viewer = LiveViewer(env.track, config=ViewerConfig(width=320, height=240, fps=0, lidar_range=None))
    try:
        assert viewer.draw(env.vehicle_state) is True
    finally:
        viewer.close()
