"""Headless render smoke test: uses SDL's dummy video driver so this runs
without an actual display (CI, no monitor). Verifies the viewer builds and
draw() doesn't crash -- not a pixel-level visual test."""
import math
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from ars.core.types import VehicleState
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


def test_car_rotation_matches_world_heading():
    # Regression test for a rotation-direction bug: at heading=+90deg (world
    # +y, "north"), the car's nose must end up on the smaller-screen-y side
    # of its own bounding box (i.e. toward the top of the screen, since
    # Camera.world_to_screen flips y). Detect the nose as the narrower-span
    # end of the sprite's opaque pixels (see test_car_sprite.py) and check
    # which half of the ROTATED bounding box it lands in.
    env = make_default_env()
    env.reset()
    viewer = LiveViewer(env.track, config=ViewerConfig(width=200, height=200, fps=0))
    try:
        state = VehicleState(
            x=env.vehicle_state.x, y=env.vehicle_state.y, heading=math.pi / 2, vx=0, vy=0, yaw_rate=0
        )
        viewer.draw(state)

        length_px = viewer.camera.scale(viewer.config.car_length)
        width_px = viewer.camera.scale(viewer.config.car_width)
        sprite = viewer._car_sprite.get_scaled(viewer.pygame, length_px, width_px)
        rotated = viewer.pygame.transform.rotate(sprite, math.degrees(state.heading))
        rw, rh = rotated.get_size()

        def opaque_span(y):
            xs = [x for x in range(rw) if rotated.get_at((x, y)).a > 0]
            return (max(xs) - min(xs)) if xs else 999

        spans = [(y, opaque_span(y)) for y in range(0, rh, 3)]
        narrowest_y = min(spans, key=lambda t: t[1])[0]
        assert narrowest_y < rh / 2, "expected the nose (narrowest span) in the top half after a +90deg heading"
    finally:
        viewer.close()
