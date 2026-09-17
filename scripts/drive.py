"""Manually drive the default env with the keyboard, rendered live.

    python scripts/drive.py

Controls: arrow keys / WASD to throttle, brake, steer. Esc or close the
window to quit. Requires the `viz` extra (pip install -e ".[viz]").
"""
from __future__ import annotations

from ars.env import make_default_env
from ars.viz import LiveViewer
from ars.viz.keyboard_control import read_keyboard_action


def main() -> None:
    env = make_default_env(off_track_terminates=False, max_episode_steps=100_000)
    obs, info = env.reset()

    viewer = LiveViewer(env.track)
    pygame = viewer.pygame

    running = True
    while running:
        viewer.poll_events()
        if not viewer.is_open:
            break

        action = read_keyboard_action(pygame)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()

        viewer.set_hud(
            [
                f"speed: {env.vehicle_state.vx:5.1f} m/s",
                f"progress: {info['progress_s']:6.1f} / {env.track.length:.1f} m",
                f"lap: {info['lap']}",
                f"off_track: {info['off_track']}",
            ]
        )
        running = viewer.draw(env.vehicle_state)

    viewer.close()


if __name__ == "__main__":
    main()
