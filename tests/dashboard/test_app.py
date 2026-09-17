"""Headless dashboard smoke tests -- SDL dummy driver, no real display.
Confirms every screen builds and draws without crashing, and that
navigation/config accumulation work across the whole 5-step flow."""
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from ars.dashboard import DashboardApp


def test_all_five_steps_build_and_draw():
    app = DashboardApp()
    assert [s.title for s in app.steps] == ["Track", "Vehicle", "Physics", "Agent", "Simulation"]
    for i in range(len(app.steps)):
        app.current_screen.draw(app.screen_surface)
        if i < len(app.steps) - 1:
            app._go_next()
    app.pygame.quit()


def test_back_and_next_navigate_without_losing_config():
    app = DashboardApp()
    app.config.vehicle.mass_kg = 1234.0
    app._go_next()
    app._go_next()
    app._go_back()
    assert app.config.vehicle.mass_kg == 1234.0
    app.pygame.quit()


def test_cannot_go_past_last_step():
    app = DashboardApp()
    for _ in range(10):
        app._go_next()
    assert app.step_index == len(app.steps) - 1
    app.pygame.quit()


def test_cannot_go_before_first_step():
    app = DashboardApp()
    app._go_back()
    assert app.step_index == 0
    app.pygame.quit()
