"""Step 5: Simulation. Read-only summary of everything chosen in Steps 1-4,
plus a Run button that builds the env + dummy expert agent from the
accumulated SessionConfig and launches the live viewer, agent-driven.
"""
from __future__ import annotations

from ars.dashboard.builder import build_env_and_agent
from ars.dashboard.screen import Screen
from ars.dashboard.widgets import COLOR_LABEL, COLOR_TITLE, Button, Rect


class SimulationScreen(Screen):
    title = "Simulation"

    def __init__(self, app):
        super().__init__(app)
        self._run_button = Button(Rect(60, 480, 220, 46), "RUN SIMULATION", self._run)
        self._status = ""

    def handle_event(self, event) -> None:
        self._run_button.handle_event(event, self.app.pygame)

    def draw(self, screen) -> None:
        title = self.app.title_font.render("Simulation Summary", True, COLOR_TITLE)
        screen.blit(title, (30, 60))

        v = self.config.vehicle
        lidar = self.config.lidar
        lines = [
            f"Track:     Simple Oval",
            f"Vehicle:   {v.mass_kg:.0f} kg, {v.length_m:.1f} x {v.width_m:.1f} m",
            f"Sensors:   Forward Lidar (range {lidar.max_range_m:.0f} m)",
            f"Physics:   {self.config.physics.model_name}",
            f"Agent:     Dummy Expert (line follower)",
        ]
        for i, line in enumerate(lines):
            surface = self.app.body_font.render(line, True, COLOR_LABEL)
            screen.blit(surface, (60, 130 + i * 30))

        self._run_button.draw(screen, self.app.pygame, self.app.body_font)

        if self._status:
            status_surface = self.app.label_font.render(self._status, True, COLOR_LABEL)
            screen.blit(status_surface, (60, 545))

        hint = self.app.small_font.render(
            "Opens a live view driven by the dummy expert. Esc or close its window to return here.",
            True,
            COLOR_LABEL,
        )
        screen.blit(hint, (60, 590))

    def _run(self) -> None:
        from ars.viz import LiveViewer, ViewerConfig

        env, agent = build_env_and_agent(self.config)
        env.reset()
        viewer = LiveViewer(env.track, config=ViewerConfig())

        running = True
        while running:
            viewer.poll_events()
            if not viewer.is_open:
                break
            action = agent.act(env.vehicle_state)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                env.reset()
            viewer.set_hud(
                [
                    f"speed: {env.vehicle_state.vx:5.1f} m/s",
                    f"progress: {info['progress_s']:6.1f} / {env.track.length:.1f} m",
                    f"lap: {info['lap']}",
                ]
            )
            running = viewer.draw(env.vehicle_state)

        viewer.close(quit_pygame=False)
        self.app.restore_window()
        self._status = "Simulation closed. Adjust settings and Run again, or go Back."
