"""DashboardApp: owns the single persistent pygame window and steps the
user through the 5-screen flow (Track -> Vehicle -> Physics -> Agent ->
Simulation). Each screen is swapped in-place; SessionConfig persists
across the whole session so Back doesn't lose earlier choices.
"""
from __future__ import annotations

from ars.dashboard.config import SessionConfig
from ars.dashboard.widgets import COLOR_ACCENT, COLOR_PANEL_BORDER, Button, Rect

WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 720
NAV_BAR_HEIGHT = 70
STEP_BAR_HEIGHT = 50
COLOR_BG = (18, 19, 23)
COLOR_STEP_INACTIVE = (70, 75, 84)
COLOR_STEP_DONE = (110, 180, 120)


class DashboardApp:
    def __init__(self):
        import pygame

        self.pygame = pygame
        pygame.init()
        pygame.display.set_caption("ARS -- Simulator Setup")
        self.screen_surface = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()
        self.title_font = pygame.font.SysFont("consolas", 26, bold=True)
        self.label_font = pygame.font.SysFont("consolas", 14)
        self.body_font = pygame.font.SysFont("consolas", 16)
        self.small_font = pygame.font.SysFont("consolas", 12)

        self.config = SessionConfig()
        self.running = True

        # Imported here (not module top) to avoid a circular import between
        # app.py and the step screens, which import Screen from this pkg.
        from ars.dashboard.steps.step1_track import TrackScreen
        from ars.dashboard.steps.step2_vehicle import VehicleScreen
        from ars.dashboard.steps.step3_physics import PhysicsScreen
        from ars.dashboard.steps.step4_agent import AgentScreen
        from ars.dashboard.steps.step5_simulation import SimulationScreen

        self.steps = [
            TrackScreen(self),
            VehicleScreen(self),
            PhysicsScreen(self),
            AgentScreen(self),
            SimulationScreen(self),
        ]
        self.step_index = 0

        self._back_button = Button(Rect(30, WINDOW_HEIGHT - 55, 120, 36), "< Back", self._go_back)
        self._next_button = Button(Rect(WINDOW_WIDTH - 150, WINDOW_HEIGHT - 55, 120, 36), "Next >", self._go_next)

        self.steps[self.step_index].on_enter()

    @property
    def current_screen(self):
        return self.steps[self.step_index]

    def restore_window(self) -> None:
        """Re-claim the shared pygame display after a nested view (e.g.
        LiveViewer, launched from Step 5) resized/retitled the one window
        pygame allows. Call after returning from such a nested loop."""
        self.pygame.display.set_caption("ARS -- Simulator Setup")
        self.screen_surface = self.pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))

    def _go_back(self) -> None:
        if self.step_index > 0:
            self.step_index -= 1
            self.current_screen.on_enter()

    def _go_next(self) -> None:
        if self.step_index < len(self.steps) - 1 and self.current_screen.can_advance():
            self.step_index += 1
            self.current_screen.on_enter()

    def run(self) -> None:
        while self.running:
            for event in self.pygame.event.get():
                if event.type == self.pygame.QUIT:
                    self.running = False
                    continue
                self._back_button.handle_event(event, self.pygame)
                self._next_button.handle_event(event, self.pygame)
                self.current_screen.handle_event(event)

            self._back_button.enabled = self.step_index > 0
            self._next_button.enabled = (
                self.step_index < len(self.steps) - 1 and self.current_screen.can_advance()
            )

            self.screen_surface.fill(COLOR_BG)
            self._draw_step_bar()
            self.current_screen.draw(self.screen_surface)
            self._draw_nav_bar()

            self.pygame.display.flip()
            self.clock.tick(60)

        self.pygame.quit()

    def _draw_step_bar(self) -> None:
        pygame = self.pygame
        n = len(self.steps)
        margin = 40
        available = WINDOW_WIDTH - 2 * margin
        gap = available / n
        for i, step in enumerate(self.steps):
            cx = margin + gap * i + gap / 2
            color = COLOR_ACCENT if i == self.step_index else (COLOR_STEP_DONE if i < self.step_index else COLOR_STEP_INACTIVE)
            pygame.draw.circle(self.screen_surface, color, (int(cx), 26), 9)
            label = self.small_font.render(f"{i + 1}. {step.title}", True, color)
            label_rect = label.get_rect(center=(int(cx), 44))
            self.screen_surface.blit(label, label_rect)
            if i < n - 1:
                line_start = (int(cx) + 14, 26)
                line_end = (int(margin + gap * (i + 1) + gap / 2) - 14, 26)
                pygame.draw.line(self.screen_surface, COLOR_STEP_INACTIVE, line_start, line_end, width=2)

    def _draw_nav_bar(self) -> None:
        pygame = self.pygame
        pygame.draw.line(
            self.screen_surface, COLOR_PANEL_BORDER, (0, WINDOW_HEIGHT - NAV_BAR_HEIGHT), (WINDOW_WIDTH, WINDOW_HEIGHT - NAV_BAR_HEIGHT), width=1
        )
        self._back_button.draw(self.screen_surface, pygame, self.body_font)
        self._next_button.draw(self.screen_surface, pygame, self.body_font)


def main() -> None:
    DashboardApp().run()


if __name__ == "__main__":
    main()
