"""Step 3: Physics. v1 has exactly one physics implementation (the
kinematic-bicycle stub -- real 4-wheel F1-style dynamics is still being
built, see ars.physics). No real selection or tunable params exist yet;
this screen is a placeholder so the step stays in the flow and the user
knows what they're getting, without pretending there's a choice to make.
"""
from __future__ import annotations

from ars.dashboard.screen import Screen
from ars.dashboard.widgets import COLOR_LABEL, COLOR_MEASUREMENT, COLOR_TITLE
from ars.physics import KinematicBicycleParams

G = 9.81


class PhysicsScreen(Screen):
    title = "Physics"

    def draw(self, screen) -> None:
        title = self.app.title_font.render("Physics", True, COLOR_TITLE)
        screen.blit(title, (30, 60))

        lines = [
            f"Model:  {self.config.physics.model_name}",
            "",
            "v1 has one physics model and no tunable parameters yet.",
            "Real 4-wheel F1-style dynamics (tire slip, load transfer,",
            "aero) is still being built -- this screen will become a",
            "real model picker + parameter form once that lands.",
        ]
        for i, line in enumerate(lines):
            color = COLOR_TITLE if i == 0 else COLOR_LABEL
            surface = self.app.body_font.render(line, True, color)
            screen.blit(surface, (60, 130 + i * 26))

        self._draw_reference_values(screen)

    def _draw_reference_values(self, screen) -> None:
        p = KinematicBicycleParams()
        header = self.app.body_font.render("REFERENCE VALUES (real F1-scale)", True, COLOR_TITLE)
        screen.blit(header, (60, 320))

        rows = [
            f"Top speed:        {p.max_speed:.0f} m/s   ({p.max_speed * 3.6:.0f} km/h)",
            f"Max acceleration: {p.max_accel:.1f} m/s^2  ({p.max_accel / G:.1f} g)",
            f"Max braking:      {p.max_decel:.1f} m/s^2  ({p.max_decel / G:.1f} g)",
            f"Max lateral grip: {p.max_lateral_accel:.1f} m/s^2  ({p.max_lateral_accel / G:.1f} g)",
        ]
        for i, row in enumerate(rows):
            surface = self.app.label_font.render(row, True, COLOR_MEASUREMENT)
            screen.blit(surface, (60, 355 + i * 24))
