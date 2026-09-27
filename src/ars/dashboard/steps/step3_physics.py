"""Step 3: Physics. v1 has two physics implementations (see ars.physics):
the kinematic-bicycle stub (flat grip cap, kept as a faster/degenerate
option for ablations) and Phase 1's dynamic-bicycle model (slip-based,
Pacejka tire curve -- the default). Still no interactive picker widget or
tunable-params form in this screen; config.physics.model_name is set
programmatically (ars.dashboard.config.PhysicsConfig), not from here yet.
"""
from __future__ import annotations

from ars.dashboard.screen import Screen
from ars.dashboard.widgets import COLOR_LABEL, COLOR_MEASUREMENT, COLOR_TITLE
from ars.physics import DynamicBicycleParams, KinematicBicycleParams

G = 9.81


class PhysicsScreen(Screen):
    title = "Physics"

    def draw(self, screen) -> None:
        title = self.app.title_font.render("Physics", True, COLOR_TITLE)
        screen.blit(title, (30, 60))

        lines = [
            f"Model:  {self.config.physics.model_name}",
            "",
            "Two physics models exist (ars.physics): dynamic_bicycle",
            "(slip-based Pacejka tire curve, the default) and",
            "kinematic_stub (flat grip cap, kept for ablations). No",
            "interactive picker or tunable-params form here yet.",
        ]
        for i, line in enumerate(lines):
            color = COLOR_TITLE if i == 0 else COLOR_LABEL
            surface = self.app.body_font.render(line, True, color)
            screen.blit(surface, (60, 130 + i * 26))

        self._draw_reference_values(screen)

    def _draw_reference_values(self, screen) -> None:
        if self.config.physics.model_name == "kinematic_stub":
            p = KinematicBicycleParams()
            grip_row = f"Max lateral grip: {p.max_lateral_accel:.1f} m/s^2  ({p.max_lateral_accel / G:.1f} g, flat cap)"
        else:
            p = DynamicBicycleParams()
            grip_row = f"Tire friction mu: {p.mu:.2f}  (grip emerges from the tire curve, no flat cap)"

        header = self.app.body_font.render("REFERENCE VALUES (real F1-scale)", True, COLOR_TITLE)
        screen.blit(header, (60, 320))

        rows = [
            f"Top speed:        {p.max_speed:.0f} m/s   ({p.max_speed * 3.6:.0f} km/h)",
            grip_row,
        ]
        for i, row in enumerate(rows):
            surface = self.app.label_font.render(row, True, COLOR_MEASUREMENT)
            screen.blit(surface, (60, 355 + i * 24))
