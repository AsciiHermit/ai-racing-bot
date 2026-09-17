"""Step 3: Physics. v1 has exactly one physics implementation (the
kinematic-bicycle stub -- real 4-wheel F1-style dynamics is still being
built, see ars.physics). No real selection or tunable params exist yet;
this screen is a placeholder so the step stays in the flow and the user
knows what they're getting, without pretending there's a choice to make.
"""
from __future__ import annotations

from ars.dashboard.screen import Screen
from ars.dashboard.widgets import COLOR_LABEL, COLOR_TITLE


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
