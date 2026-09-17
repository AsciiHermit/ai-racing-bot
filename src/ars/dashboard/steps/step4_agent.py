"""Step 4: Agent. v1 ships one option -- a hand-coded "dummy expert" line
follower (ars.agents.DummyExpertAgent). "Choose your own RL agent" /
"train it" are the intended v1.1+ paths (this screen names them so the
eventual UI slot is obvious) but aren't implemented yet.
"""
from __future__ import annotations

from ars.dashboard.screen import Screen
from ars.dashboard.widgets import COLOR_ACCENT, COLOR_LABEL, COLOR_TITLE


class AgentScreen(Screen):
    title = "Agent"

    def draw(self, screen) -> None:
        title = self.app.title_font.render("Autonomous Agent", True, COLOR_TITLE)
        screen.blit(title, (30, 60))

        selected = self.app.body_font.render("[x] Dummy Expert -- line-following controller (v1 default)", True, COLOR_ACCENT)
        screen.blit(selected, (60, 130))

        disabled_lines = [
            "[ ] Bring your own RL agent      -- not available yet (v1.1+)",
            "[ ] Train an agent in-session    -- not available yet (v1.1+)",
        ]
        for i, line in enumerate(disabled_lines):
            surface = self.app.body_font.render(line, True, (100, 103, 110))
            screen.blit(surface, (60, 170 + i * 30))

        desc_lines = [
            "The dummy expert is not learned -- it's a simple hand-coded",
            "controller that steers toward the track centerline and eases",
            "off the throttle in turns, proportional to curvature. Useful",
            "as a sanity-check baseline before any real RL agent exists.",
        ]
        for i, line in enumerate(desc_lines):
            surface = self.app.label_font.render(line, True, COLOR_LABEL)
            screen.blit(surface, (60, 260 + i * 22))
