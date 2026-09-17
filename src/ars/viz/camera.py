"""World <-> screen coordinate transform for the 2D top-down viewer.

Kept separate from the drawing code so it's trivially unit-testable without
a display, and so panning/zoom can change later without touching renderers.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Camera:
    screen_width: int
    screen_height: int
    center_x: float = 0.0  # world coords the screen center is looking at
    center_y: float = 0.0
    pixels_per_meter: float = 4.0

    def world_to_screen(self, x: float, y: float) -> tuple[int, int]:
        sx = self.screen_width / 2 + (x - self.center_x) * self.pixels_per_meter
        # screen y grows downward; world y grows "up" (standard math convention)
        sy = self.screen_height / 2 - (y - self.center_y) * self.pixels_per_meter
        return int(sx), int(sy)

    def scale(self, meters: float) -> int:
        return max(1, int(meters * self.pixels_per_meter))

    def fit_to_bounds(self, min_x: float, max_x: float, min_y: float, max_y: float, margin: float = 1.1) -> None:
        """Center the camera on the given world-space bounds and pick a
        pixels_per_meter that fits them on screen, with a margin."""
        self.center_x = (min_x + max_x) / 2
        self.center_y = (min_y + max_y) / 2
        span_x = max(max_x - min_x, 1e-6) * margin
        span_y = max(max_y - min_y, 1e-6) * margin
        ppm_x = self.screen_width / span_x
        ppm_y = self.screen_height / span_y
        self.pixels_per_meter = min(ppm_x, ppm_y)
