"""Pygame live viewer: renders a RacingEnv's track + car in real time.

Deliberately decoupled from RacingEnv internals -- reads only the public
`env.track` and `env.vehicle_state` (x, y, heading), so it keeps working
whichever physics/track implementation is plugged in behind those
Protocols. Import pygame lazily so the rest of the package (training,
headless CI) never needs it installed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ars.viz.camera import Camera
from ars.viz.track_geometry import TrackPolylines, build_track_polylines

# Colors: brand-neutral, readable on the default pygame black-ish background.
COLOR_BG = (24, 26, 30)
COLOR_TRACK_SURFACE = (58, 62, 70)
COLOR_BOUNDARY = (235, 235, 235)
COLOR_CENTERLINE = (90, 96, 110)
COLOR_CAR = (226, 84, 64)
COLOR_TEXT = (220, 220, 220)


@dataclass
class ViewerConfig:
    width: int = 900
    height: int = 700
    fps: int = 60
    car_length: float = 4.5  # m, drawn size (F1-scale, independent of physics model)
    car_width: float = 2.0  # m


class LiveViewer:
    """Owns the pygame window. One instance per on-screen session.

    Usage:
        viewer = LiveViewer(env.track, config=ViewerConfig())
        while viewer.is_open:
            ... step env ...
            if not viewer.draw(env.vehicle_state):
                break
        viewer.close()
    """

    def __init__(self, track, config: ViewerConfig | None = None):
        import pygame  # local import: viz is an optional extra

        self.pygame = pygame  # exposed for callers that need raw pygame (e.g. keyboard input)
        self._pygame = pygame
        self.config = config or ViewerConfig()
        pygame.init()
        pygame.display.set_caption("ARS -- Autonomous Racing Simulator")
        self._screen = pygame.display.set_mode((self.config.width, self.config.height))
        self._clock = pygame.time.Clock()
        self._font = pygame.font.SysFont("consolas", 16)

        self.camera = Camera(screen_width=self.config.width, screen_height=self.config.height)
        self.track = track
        self._polylines: TrackPolylines = build_track_polylines(track)
        min_x, max_x, min_y, max_y = self._polylines.bounds
        self.camera.fit_to_bounds(min_x, max_x, min_y, max_y)

        self.is_open = True
        self._hud_lines: list[str] = []

    def poll_events(self) -> None:
        for event in self._pygame.event.get():
            if event.type == self._pygame.QUIT:
                self.is_open = False
            elif event.type == self._pygame.KEYDOWN and event.key == self._pygame.K_ESCAPE:
                self.is_open = False

    def set_hud(self, lines: list[str]) -> None:
        self._hud_lines = lines

    def draw(self, vehicle_state) -> bool:
        """Render one frame for the given vehicle_state (needs .x, .y,
        .heading). Returns False if the user closed the window."""
        self.poll_events()
        if not self.is_open:
            return False

        pygame = self._pygame
        screen = self._screen
        screen.fill(COLOR_BG)

        outer_px = [self.camera.world_to_screen(x, y) for x, y in self._polylines.outer_boundary]
        inner_px = [self.camera.world_to_screen(x, y) for x, y in self._polylines.inner_boundary]
        pygame.draw.polygon(screen, COLOR_TRACK_SURFACE, outer_px + list(reversed(inner_px)))
        pygame.draw.lines(screen, COLOR_BOUNDARY, True, outer_px, width=2)
        pygame.draw.lines(screen, COLOR_BOUNDARY, True, inner_px, width=2)

        centerline_px = [self.camera.world_to_screen(x, y) for x, y in self._polylines.centerline]
        pygame.draw.lines(screen, COLOR_CENTERLINE, True, centerline_px, width=1)

        self._draw_car(vehicle_state)
        self._draw_hud()

        pygame.display.flip()
        self._clock.tick(self.config.fps)
        return self.is_open

    def _draw_car(self, state) -> None:
        pygame = self._pygame
        half_l = self.config.car_length / 2
        half_w = self.config.car_width / 2
        # car-frame corners (nose = +x), rotated by heading, translated to world pos
        corners = [(half_l, -half_w), (half_l, half_w), (-half_l, half_w), (-half_l, -half_w)]
        cos_h, sin_h = _cos_sin(state.heading)
        world_corners = [
            (state.x + cx * cos_h - cy * sin_h, state.y + cx * sin_h + cy * cos_h) for cx, cy in corners
        ]
        screen_corners = [self.camera.world_to_screen(x, y) for x, y in world_corners]
        pygame.draw.polygon(self._screen, COLOR_CAR, screen_corners)

        nose_world = (state.x + half_l * 1.3 * cos_h, state.y + half_l * 1.3 * sin_h)
        pygame.draw.line(
            self._screen,
            COLOR_CAR,
            self.camera.world_to_screen(state.x, state.y),
            self.camera.world_to_screen(*nose_world),
            width=2,
        )

    def _draw_hud(self) -> None:
        for i, line in enumerate(self._hud_lines):
            surface = self._font.render(line, True, COLOR_TEXT)
            self._screen.blit(surface, (10, 10 + i * 20))

    def close(self) -> None:
        self._pygame.quit()
        self.is_open = False


def _cos_sin(angle: float) -> tuple[float, float]:
    import math

    return math.cos(angle), math.sin(angle)
