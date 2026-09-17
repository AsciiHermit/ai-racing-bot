"""Pygame live viewer: renders a RacingEnv's track + car in real time.

Deliberately decoupled from RacingEnv internals -- reads only the public
`env.track` and `env.vehicle_state` (x, y, heading), so it keeps working
whichever physics/track implementation is plugged in behind those
Protocols. Import pygame lazily so the rest of the package (training,
headless CI) never needs it installed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from ars.viz.camera import Camera
from ars.viz.car_sprite import CarSprite
from ars.viz.track_geometry import TrackPolylines, build_track_polylines

# Colors: brand-neutral, readable on the default pygame black-ish background.
COLOR_BG = (24, 26, 30)
COLOR_TRACK_SURFACE = (58, 62, 70)
COLOR_BOUNDARY = (235, 235, 235)
COLOR_CENTERLINE = (90, 96, 110)
COLOR_CAR = (226, 84, 64)
COLOR_TEXT = (220, 220, 220)
COLOR_SENSOR_RANGE = (255, 210, 90)
COLOR_SENSOR_ORIGIN = (255, 230, 140)


@dataclass
class ViewerConfig:
    width: int = 900
    height: int = 700
    fps: int = 60
    car_length: float = 4.5  # m, drawn size (F1-scale, independent of physics model)
    car_width: float = 2.0  # m
    lidar_range: float | None = 40.0  # m, None to hide the sensor overlay
    lidar_fov: float = 0.0  # rad, total field of view centered on heading (0 = forward only)


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

        self._car_sprite = CarSprite()

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
        self._draw_sensor_overlay(state)

        pygame = self._pygame
        length_px = self.camera.scale(self.config.car_length)
        width_px = self.camera.scale(self.config.car_width)
        sprite = self._car_sprite.get_scaled(pygame, length_px, width_px)

        # pygame.transform.rotate(+angle) turns the (nose-facing-+x) sprite
        # counter-clockwise as drawn on screen: rotate(+90) points the nose
        # up. World heading also increases counter-clockwise (world +y is
        # "up" after Camera.world_to_screen's y-flip), so the two match
        # directly with no sign flip needed -- verified by rendering
        # heading=0/90/180/270 against a drawn +x/+y reference and checking
        # the nose lines up with the expected world direction each time.
        heading_deg = math.degrees(state.heading)
        rotated = pygame.transform.rotate(sprite, heading_deg)
        rect = rotated.get_rect(center=self.camera.world_to_screen(state.x, state.y))
        self._screen.blit(rotated, rect)

    def _draw_sensor_overlay(self, state) -> None:
        """Draw the forward lidar's origin and range as an arc/cone, so the
        sensor's field of view is visible relative to the car -- not just
        implied by a hidden max_range number."""
        if self.config.lidar_range is None:
            return
        pygame = self._pygame
        half_fov = self.config.lidar_fov / 2
        cos_h, sin_h = _cos_sin(state.heading - half_fov)
        cos_h2, sin_h2 = _cos_sin(state.heading + half_fov)

        origin_px = self.camera.world_to_screen(state.x, state.y)
        pygame.draw.circle(self._screen, COLOR_SENSOR_ORIGIN, origin_px, 3)

        if self.config.lidar_fov <= 1e-6:
            # single forward ray
            tip_world = (
                state.x + self.config.lidar_range * math.cos(state.heading),
                state.y + self.config.lidar_range * math.sin(state.heading),
            )
            pygame.draw.line(self._screen, COLOR_SENSOR_RANGE, origin_px, self.camera.world_to_screen(*tip_world), width=1)
        else:
            edge1 = (state.x + self.config.lidar_range * cos_h, state.y + self.config.lidar_range * sin_h)
            edge2 = (state.x + self.config.lidar_range * cos_h2, state.y + self.config.lidar_range * sin_h2)
            pygame.draw.line(self._screen, COLOR_SENSOR_RANGE, origin_px, self.camera.world_to_screen(*edge1), width=1)
            pygame.draw.line(self._screen, COLOR_SENSOR_RANGE, origin_px, self.camera.world_to_screen(*edge2), width=1)

    def _draw_hud(self) -> None:
        for i, line in enumerate(self._hud_lines):
            surface = self._font.render(line, True, COLOR_TEXT)
            self._screen.blit(surface, (10, 10 + i * 20))

    def close(self, quit_pygame: bool = True) -> None:
        """quit_pygame=False when a caller (e.g. the dashboard) owns the
        pygame session and will keep using it after this viewer closes --
        pygame.quit() tears down the whole subsystem (display, fonts,
        events), not just this window."""
        if quit_pygame:
            self._pygame.quit()
        self.is_open = False


def _cos_sin(angle: float) -> tuple[float, float]:
    import math

    return math.cos(angle), math.sin(angle)
