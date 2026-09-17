"""Step 1: Track. v1 has exactly one fixed track (simple oval, no banking)
and no configurable parameters -- this screen is purely a visualizer with
distance measurements (total length, per-segment length, turn radius,
track width), so the user can see what they're racing on before moving on.
"""
from __future__ import annotations

import math

from ars.dashboard.screen import Screen
from ars.dashboard.widgets import COLOR_LABEL, COLOR_TITLE
from ars.track import make_simple_oval
from ars.viz.camera import Camera
from ars.viz.track_geometry import build_track_polylines

COLOR_TRACK_SURFACE = (58, 62, 70)
COLOR_BOUNDARY = (235, 235, 235)
COLOR_CENTERLINE = (90, 96, 110)
COLOR_MEASUREMENT = (140, 200, 255)
COLOR_MEASUREMENT_LINE = (90, 130, 160)

PLOT_AREA = (30, 90, 640, 520)  # x, y, w, h -- left panel
INFO_PANEL_X = 690


class TrackScreen(Screen):
    title = "Track"

    def __init__(self, app):
        super().__init__(app)
        self.track = make_simple_oval()
        self._polylines = build_track_polylines(self.track)
        self._segments = self.track.segments()

        px, py, pw, ph = PLOT_AREA
        self.camera = Camera(screen_width=pw, screen_height=ph)
        min_x, max_x, min_y, max_y = self._polylines.bounds
        self.camera.fit_to_bounds(min_x, max_x, min_y, max_y, margin=1.25)
        self._origin = (px, py)

    def _to_screen(self, x: float, y: float) -> tuple[int, int]:
        sx, sy = self.camera.world_to_screen(x, y)
        return sx + self._origin[0], sy + self._origin[1]

    def draw(self, screen) -> None:
        pygame = self.app.pygame

        title = self.app.title_font.render("Track: Simple Oval (v1 fixed, not configurable)", True, COLOR_TITLE)
        screen.blit(title, (30, 60))

        px, py, pw, ph = PLOT_AREA
        pygame.draw.rect(screen, (14, 15, 18), (px, py, pw, ph), border_radius=6)

        outer_px = [self._to_screen(x, y) for x, y in self._polylines.outer_boundary]
        inner_px = [self._to_screen(x, y) for x, y in self._polylines.inner_boundary]
        pygame.draw.polygon(screen, COLOR_TRACK_SURFACE, outer_px + list(reversed(inner_px)))
        pygame.draw.lines(screen, COLOR_BOUNDARY, True, outer_px, width=2)
        pygame.draw.lines(screen, COLOR_BOUNDARY, True, inner_px, width=2)

        centerline_px = [self._to_screen(x, y) for x, y in self._polylines.centerline]
        pygame.draw.lines(screen, COLOR_CENTERLINE, True, centerline_px, width=1)

        self._draw_segment_labels(screen)
        self._draw_width_dimension(screen)
        self._draw_info_panel(screen)

    def _draw_segment_labels(self, screen) -> None:
        # Track centroid, to decide which side of each segment is "outside"
        # (so labels push away from the track body instead of sitting on
        # top of a boundary line).
        cx = sum(x for x, _ in self._polylines.centerline) / len(self._polylines.centerline)
        cy = sum(y for _, y in self._polylines.centerline) / len(self._polylines.centerline)

        for seg in self._segments:
            mid_s = (seg.s_start + seg.s_end) / 2
            sample = self.track.sample_at_s(mid_s)
            nx = -math.sin(sample.heading)
            ny = math.cos(sample.heading)

            # pick the normal direction (+n or -n) that points away from centroid
            to_point_x = sample.centerline_x - cx
            to_point_y = sample.centerline_y - cy
            outward = 1.0 if (nx * to_point_x + ny * to_point_y) >= 0 else -1.0

            offset = sample.width / 2 + 14
            label_x = sample.centerline_x + nx * offset * outward
            label_y = sample.centerline_y + ny * offset * outward

            text = f"{seg.length:.0f}m" if seg.kind == "straight" else f"{seg.length:.0f}m (r={seg.radius:.0f}m)"
            sx, sy = self._to_screen(label_x, label_y)
            surface = self.app.small_font.render(text, True, COLOR_MEASUREMENT)
            screen.blit(surface, (sx - surface.get_width() // 2, sy - surface.get_height() // 2))

    def _draw_width_dimension(self, screen) -> None:
        pygame = self.app.pygame
        sample = self.track.sample_at_s(0.0)
        nx = -math.sin(sample.heading)
        ny = math.cos(sample.heading)
        half_w = sample.width / 2
        p1 = self._to_screen(sample.centerline_x - nx * half_w, sample.centerline_y - ny * half_w)
        p2 = self._to_screen(sample.centerline_x + nx * half_w, sample.centerline_y + ny * half_w)
        pygame.draw.line(screen, COLOR_MEASUREMENT_LINE, p1, p2, width=2)
        mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
        label = self.app.small_font.render(f"width {sample.width:.0f}m", True, COLOR_MEASUREMENT)
        screen.blit(label, (mid[0] + 8, mid[1] - 8))

    def _draw_info_panel(self, screen) -> None:
        x = INFO_PANEL_X
        y = 90
        lines = [
            "MEASUREMENTS",
            "",
            f"Total length:  {self.track.length:.1f} m",
            f"Track width:   {self.track.width:.1f} m",
            "",
            "Segments:",
        ]
        for i, seg in enumerate(self._segments):
            if seg.kind == "straight":
                lines.append(f"  {i + 1}. straight  {seg.length:.1f} m")
            else:
                lines.append(f"  {i + 1}. turn      {seg.length:.1f} m")
                lines.append(f"      radius {seg.radius:.1f} m, {math.degrees(seg.angle):.0f} deg")

        for i, line in enumerate(lines):
            color = COLOR_TITLE if i == 0 else COLOR_LABEL
            surface = self.app.body_font.render(line, True, color)
            screen.blit(surface, (x, y + i * 22))
