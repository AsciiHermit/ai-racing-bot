"""Step 2: Vehicle config + sensors. v1 lets the user set mass and rough
dimensions (consumed by physics/rendering), and shows the fixed v1 sensor
set: a forward lidar (range editable), plus GPS + IMU for localization
(ideal/noiseless, no config yet -- see ars.sensors.GpsSensor/ImuSensor).
Sensor count/type is not otherwise configurable yet (v1.1 backlog).
"""
from __future__ import annotations

import math

from ars.dashboard.screen import Screen
from ars.dashboard.widgets import (
    COLOR_LABEL,
    COLOR_MEASUREMENT,
    COLOR_PANEL_BORDER,
    COLOR_TITLE,
    NumberField,
    Rect,
)
from ars.viz.camera import Camera
from ars.viz.car_sprite import CarSprite

FIELD_W = 200
FIELD_H = 32

PREVIEW_AREA = (60, 460, 880, 190)  # x, y, w, h
COLOR_SENSOR_RANGE = (255, 210, 90)


class VehicleScreen(Screen):
    title = "Vehicle"

    def __init__(self, app):
        super().__init__(app)
        self._fields: list[NumberField] = []
        self._car_sprite = CarSprite()

    def on_enter(self) -> None:
        v = self.config.vehicle
        l = self.config.lidar
        self._fields = [
            NumberField(Rect(60, 160, FIELD_W, FIELD_H), "Mass (kg)", v.mass_kg, 100.0, 3000.0),
            NumberField(Rect(60, 240, FIELD_W, FIELD_H), "Length (m)", v.length_m, 1.0, 10.0),
            NumberField(Rect(60, 320, FIELD_W, FIELD_H), "Width (m)", v.width_m, 0.5, 4.0),
            NumberField(Rect(560, 280, FIELD_W, FIELD_H), "Lidar max range (m)", l.max_range_m, 5.0, 200.0),
        ]

    def handle_event(self, event) -> None:
        for field in self._fields:
            field.handle_event(event, self.app.pygame)
        # commit into config live so Back/Next never loses in-progress edits
        v = self.config.vehicle
        v.mass_kg, v.length_m, v.width_m = (f.value for f in self._fields[:3])
        self.config.lidar.max_range_m = self._fields[3].value

    def draw(self, screen) -> None:
        title = self.app.title_font.render("Vehicle Configuration", True, COLOR_TITLE)
        screen.blit(title, (30, 60))

        section1 = self.app.body_font.render("VEHICLE", True, COLOR_TITLE)
        screen.blit(section1, (60, 120))
        for field in self._fields[:3]:
            field.draw(screen, self.app.pygame, self.app.body_font, self.app.label_font)

        section2 = self.app.body_font.render("SENSORS  (v1 fixed set)", True, COLOR_TITLE)
        screen.blit(section2, (560, 120))

        info_lines = [
            "Forward Lidar -- 1 ray, straight ahead, fixed direction",
            "GPS   -- world position as lat/lon (ideal, no noise)",
            "IMU   -- accelerometer + gyroscope (ideal, no noise)",
        ]
        for i, line in enumerate(info_lines):
            surface = self.app.label_font.render(line, True, COLOR_LABEL)
            screen.blit(surface, (560, 165 + i * 20))

        self._fields[3].draw(screen, self.app.pygame, self.app.body_font, self.app.label_font)

        note = self.app.small_font.render(
            "Multiple / configurable sensor types, and sensor noise, are v1.1+ -- not available yet.",
            True,
            COLOR_LABEL,
        )
        screen.blit(note, (60, 420))

        self._draw_preview(screen)

    def _draw_preview(self, screen) -> None:
        """Top-down preview: car drawn to its configured length/width,
        with the forward lidar's range drawn to the same scale -- so
        changing a field visibly changes the picture, not just a number."""
        pygame = self.app.pygame
        px, py, pw, ph = PREVIEW_AREA
        pygame.draw.rect(screen, (14, 15, 18), (px, py, pw, ph), border_radius=6)
        pygame.draw.rect(screen, COLOR_PANEL_BORDER, (px, py, pw, ph), width=1, border_radius=6)

        header = self.app.label_font.render("PREVIEW (top-down, to scale)", True, COLOR_TITLE)
        screen.blit(header, (px + 10, py + 8))

        v = self.config.vehicle
        lidar = self.config.lidar

        # Camera framed on the car + its sensor range, so the whole ray
        # fits in the panel regardless of the configured range/length.
        camera = Camera(screen_width=pw, screen_height=ph - 30)
        half_extent = max(v.length_m, lidar.max_range_m) * 1.15
        camera.fit_to_bounds(-v.length_m, half_extent, -half_extent * 0.4, half_extent * 0.4, margin=1.0)
        origin = (px, py + 30)

        def to_screen(x: float, y: float) -> tuple[int, int]:
            sx, sy = camera.world_to_screen(x, y)
            return sx + origin[0], sy + origin[1]

        # lidar ray, forward along +x (car-local nose direction)
        ray_end = to_screen(lidar.max_range_m, 0.0)
        car_nose = to_screen(v.length_m / 2, 0.0)
        pygame.draw.line(screen, COLOR_SENSOR_RANGE, car_nose, ray_end, width=2)
        range_label = self.app.small_font.render(f"{lidar.max_range_m:.0f} m", True, COLOR_SENSOR_RANGE)
        screen.blit(range_label, (ray_end[0] - range_label.get_width() // 2, ray_end[1] - 20))

        # car sprite, nose at +x, centered at world origin
        length_px = camera.scale(v.length_m)
        width_px = camera.scale(v.width_m)
        sprite = self._car_sprite.get_scaled(pygame, length_px, width_px)
        rect = sprite.get_rect(center=to_screen(0.0, 0.0))
        screen.blit(sprite, rect)
        pygame.draw.circle(screen, COLOR_SENSOR_RANGE, car_nose, 3)
