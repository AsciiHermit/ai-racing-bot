"""Step 2: Vehicle config + sensors. v1 lets the user set mass and rough
dimensions (consumed by physics/rendering), and shows the one fixed sensor
-- a forward lidar -- with its range editable. Sensor count/type is not
configurable yet (v1.1 backlog): only one lidar, always forward-facing.
"""
from __future__ import annotations

from ars.dashboard.screen import Screen
from ars.dashboard.widgets import COLOR_LABEL, COLOR_TITLE, NumberField, Rect

FIELD_W = 200
FIELD_H = 32


class VehicleScreen(Screen):
    title = "Vehicle"

    def __init__(self, app):
        super().__init__(app)
        self._fields: list[NumberField] = []

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

        section2 = self.app.body_font.render("SENSORS  (v1: exactly one forward lidar, fixed)", True, COLOR_TITLE)
        screen.blit(section2, (560, 120))

        info_lines = [
            "Type:      Forward Lidar",
            "Rays:      1 (straight ahead)",
            "Direction: fixed, aligned with heading",
        ]
        for i, line in enumerate(info_lines):
            surface = self.app.label_font.render(line, True, COLOR_LABEL)
            screen.blit(surface, (560, 165 + i * 20))

        self._fields[3].draw(screen, self.app.pygame, self.app.body_font, self.app.label_font)

        note = self.app.small_font.render(
            "Multiple / configurable sensor types are v1.1+ -- not available yet.", True, COLOR_LABEL
        )
        screen.blit(note, (60, 420))
