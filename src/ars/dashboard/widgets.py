"""Minimal pygame widgets shared across dashboard screens: buttons and
numeric text fields. Deliberately small -- v1 needs a handful of controls,
not a full UI toolkit. Swap for a real framework later if the dashboard
grows past what this can comfortably do.
"""
from __future__ import annotations

from dataclasses import dataclass


COLOR_PANEL = (36, 39, 46)
COLOR_PANEL_BORDER = (70, 75, 84)
COLOR_BUTTON = (58, 62, 70)
COLOR_BUTTON_HOVER = (78, 84, 96)
COLOR_BUTTON_TEXT = (230, 230, 230)
COLOR_FIELD_BG = (20, 22, 26)
COLOR_FIELD_BORDER = (90, 96, 110)
COLOR_FIELD_BORDER_ACTIVE = (226, 84, 64)
COLOR_LABEL = (200, 202, 208)
COLOR_TITLE = (240, 240, 240)
COLOR_ACCENT = (226, 84, 64)
COLOR_MEASUREMENT = (140, 200, 255)


@dataclass
class Rect:
    x: int
    y: int
    w: int
    h: int

    def contains(self, px: int, py: int) -> bool:
        return self.x <= px <= self.x + self.w and self.y <= py <= self.y + self.h

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)


class Button:
    def __init__(self, rect: Rect, label: str, on_click):
        self.rect = rect
        self.label = label
        self.on_click = on_click
        self.enabled = True

    def handle_event(self, event, pygame_module) -> None:
        if not self.enabled:
            return
        if event.type == pygame_module.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.contains(*event.pos):
                self.on_click()

    def draw(self, screen, pygame_module, font) -> None:
        mouse_pos = pygame_module.mouse.get_pos()
        hovered = self.enabled and self.rect.contains(*mouse_pos)
        color = COLOR_BUTTON_HOVER if hovered else COLOR_BUTTON
        if not self.enabled:
            color = (44, 46, 50)
        pygame_module.draw.rect(screen, color, self.rect.as_tuple(), border_radius=6)
        pygame_module.draw.rect(screen, COLOR_PANEL_BORDER, self.rect.as_tuple(), width=1, border_radius=6)
        text_color = COLOR_BUTTON_TEXT if self.enabled else (110, 110, 110)
        surface = font.render(self.label, True, text_color)
        text_rect = surface.get_rect(center=(self.rect.x + self.rect.w // 2, self.rect.y + self.rect.h // 2))
        screen.blit(surface, text_rect)


class NumberField:
    """Click to focus, type digits/./- , Enter or click-away to commit."""

    def __init__(self, rect: Rect, label: str, value: float, min_value: float, max_value: float):
        self.rect = rect
        self.label = label
        self.value = value
        self.min_value = min_value
        self.max_value = max_value
        self._text = _format_value(value)
        self._focused = False

    def handle_event(self, event, pygame_module) -> None:
        if event.type == pygame_module.MOUSEBUTTONDOWN and event.button == 1:
            was_focused = self._focused
            self._focused = self.rect.contains(*event.pos)
            if was_focused and not self._focused:
                self._commit()
        elif event.type == pygame_module.KEYDOWN and self._focused:
            if event.key == pygame_module.K_RETURN:
                self._commit()
                self._focused = False
            elif event.key == pygame_module.K_BACKSPACE:
                self._text = self._text[:-1]
            elif event.unicode in "0123456789.-":
                self._text += event.unicode

    def _commit(self) -> None:
        try:
            parsed = float(self._text)
            self.value = max(self.min_value, min(self.max_value, parsed))
        except ValueError:
            pass
        self._text = _format_value(self.value)

    def draw(self, screen, pygame_module, font, label_font) -> None:
        label_surface = label_font.render(self.label, True, COLOR_LABEL)
        screen.blit(label_surface, (self.rect.x, self.rect.y - 20))

        border_color = COLOR_FIELD_BORDER_ACTIVE if self._focused else COLOR_FIELD_BORDER
        pygame_module.draw.rect(screen, COLOR_FIELD_BG, self.rect.as_tuple(), border_radius=4)
        pygame_module.draw.rect(screen, border_color, self.rect.as_tuple(), width=1, border_radius=4)

        display_text = self._text if self._focused else _format_value(self.value)
        text_surface = font.render(display_text, True, COLOR_BUTTON_TEXT)
        screen.blit(text_surface, (self.rect.x + 8, self.rect.y + (self.rect.h - text_surface.get_height()) // 2))


def _format_value(value: float) -> str:
    if value == int(value):
        return str(int(value))
    return f"{value:.2f}"
