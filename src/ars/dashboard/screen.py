"""Base class for one dashboard step/screen."""
from __future__ import annotations

from ars.dashboard.config import SessionConfig


class Screen:
    """One step in the 5-step flow. Subclasses override the hooks below.
    `app` gives access to the shared pygame surface/font/config so screens
    don't each reinitialize pygame."""

    title = "Screen"

    def __init__(self, app):
        self.app = app

    @property
    def config(self) -> SessionConfig:
        return self.app.config

    def on_enter(self) -> None:
        """Called each time this screen becomes active (e.g. rebuild
        widgets from current config, since Back may have changed it)."""

    def handle_event(self, event) -> None:
        pass

    def draw(self, screen) -> None:
        pass

    def can_advance(self) -> bool:
        """Whether Next should be enabled. Override to gate progression."""
        return True
