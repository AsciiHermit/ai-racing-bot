"""Loads the top-down F1 car image and prepares it for world-scale, heading-
aware rendering: crop the white margin so pixel size maps to true car
size, orient it so "nose" aligns with heading=0 (+x, i.e. facing right),
and cache per-(pixel-length) scaled+rotated copies since rotating a
pygame Surface every frame is wasteful.
"""
from __future__ import annotations

import os

_ASSET_PATH = os.path.join(os.path.dirname(__file__), "assets", "f1_car_top_down.jpg")

# Source image: car nose points toward -x (left) in the raw file, with a
# roughly 1198x750px content bounding box inside a 1200x1200 white canvas
# (measured once at authoring time -- re-measure if the asset changes).
_SOURCE_BBOX = (0, 224, 1198, 974)  # left, top, right, bottom
_SOURCE_FACES_LEFT = True


class CarSprite:
    """Lazily loads and caches the car image. One instance per LiveViewer/
    dashboard screen; cheap to construct (loading is deferred to first use)."""

    def __init__(self):
        self._pygame = None
        self._base_surface = None  # cropped, nose-to-+x, alpha-keyed on white
        self._cache: dict[tuple[int, int], "object"] = {}

    def _ensure_loaded(self, pygame_module) -> None:
        if self._base_surface is not None:
            return
        self._pygame = pygame_module
        raw = pygame_module.image.load(_ASSET_PATH).convert()
        left, top, right, bottom = _SOURCE_BBOX
        cropped = raw.subsurface(pygame_module.Rect(left, top, right - left, bottom - top)).copy()
        if _SOURCE_FACES_LEFT:
            cropped = pygame_module.transform.flip(cropped, True, False)
        cropped.set_colorkey((255, 255, 255))  # white background -> transparent
        self._base_surface = cropped

    def get_scaled(self, pygame_module, length_px: int, width_px: int):
        """Return a surface sized (length_px, width_px) with the car's nose
        pointing toward +x, alpha-keyed so the track shows through the
        margins. Cached per integer pixel size."""
        self._ensure_loaded(pygame_module)
        key = (length_px, width_px)
        cached = self._cache.get(key)
        if cached is None:
            cached = pygame_module.transform.smoothscale(self._base_surface, (length_px, width_px))
            self._cache[key] = cached
        return cached
