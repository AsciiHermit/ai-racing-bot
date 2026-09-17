"""Loads the top-down F1 car image and prepares it for world-scale, heading-
aware rendering: crop the white margin so pixel size maps to true car
size, orient it so "nose" aligns with heading=0 (+x, i.e. facing right),
and cache per-(pixel-length) scaled+rotated copies since rotating a
pygame Surface every frame is wasteful.
"""
from __future__ import annotations

import os

_ASSET_PATH = os.path.join(os.path.dirname(__file__), "assets", "f1_car_top_down.jpg")

# Source image: the pointed nose (cockpit visible) already points toward
# +x (right) in the raw file -- the wide flat block on the left is the
# rear wing, not the front, so no flip is needed. Content bounding box is
# roughly 1198x750px inside a 1200x1200 white canvas (measured once at
# authoring time -- re-measure if the asset changes). Verified by
# rendering both orientations and visually checking which end is the
# tapered nose vs. the flat wing.
_SOURCE_BBOX = (0, 224, 1198, 974)  # left, top, right, bottom


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
        cropped = _make_white_transparent(cropped, pygame_module)
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


def _make_white_transparent(surface, pygame_module, threshold: int = 235):
    """Per-pixel alpha, not colorkey: a JPEG's white background has
    compression noise (near-white, not exactly (255,255,255)), so an exact
    colorkey leaves a visible whitish halo around the car. Any pixel whose
    channels are all >= threshold is treated as background. Runs once at
    load time (cached by CarSprite), so the pixel-by-pixel cost is fine.
    """
    surface = surface.convert_alpha()
    w, h = surface.get_size()
    surface.lock()
    for x in range(w):
        for y in range(h):
            r, g, b, a = surface.get_at((x, y))
            if r >= threshold and g >= threshold and b >= threshold:
                surface.set_at((x, y), (r, g, b, 0))
    surface.unlock()
    return surface
