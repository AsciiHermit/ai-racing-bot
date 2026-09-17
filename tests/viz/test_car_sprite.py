import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

from ars.viz.car_sprite import CarSprite


def _ensure_display():
    if not pygame.get_init():
        pygame.init()
    if pygame.display.get_surface() is None:
        pygame.display.set_mode((10, 10))


def test_get_scaled_returns_requested_size():
    _ensure_display()
    sprite = CarSprite()
    surface = sprite.get_scaled(pygame, 100, 40)
    assert surface.get_size() == (100, 40)


def test_get_scaled_is_cached_for_same_size():
    _ensure_display()
    sprite = CarSprite()
    a = sprite.get_scaled(pygame, 80, 30)
    b = sprite.get_scaled(pygame, 80, 30)
    assert a is b


def test_get_scaled_different_sizes_not_cached_together():
    _ensure_display()
    sprite = CarSprite()
    a = sprite.get_scaled(pygame, 80, 30)
    b = sprite.get_scaled(pygame, 120, 45)
    assert a is not b
    assert a.get_size() != b.get_size()


def test_background_has_per_pixel_alpha():
    # Colorkey alone leaves a visible white halo around the car (JPEG
    # compression noise means the background isn't pure (255,255,255)
    # everywhere) -- background removal uses per-pixel alpha instead.
    _ensure_display()
    sprite = CarSprite()
    surface = sprite.get_scaled(pygame, 100, 40)
    assert surface.get_flags() & pygame.SRCALPHA


def test_corner_pixels_are_transparent():
    # Corners of the bounding box are background (car doesn't reach the
    # corners of its own bounding rect), so they should be fully transparent.
    _ensure_display()
    sprite = CarSprite()
    surface = sprite.get_scaled(pygame, 100, 40)
    for corner in [(0, 0), (99, 0), (0, 39), (99, 39)]:
        assert surface.get_at(corner).a == 0


def test_nose_faces_positive_x_before_rotation():
    # Regression test: the source image's pointed nose (not the flat rear
    # wing) must land on the +x side after cropping, since LiveViewer
    # rotates this base orientation by the vehicle's heading with no
    # additional flip. The car tapers to its narrowest vertical span right
    # at the nose tip -- find where that minimum occurs and check it's in
    # the +x half of the image, not the -x (rear wing) half.
    _ensure_display()
    sprite = CarSprite()
    surface = sprite.get_scaled(pygame, 200, 80)
    w, h = surface.get_size()

    def opaque_span(x):
        ys = [y for y in range(h) if surface.get_at((x, y)).a > 0]
        return (max(ys) - min(ys)) if ys else 999

    spans = [(x, opaque_span(x)) for x in range(0, w, 5)]
    narrowest_x = min(spans, key=lambda t: t[1])[0]
    assert narrowest_x > w / 2, "expected the tapered nose tip in the +x/right half of the image"
