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


def test_background_is_colorkeyed_transparent():
    _ensure_display()
    sprite = CarSprite()
    surface = sprite.get_scaled(pygame, 100, 40)
    assert surface.get_colorkey() is not None
