"""Arrow-key/WASD -> VehicleAction, for manually driving the car in the
live viewer. Only useful with pygame installed (reads pygame's key state)."""
from __future__ import annotations

import numpy as np


def read_keyboard_action(pygame_module) -> np.ndarray:
    """Return a [throttle, brake, steer] array from current key state.
    Call once per frame, after LiveViewer.poll_events()."""
    keys = pygame_module.key.get_pressed()
    throttle = 1.0 if (keys[pygame_module.K_UP] or keys[pygame_module.K_w]) else 0.0
    brake = 1.0 if (keys[pygame_module.K_DOWN] or keys[pygame_module.K_s]) else 0.0
    steer = 0.0
    if keys[pygame_module.K_LEFT] or keys[pygame_module.K_a]:
        steer -= 1.0
    if keys[pygame_module.K_RIGHT] or keys[pygame_module.K_d]:
        steer += 1.0
    return np.array([throttle, brake, steer], dtype=np.float32)
