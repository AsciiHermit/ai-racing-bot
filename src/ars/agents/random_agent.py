"""Trivial baseline agent: uniform-random actions. Exists to validate the
env end-to-end without depending on any RL library. The RL teammate's
PPO/SAC baselines and the novel algorithm live alongside this module."""
from __future__ import annotations

import numpy as np


class RandomAgent:
    def __init__(self, action_space, seed: int | None = None):
        self.action_space = action_space
        if seed is not None:
            self.action_space.seed(seed)

    def act(self, obs: np.ndarray) -> np.ndarray:
        return self.action_space.sample()
