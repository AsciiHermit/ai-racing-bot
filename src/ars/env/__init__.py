from ars.env.factory import make_default_env
from ars.env.racing_env import RacingEnv

__all__ = ["RacingEnv", "make_default_env"]

try:
    import gymnasium as gym

    gym.register(
        id="ARS-Racing-v0",
        entry_point="ars.env.factory:make_default_env",
    )
except ImportError:
    pass
