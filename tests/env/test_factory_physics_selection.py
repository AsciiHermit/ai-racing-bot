from ars.env.factory import make_default_env
from ars.physics import DynamicBicyclePhysics, KinematicBicyclePhysics


def test_default_physics_is_dynamic_bicycle():
    env = make_default_env()
    assert isinstance(env.physics, DynamicBicyclePhysics)


def test_physics_backend_is_overridable_for_ablations():
    env = make_default_env(physics=KinematicBicyclePhysics())
    assert isinstance(env.physics, KinematicBicyclePhysics)
