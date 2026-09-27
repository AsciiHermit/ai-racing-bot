from ars.dashboard.builder import build_env_and_agent
from ars.dashboard.config import SessionConfig
from ars.physics import DynamicBicyclePhysics, KinematicBicyclePhysics


def test_default_session_config_selects_dynamic_bicycle():
    config = SessionConfig()
    env, _ = build_env_and_agent(config)
    assert isinstance(env.physics, DynamicBicyclePhysics)


def test_kinematic_stub_model_name_selects_the_kinematic_stub():
    config = SessionConfig()
    config.physics.model_name = "kinematic_stub"
    env, _ = build_env_and_agent(config)
    assert isinstance(env.physics, KinematicBicyclePhysics)
