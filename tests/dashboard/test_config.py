from ars.dashboard.config import SessionConfig


def test_default_session_config_has_sane_values():
    config = SessionConfig()
    assert config.vehicle.mass_kg > 0
    assert config.lidar.num_rays == 1
    assert config.agent.kind == "dummy_expert"
