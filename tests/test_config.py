import pytest

from rabbithole.config import load_app_config


def test_live_mode_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv("RABBITHOLE_LIVE_ENABLED", raising=False)
    monkeypatch.delenv("RABBITHOLE_MODELS", raising=False)
    config = load_app_config()
    assert config.live_agent_enabled is False
    assert config.models == ("glm-4-flash",)


def test_server_limits_reject_unsafe_values(monkeypatch):
    monkeypatch.setenv("RABBITHOLE_MAX_ITERATIONS", "99")
    with pytest.raises(ValueError, match="RABBITHOLE_MAX_ITERATIONS"):
        load_app_config()


def test_model_allowlist_controls_default(monkeypatch):
    monkeypatch.setenv("RABBITHOLE_MODELS", "glm-4-flash,glm-4-air")
    monkeypatch.setenv("ZHIPU_MODEL", "unlisted-model")
    config = load_app_config()
    assert config.models == ("glm-4-flash", "glm-4-air")
    assert config.default_model == "glm-4-flash"
