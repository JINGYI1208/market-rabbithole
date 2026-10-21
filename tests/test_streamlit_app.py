from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_public_demo_renders_without_provider_keys(monkeypatch):
    monkeypatch.delenv("ZHIPUAI_API_KEY", raising=False)
    monkeypatch.delenv("ZHIPU_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.setenv("RABBITHOLE_LIVE_ENABLED", "0")

    app = AppTest.from_file(Path(__file__).parents[1] / "streamlit_app.py", default_timeout=10).run()

    assert not app.exception
    assert app.radio[0].options == ["Research", "Research Library", "Demo Gallery"]
    app.radio[0].set_value("Demo Gallery").run()
    assert any("Fixture-based recorded run" in item.value for item in app.info)


def test_frontend_cannot_enable_server_disabled_live_mode(monkeypatch):
    monkeypatch.setenv("RABBITHOLE_LIVE_ENABLED", "0")
    app = AppTest.from_file(Path(__file__).parents[1] / "streamlit_app.py", default_timeout=10).run()

    app.radio[0].set_value("Research").run()

    assert not app.exception
    assert any("disabled on this deployment" in item.value for item in app.warning)
    assert not app.button


def test_server_disabled_library_is_also_inaccessible(monkeypatch):
    monkeypatch.setenv("RABBITHOLE_LIVE_ENABLED", "0")
    app = AppTest.from_file(Path(__file__).parents[1] / "streamlit_app.py", default_timeout=10).run()

    app.radio[0].set_value("Research Library").run()

    assert not app.exception
    assert any("Research Library are disabled" in item.value for item in app.warning)
