from types import SimpleNamespace

from rabbithole import browser_storage


def test_browser_storage_surfaces_quota_failure_without_exception_message(monkeypatch):
    captured = {}

    def fake_component(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(payload=None, error="QuotaExceededError", ready=True)

    monkeypatch.setattr(browser_storage, "_browser_storage_component", fake_component)

    result = browser_storage.mount_browser_storage("write", "safe payload", key="test-writer")

    assert result.ready is True
    assert result.error == "QuotaExceededError"
    assert result.payload is None
    assert captured["data"]["storageKey"] == browser_storage.STORAGE_KEY
    assert captured["data"]["payload"] == "safe payload"


def test_browser_storage_rejects_unknown_operation():
    try:
        browser_storage.mount_browser_storage("delete", key="test-invalid")
    except ValueError as exc:
        assert str(exc) == "Unsupported browser storage operation"
    else:
        raise AssertionError("Unsupported operation was accepted")
