import json

import pytest

from rabbithole.providers import (
    ProviderError,
    ResilientWebSearchProvider,
    TavilySearchProvider,
    ZhipuStructuredLLM,
    log_safe_exception,
)


class FakeHTTPResponse:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps({"results": [{
            "title": "Market update",
            "url": "https://news.example.com/update",
            "published_date": "2026-09-24",
            "content": "Current market evidence.",
        }]}).encode()


def test_tavily_search_uses_basic_low_credit_request(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured.update(json.loads(request.data.decode()))
        return FakeHTTPResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    results = TavilySearchProvider(api_key="test-key").search("market question", max_results=1)

    assert captured["search_depth"] == "basic"
    assert captured["max_results"] == 1
    assert captured["include_raw_content"] is False
    assert results[0]["url"] == "https://news.example.com/update"
    assert results[0]["source"] == "news.example.com"


def test_zhipu_parses_mocked_structured_response():
    class Completions:
        def create(self, **kwargs):
            message = type("Message", (), {"content": '{"angles": [{"name": "Supply"}]}'})()
            choice = type("Choice", (), {"message": message})()
            return type("Response", (), {"choices": [choice]})()

    provider = ZhipuStructuredLLM.__new__(ZhipuStructuredLLM)
    provider.api_key = "test-secret-key"
    provider.base_url = ""
    provider.client = type("Client", (), {"chat": type("Chat", (), {"completions": Completions()})()})()
    provider.model = "test-model"

    result = provider.complete_json("system", "user")

    assert result["angles"][0]["name"] == "Supply"


def test_zhipu_debug_log_shows_status_but_redacts_key(monkeypatch, capsys):
    class AuthenticationFailure(Exception):
        status_code = 401
        code = "401"
        message = "Authorization: Bearer test-secret-key token expired or incorrect"

    class Completions:
        def create(self, **kwargs):
            raise AuthenticationFailure()

    provider = ZhipuStructuredLLM.__new__(ZhipuStructuredLLM)
    provider.api_key = "test-secret-key"
    provider.base_url = ""
    provider.client = type("Client", (), {"chat": type("Chat", (), {"completions": Completions()})()})()
    provider.model = "glm-test"
    monkeypatch.setenv("RABBITHOLE_DEBUG", "1")

    with pytest.raises(ProviderError, match="Zhipu reasoning request failed"):
        provider.complete_json("system", "user")

    error = capsys.readouterr().err
    assert "status=401" in error
    assert "model=glm-test" in error
    assert "test-secret-key" not in error
    assert "Authorization" not in error
    assert "type=AuthenticationFailure" in error


def test_resilient_tavily_logs_status_without_query_or_secret(capsys):
    class RateLimitFailure(Exception):
        status_code = 429

    class FailingSearch:
        def search(self, query, max_results=1):
            raise RateLimitFailure(f"Bearer private-tavily-key query={query}")

    provider = ResilientWebSearchProvider(FailingSearch())
    results = provider.search("private acquisition research", 1)

    error = capsys.readouterr().err
    assert results == []
    assert "component=Tavily" in error
    assert "stage=live web search" in error
    assert "type=RateLimitFailure" in error
    assert "status=429" in error
    assert "private-tavily-key" not in error
    assert "private acquisition research" not in error


def test_unhandled_exception_log_contains_location_but_no_research_text(capsys):
    try:
        raise RuntimeError("secret-key private uploaded earnings forecast")
    except RuntimeError as exc:
        log_safe_exception("LiveAgent", "run bounded research agent", exc)

    error = capsys.readouterr().err
    assert "component=LiveAgent" in error
    assert "stage=run bounded research agent" in error
    assert "type=RuntimeError" in error
    assert "test_providers.py:" in error
    assert "secret-key" not in error
    assert "private uploaded earnings forecast" not in error
