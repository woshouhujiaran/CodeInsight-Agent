from __future__ import annotations

import io
from urllib import error

import app.llm.llm as llm_module
from app.llm.llm import LLMClient


class _FakeResponse:
    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def read(self) -> bytes:
        return b'{"choices":[{"message":{"content":"ok"}}]}'


def test_openai_provider_uses_configured_base_url(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_urlopen(req, timeout: int = 60):
        captured["url"] = req.full_url
        captured["timeout"] = timeout
        return _FakeResponse()

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.com/custom/v1")
    monkeypatch.setattr(llm_module.request, "urlopen", fake_urlopen)

    client = LLMClient(model="gpt-4o-mini", provider="openai")
    answer = client.generate_text(prompt="hello")

    assert answer == "ok"
    assert captured["url"] == "https://example.com/custom/v1/chat/completions"


def test_llm_client_retries_retryable_http_errors(monkeypatch) -> None:
    calls = {"count": 0}

    def fake_urlopen(req, timeout: int = 60):
        calls["count"] += 1
        if calls["count"] == 1:
            raise error.HTTPError(
                req.full_url,
                503,
                "service unavailable",
                hdrs=None,
                fp=io.BytesIO(b"temporary"),
            )
        return _FakeResponse()

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(llm_module.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(llm_module.time, "sleep", lambda _seconds: None)

    client = LLMClient(model="gpt-4o-mini", provider="openai", max_retries=2)
    answer = client.generate_text(prompt="hello")

    assert answer == "ok"
    assert calls["count"] == 2


def test_llm_client_does_not_retry_non_retryable_http_errors(monkeypatch) -> None:
    calls = {"count": 0}

    def fake_urlopen(req, timeout: int = 60):
        calls["count"] += 1
        raise error.HTTPError(
            req.full_url,
            401,
            "unauthorized",
            hdrs=None,
            fp=io.BytesIO(b"unauthorized"),
        )

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(llm_module.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(llm_module.time, "sleep", lambda _seconds: None)

    client = LLMClient(model="gpt-4o-mini", provider="openai", max_retries=2)
    answer = client.generate_text(prompt="hello")

    assert answer.startswith("已完成分析")
    assert calls["count"] == 1
