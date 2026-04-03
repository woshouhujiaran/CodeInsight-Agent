from __future__ import annotations

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
