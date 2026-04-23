from __future__ import annotations

import json

import numpy as np

import app.rag.embeddings as emb_module
from app.rag.embeddings import OllamaEmbedding, create_embedding_backend


class _FakeResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


def test_ollama_embedding_calls_embed_api_and_normalizes(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_urlopen(req, timeout: int = 60):
        captured["url"] = req.full_url
        captured["timeout"] = timeout
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _FakeResponse({"embeddings": [[3.0, 4.0], [0.0, 5.0]]})

    monkeypatch.setattr(emb_module.request, "urlopen", fake_urlopen)

    backend = OllamaEmbedding(model="nomic-embed-text", base_url="http://127.0.0.1:11434")
    out = backend.embed_texts(["a", "b"])

    assert captured["url"] == "http://127.0.0.1:11434/api/embed"
    assert captured["body"] == {"model": "nomic-embed-text", "input": ["a", "b"]}
    assert out.shape == (2, 2)
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), np.ones(2), rtol=1e-6, atol=1e-6)


def test_create_embedding_backend_selects_ollama(monkeypatch) -> None:
    monkeypatch.setenv("EMBEDDING_BACKEND", "ollama")
    monkeypatch.setenv("OLLAMA_EMBEDDING_MODEL", "bge-m3")
    backend = create_embedding_backend()

    assert isinstance(backend, OllamaEmbedding)
    assert backend.model == "bge-m3"
