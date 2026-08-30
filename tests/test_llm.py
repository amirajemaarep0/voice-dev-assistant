"""Prompt assembly and the Ollama client, with the network mocked out."""
from __future__ import annotations

import json

import pytest
import requests

from assistant import llm

# Captured before the autouse `offline` fixture replaces it.
REAL_MODEL_CAPS = llm.model_capabilities


class TestIsLocal:
    def test_local_models_pass(self):
        assert llm._is_local("qwen3:1.7b")
        assert llm._is_local("llama3:latest")

    def test_cloud_models_are_rejected_by_name(self):
        # Sending code to a cloud model would break the whole premise.
        assert not llm._is_local("gpt-oss:120b-cloud")

    def test_cloud_models_are_rejected_by_remote_host(self):
        # A remote model that is NOT named "-cloud" must still be caught.
        entry = {"name": "innocent:7b", "remote_host": "https://ollama.com:443"}
        assert not llm._is_local("innocent:7b", entry)

    def test_remote_model_field_is_caught(self):
        entry = {"name": "innocent:7b", "remote_model": "gpt-oss:120b"}
        assert not llm._is_local("innocent:7b", entry)

    def test_plain_local_entry_passes(self):
        entry = {"name": "qwen3:1.7b", "size": 1359293444}
        assert llm._is_local("qwen3:1.7b", entry)


class TestBuildPrompt:
    def test_includes_question_and_code(self, sample_chunks):
        prompt = llm.build_prompt("What does add do?", sample_chunks)
        assert "What does add do?" in prompt
        assert "def add(a, b)" in prompt

    def test_includes_file_labels_for_citation(self, sample_chunks):
        prompt = llm.build_prompt("q", sample_chunks)
        assert "src/calc.py" in prompt

    def test_no_chunks_produces_honest_fallback(self):
        prompt = llm.build_prompt("What does add do?", [])
        assert "No relevant code" in prompt
        assert "What does add do?" in prompt

    def test_all_chunks_present(self, sample_chunks):
        prompt = llm.build_prompt("q", sample_chunks)
        for chunk in sample_chunks:
            assert chunk.text in prompt


class _FakeResponse:
    def __init__(self, lines, status_ok=True):
        self._lines = lines
        self._ok = status_ok

    def raise_for_status(self):
        if not self._ok:
            raise requests.HTTPError("500")

    def iter_lines(self):
        yield from self._lines

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _tokens(*words, done=True):
    lines = [json.dumps({"response": w, "done": False}).encode() for w in words]
    if done:
        lines.append(json.dumps({"response": "", "done": True}).encode())
    return lines


class TestStreamAnswer:
    def test_yields_tokens_in_order(self, monkeypatch):
        monkeypatch.setattr(
            llm.requests, "post",
            lambda *a, **k: _FakeResponse(_tokens("Hello", " ", "world")),
        )
        assert llm.answer("prompt") == "Hello world"

    def test_skips_malformed_lines(self, monkeypatch):
        lines = [b"not json", *_tokens("ok")]
        monkeypatch.setattr(llm.requests, "post",
                            lambda *a, **k: _FakeResponse(lines))
        assert llm.answer("prompt") == "ok"

    def test_server_error_field_raises(self, monkeypatch):
        lines = [json.dumps({"error": "model not found"}).encode()]
        monkeypatch.setattr(llm.requests, "post",
                            lambda *a, **k: _FakeResponse(lines))
        with pytest.raises(llm.OllamaError, match="model not found"):
            llm.answer("prompt")

    def test_connection_failure_raises_ollama_error(self, monkeypatch):
        def boom(*a, **k):
            raise requests.ConnectionError("refused")

        monkeypatch.setattr(llm.requests, "post", boom)
        with pytest.raises(llm.OllamaError):
            llm.answer("prompt")


class TestListLocalModels:
    def test_filters_cloud_models(self, monkeypatch):
        class R:
            @staticmethod
            def raise_for_status():
                return None

            @staticmethod
            def json():
                return {"models": [{"name": "qwen3:1.7b"},
                                   {"name": "gpt-oss:120b-cloud"},
                                   {"name": "llama3:latest"}]}

        monkeypatch.setattr(llm.requests, "get", lambda *a, **k: R())
        assert llm.list_local_models() == ["llama3:latest", "qwen3:1.7b"]

    def test_unreachable_server_raises(self, monkeypatch):
        def boom(*a, **k):
            raise requests.ConnectionError("refused")

        monkeypatch.setattr(llm.requests, "get", boom)
        with pytest.raises(llm.OllamaError):
            llm.list_local_models()
        assert llm.is_available() is False


class _Recorder:
    """Captures the JSON payload sent to /api/generate."""

    def __init__(self):
        self.payload = None

    def __call__(self, url, json=None, **kwargs):
        self.payload = json
        return _FakeResponse(_tokens("ok"))


class TestGenerationPayload:
    def test_thinking_is_disabled_for_reasoning_models(self, monkeypatch):
        # qwen3 spends ~100s on a hidden reasoning pass before the first
        # visible token. For grounded Q&A that is pure latency.
        monkeypatch.setattr(llm, "model_capabilities",
                            lambda m, host=None: {"completion", "thinking"})
        rec = _Recorder()
        monkeypatch.setattr(llm.requests, "post", rec)
        llm.answer("p", model="qwen3:1.7b")
        assert rec.payload["think"] is False

    def test_think_key_absent_for_plain_models(self, monkeypatch):
        # Sending `think` to a model without the capability makes Ollama 400.
        monkeypatch.setattr(llm, "model_capabilities",
                            lambda m, host=None: {"completion"})
        rec = _Recorder()
        monkeypatch.setattr(llm.requests, "post", rec)
        llm.answer("p", model="llama3:latest")
        assert "think" not in rec.payload

    def test_keep_alive_and_context_are_set(self, monkeypatch):
        rec = _Recorder()
        monkeypatch.setattr(llm.requests, "post", rec)
        llm.answer("p")
        assert rec.payload["keep_alive"] == llm.config.KEEP_ALIVE
        assert rec.payload["options"]["num_ctx"] == llm.config.NUM_CTX

    def test_system_prompt_is_sent(self, monkeypatch):
        rec = _Recorder()
        monkeypatch.setattr(llm.requests, "post", rec)
        llm.answer("p")
        assert rec.payload["system"] == llm.SYSTEM_PROMPT
        assert rec.payload["stream"] is True

    def test_temperature_is_forwarded(self, monkeypatch):
        rec = _Recorder()
        monkeypatch.setattr(llm.requests, "post", rec)
        llm.answer("p", temperature=0.7)
        assert rec.payload["options"]["temperature"] == 0.7


class TestModelCapabilities:
    def test_reads_capabilities_from_tags(self, monkeypatch):
        class R:
            @staticmethod
            def raise_for_status():
                return None

            @staticmethod
            def json():
                return {"models": [
                    {"name": "qwen3:1.7b",
                     "capabilities": ["completion", "thinking"]},
                    {"name": "llama3:latest", "capabilities": ["completion"]},
                ]}

        llm._CAPS_CACHE.clear()
        monkeypatch.setattr(llm.requests, "get", lambda *a, **k: R())
        assert "thinking" in REAL_MODEL_CAPS("qwen3:1.7b")
        assert "thinking" not in REAL_MODEL_CAPS("llama3:latest")

    def test_unreachable_server_returns_empty(self, monkeypatch):
        def boom(*a, **k):
            raise requests.ConnectionError("refused")

        llm._CAPS_CACHE.clear()
        monkeypatch.setattr(llm.requests, "get", boom)
        assert REAL_MODEL_CAPS("qwen3:1.7b") == set()
