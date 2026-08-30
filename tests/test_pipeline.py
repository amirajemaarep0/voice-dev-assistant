"""End-to-end flow with a fake store and a mocked model."""
from __future__ import annotations

from assistant import config, llm
from assistant.pipeline import Answer, Assistant

from .conftest import FakeStore


class TestAssistant:
    def test_retrieve_uses_configured_top_k(self, sample_chunks):
        store = FakeStore(results=sample_chunks)
        a = Assistant(store, settings=config.Settings(top_k=1))
        assert len(a.retrieve("how does add work?")) == 1
        assert store.last_query == "how does add work?"

    def test_ask_returns_grounded_answer(self, monkeypatch, sample_chunks):
        monkeypatch.setattr(llm, "stream_answer",
                            lambda *a, **k: iter(["It ", "adds."]))
        a = Assistant(FakeStore(results=sample_chunks))
        result = a.ask("What does add do?")
        assert result.text == "It adds."
        assert result.sources == sample_chunks

    def test_prompt_receives_retrieved_context(self, monkeypatch, sample_chunks):
        captured = {}

        def fake_stream(prompt, **kwargs):
            captured["prompt"] = prompt
            captured["model"] = kwargs.get("model")
            return iter(["ok"])

        monkeypatch.setattr(llm, "stream_answer", fake_stream)
        settings = config.Settings(model="qwen3:1.7b")
        Assistant(FakeStore(results=sample_chunks), settings).ask("q")
        assert "def add(a, b)" in captured["prompt"]
        assert captured["model"] == "qwen3:1.7b"

    def test_empty_index_still_answers(self, monkeypatch):
        monkeypatch.setattr(llm, "stream_answer",
                            lambda *a, **k: iter(["I need a file."]))
        result = Assistant(FakeStore(results=[])).ask("What does add do?")
        assert result.sources == []
        assert result.text

    def test_stream_returns_sources_before_tokens(self, monkeypatch, sample_chunks):
        monkeypatch.setattr(llm, "stream_answer", lambda *a, **k: iter(["x"]))
        chunks, stream = Assistant(FakeStore(results=sample_chunks)).stream("q")
        assert chunks == sample_chunks   # available before consuming the stream
        assert "".join(stream) == "x"


class TestAnswer:
    def test_cited_files_are_deduplicated(self, sample_chunks):
        ans = Answer(question="q", text="t", sources=sample_chunks)
        assert ans.cited_files == ["src/calc.py"]

    def test_no_sources(self):
        assert Answer(question="q", text="t").cited_files == []
