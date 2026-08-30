"""Shared fixtures. Nothing here touches Ollama, Whisper or the network."""
from __future__ import annotations

from pathlib import Path

import pytest

from assistant.store import Retrieved

SAMPLE_PY = '''
def add(a, b):
    """Return the sum of a and b."""
    return a + b


class Calculator:
    def __init__(self):
        self.total = 0

    def accumulate(self, value):
        self.total += value
        return self.total
'''


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """Keep the suite hermetic.

    `stream_answer` asks Ollama for model capabilities. On a dev machine
    Ollama is often actually running, which would make these tests hit a
    live server. Stub it out for every test; the capability tests use the
    real function via a reference captured at import time.
    """
    from assistant import llm

    llm._CAPS_CACHE.clear()
    monkeypatch.setattr(
        llm, "model_capabilities", lambda model, host=llm.config.OLLAMA_HOST: set()
    )


@pytest.fixture
def sample_project(tmp_path: Path) -> Path:
    """A small on-disk project with noise that must be ignored."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "calc.py").write_text(SAMPLE_PY, encoding="utf-8")
    (tmp_path / "src" / "notes.md").write_text("# Notes\n\nSome docs.\n",
                                               encoding="utf-8")
    (tmp_path / "README.md").write_text("# Demo project\n", encoding="utf-8")

    # --- noise that the indexer must skip ---
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "lib.js").write_text("var x=1;", encoding="utf-8")
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "site.py").write_text("# venv", encoding="utf-8")
    (tmp_path / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    return tmp_path


class FakeStore:
    """In-memory stand-in for ProjectStore - no chromadb, no embeddings."""

    def __init__(self, results: list[Retrieved] | None = None) -> None:
        self.ids: list[str] = []
        self.documents: list[str] = []
        self.metadatas: list[dict] = []
        self.reset_calls = 0
        self._results = results or []
        self.last_query: str | None = None
        self.indexed_root: str = ""
        self.indexed_fingerprint: str = ""
        self.root_writes: list[str] = []

    def reset(self) -> None:
        self.reset_calls += 1
        self.ids.clear()
        self.documents.clear()
        self.metadatas.clear()

    def set_indexed_root(self, root, fingerprint: str = "") -> None:
        self.indexed_root = str(Path(root).resolve()) if root else ""
        self.indexed_fingerprint = fingerprint if root else ""
        self.root_writes.append(self.indexed_root)

    def sources(self) -> list[str]:
        """Distinct file paths, from indexed metadata or canned results."""
        known = {m["source"] for m in self.metadatas}
        known.update(r.source for r in self._results)
        return sorted(known)

    def chunks_for_source(self, source: str, limit: int = 8) -> list[Retrieved]:
        indexed = [
            Retrieved(text=doc, source=meta["source"],
                      position=meta["position"], distance=0.0)
            for doc, meta in zip(self.documents, self.metadatas)
            if meta["source"] == source
        ]
        canned = [
            Retrieved(text=r.text, source=r.source, position=r.position,
                      distance=0.0)
            for r in self._results
            if r.source == source
        ]
        return (indexed or canned)[:limit]

    def add(self, ids, documents, metadatas) -> None:
        self.ids.extend(ids)
        self.documents.extend(documents)
        self.metadatas.extend(metadatas)

    def count(self) -> int:
        return len(self.ids)

    def search(self, query: str, top_k: int = 4) -> list[Retrieved]:
        self.last_query = query
        return self._results[:top_k]


@pytest.fixture
def fake_store() -> FakeStore:
    return FakeStore()


@pytest.fixture
def sample_chunks() -> list[Retrieved]:
    return [
        Retrieved(text="def add(a, b):\n    return a + b",
                  source="src/calc.py", position=0, distance=0.12),
        Retrieved(text="class Calculator:\n    pass",
                  source="src/calc.py", position=1, distance=0.31),
    ]
