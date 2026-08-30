"""Orchestration: question in, grounded answer out.

Keeps the UI dumb. Everything the app does end-to-end lives here, so the
whole flow can be exercised from a test or a script with no Streamlit.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

from . import config, llm
from .indexer import extract_file_references
from .store import ProjectStore, Retrieved


@dataclass
class Answer:
    question: str
    text: str
    sources: list[Retrieved] = field(default_factory=list)
    missing_files: list[str] = field(default_factory=list)

    @property
    def cited_files(self) -> list[str]:
        seen: list[str] = []
        for s in self.sources:
            if s.source not in seen:
                seen.append(s.source)
        return seen


class Assistant:
    """Retrieval-augmented question answering over an indexed project."""

    def __init__(
        self,
        store: ProjectStore,
        settings: config.Settings | None = None,
    ) -> None:
        self.store = store
        self.settings = settings or config.Settings()

    def resolve_files(self, question: str) -> tuple[list[str], list[str]]:
        """Filenames the question names, split into indexed and missing."""
        return extract_file_references(question, self.store.sources())

    def retrieve(self, question: str) -> list[Retrieved]:
        """The chunks a question should be answered from."""
        return self.gather(question)[0]

    def gather(self, question: str) -> tuple[list[Retrieved], list[str]]:
        """Gather context for a question, plus any filenames it names in vain.

        A named file is fetched whole by metadata rather than left to
        similarity search, which reliably misses it: "what is in stt.py?"
        shares almost no vocabulary with stt.py's contents. Similarity hits
        then fill whatever budget is left, so questions that name a file and
        ask something broader still get both.
        """
        named, missing = self.resolve_files(question)

        chunks: list[Retrieved] = []
        seen: set[tuple[str, int]] = set()
        for source in named:
            for chunk in self.store.chunks_for_source(source):
                key = (chunk.source, chunk.position)
                if key not in seen:
                    seen.add(key)
                    chunks.append(chunk)

        budget = max(self.settings.top_k, len(chunks) + self.settings.top_k)
        for chunk in self.store.search(question, top_k=self.settings.top_k):
            if len(chunks) >= budget:
                break
            key = (chunk.source, chunk.position)
            if key not in seen:
                seen.add(key)
                chunks.append(chunk)
        return chunks, missing

    def stream(self, question: str) -> tuple[list[Retrieved], Iterator[str]]:
        """Return the sources immediately, plus a token stream.

        Returning sources up front lets the UI show what it is reading from
        while the model is still generating.
        """
        chunks, missing = self.gather(question)
        prompt = llm.build_prompt(question, chunks, missing_files=missing)
        stream = llm.stream_answer(
            prompt,
            model=self.settings.model,
            temperature=self.settings.temperature,
        )
        return chunks, stream

    def ask(self, question: str) -> Answer:
        """Blocking convenience call - used by tests and scripts."""
        chunks, missing = self.gather(question)
        prompt = llm.build_prompt(question, chunks, missing_files=missing)
        text = "".join(
            llm.stream_answer(
                prompt,
                model=self.settings.model,
                temperature=self.settings.temperature,
            )
        )
        return Answer(
            question=question, text=text, sources=chunks, missing_files=missing
        )
