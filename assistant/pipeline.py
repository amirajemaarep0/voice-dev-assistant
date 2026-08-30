"""Orchestration: question in, grounded answer out.

Keeps the UI dumb. Everything the app does end-to-end lives here, so the
whole flow can be exercised from a test or a script with no Streamlit.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

from . import config, llm
from .store import ProjectStore, Retrieved


@dataclass
class Answer:
    question: str
    text: str
    sources: list[Retrieved] = field(default_factory=list)

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

    def retrieve(self, question: str) -> list[Retrieved]:
        return self.store.search(question, top_k=self.settings.top_k)

    def stream(self, question: str) -> tuple[list[Retrieved], Iterator[str]]:
        """Return the sources immediately, plus a token stream.

        Returning sources up front lets the UI show what it is reading from
        while the model is still generating.
        """
        chunks = self.retrieve(question)
        prompt = llm.build_prompt(question, chunks)
        stream = llm.stream_answer(
            prompt,
            model=self.settings.model,
            temperature=self.settings.temperature,
        )
        return chunks, stream

    def ask(self, question: str) -> Answer:
        """Blocking convenience call - used by tests and scripts."""
        chunks, stream = self.stream(question)
        return Answer(question=question, text="".join(stream), sources=chunks)
