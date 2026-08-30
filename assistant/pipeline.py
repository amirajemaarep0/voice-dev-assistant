"""Orchestration: question in, grounded answer out.

Keeps the UI dumb. Everything the app does end-to-end lives here, so the
whole flow can be exercised from a test or a script with no Streamlit.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

from . import config, intents, llm, tools
from .indexer import extract_file_references
from .intents import Intent
from .store import ProjectStore, Retrieved

# How much of an existing test file to show the model as a style example.
STYLE_EXAMPLE_CHARS = 1400

# Tasks whose answer comes entirely from tool output. Retrieved excerpts add
# nothing here and measurably hurt: they give a small model unrelated code to
# talk about instead of the finding it was handed.
FACTS_ONLY = frozenset({
    intents.SYNTAX,
    intents.STYLE,
    intents.HEALTH,
    intents.TESTRUN,
})


@dataclass
class Context:
    """Everything assembled for one question, before the model runs.

    `facts` is the important field: output from ast and ruff, which is
    correct by construction. The model is asked to explain it, never to
    derive it.
    """

    question: str
    intent: Intent
    chunks: list[Retrieved] = field(default_factory=list)
    missing_files: list[str] = field(default_factory=list)
    facts: str = ""
    symbols: list[tools.Symbol] = field(default_factory=list)
    # True when the tools ran and found nothing wrong, False when they
    # found something, None when no tool ran. Selects which task
    # instruction the model gets: telling it "say the file parses cleanly
    # if there are no errors" makes it echo that phrase under an error.
    findings_clean: bool | None = None


@dataclass
class Answer:
    question: str
    text: str
    sources: list[Retrieved] = field(default_factory=list)
    missing_files: list[str] = field(default_factory=list)
    facts: str = ""
    intent: str = intents.GENERAL

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

    # --- deterministic analysis --------------------------------------
    @property
    def root(self) -> str:
        """The project folder on disk, for tools that need real files."""
        return self.store.indexed_root

    def _syntax_facts(self, named: list[str]) -> tuple[str, bool | None]:
        """Run Python's own parser over the named files, or the whole tree."""
        if not self.root:
            return "", None
        issues: list[tools.SyntaxIssue] = []
        checked: list[str] = []

        targets = [s for s in named if s.lower().endswith(".py")]
        if targets:
            for source in targets:
                path = tools.resolve_in_project(self.root, source)
                if path is None:
                    continue
                checked.append(source)
                issue = tools.check_file(path, display_name=source)
                if issue is not None:
                    issues.append(issue)
        else:
            issues = tools.check_project(self.root)
            checked = ["the whole project"]

        if not checked:
            return "", None
        if not issues:
            return (
                f"SYNTAX CHECK ({', '.join(checked)}): no syntax errors. "
                "Every file parses with Python's own parser."
            ), True
        body = "\n\n".join(i.as_text() for i in issues[:8])
        more = "" if len(issues) <= 8 else f"\n\n(+{len(issues) - 8} more)"
        return (
            f"SYNTAX CHECK - errors found by Python's parser:\n{body}{more}",
            False,
        )

    def _style_facts(self, named: list[str]) -> str:
        """Ask ruff what it would change, without changing anything."""
        if not self.root:
            return ""
        reports = []
        for source in [s for s in named if s.lower().endswith(".py")][:3]:
            path = tools.resolve_in_project(self.root, source)
            if path is not None:
                reports.append(tools.style_report(path, display_name=source))
        if not reports:
            return ""
        return "STYLE REPORT (ruff):\n" + "\n\n".join(r.as_text() for r in reports)

    def _symbol_facts(self, symbol: str) -> tuple[str, list[tools.Symbol]]:
        """Fetch a symbol's exact definition rather than a nearby chunk."""
        if not self.root or not symbol:
            return "", []
        found = tools.find_symbol(self.root, symbol)
        if not found:
            return (
                f"DEFINITION LOOKUP: no function, class or variable named "
                f"'{symbol}' exists in the indexed project. Say so rather "
                "than inventing one."
            ), []
        body = "\n\n".join(s.as_text() for s in found)
        return f"EXACT DEFINITION(S) from the project:\n{body}", found

    def _test_style_example(self) -> str:
        """A slice of a real test file, so generated tests match the project."""
        for source in self.store.sources():
            # A real test file, not conftest.py: the model should copy how
            # tests are written here, not how fixtures are.
            if source.startswith("tests/") and source.rsplit("/", 1)[-1].startswith(
                "test_"
            ):
                path = tools.resolve_in_project(self.root, source)
                if path is None:
                    continue
                from .indexer import read_text

                text = read_text(path)
                if text:
                    return (
                        f"HOUSE TEST STYLE - an existing test file "
                        f"({source}), match its conventions:\n"
                        f"{text[:STYLE_EXAMPLE_CHARS]}"
                    )
        return ""

    def _health_facts(self) -> tuple[str, bool | None]:
        """Everything statically wrong with the project, in one report."""
        if not self.root:
            return "", None
        parts: list[str] = []
        clean = True

        syntax = tools.check_project(self.root)
        clean = clean and not syntax
        if syntax:
            body = "\n\n".join(i.as_text() for i in syntax[:8])
            more = "" if len(syntax) <= 8 else f"\n\n(+{len(syntax) - 8} more)"
            parts.append(
                f"SYNTAX ERRORS ({len(syntax)} file(s) will not parse):\n"
                f"{body}{more}"
            )
        else:
            parts.append("SYNTAX: every Python file in the project parses.")

        issues, available, error = tools.lint_project(self.root)
        clean = clean and available and not issues
        if not available:
            parts.append(f"LINT: unavailable ({error})")
        elif issues:
            parts.append(
                f"LINT FINDINGS ({len(issues)} shown, ruff E/W/F/I):\n"
                + "\n".join(issues)
            )
        else:
            parts.append("LINT: ruff reports no findings.")
        return "\n\n".join(parts), clean

    def _testrun_facts(self) -> tuple[str, bool | None]:
        """Actually run the suite - the only way to see a runtime error."""
        if not self.root:
            return "", None
        run = tools.run_tests(self.root)
        return run.as_text(), run.is_green

    def build_context(self, question: str) -> Context:
        """Classify the question, then assemble exactly what it needs."""
        intent = intents.detect_intent(question)
        chunks, missing = self.gather(question)
        named, _ = self.resolve_files(question)

        facts = ""
        clean: bool | None = None
        symbols: list[tools.Symbol] = []
        if intent.kind == intents.SYNTAX:
            facts, clean = self._syntax_facts(named)
        elif intent.kind == intents.HEALTH:
            facts, clean = self._health_facts()
        elif intent.kind == intents.TESTRUN:
            facts, clean = self._testrun_facts()
        elif intent.kind == intents.STYLE:
            facts = self._style_facts(named)
        elif intent.needs_symbol and intent.symbol:
            facts, symbols = self._symbol_facts(intent.symbol)
            if intent.kind == intents.TEST:
                example = self._test_style_example()
                facts = f"{facts}\n\n{example}".strip() if example else facts

        # A named symbol's real definition beats a retrieved chunk of it.
        for symbol in symbols:
            chunks = [
                c for c in chunks
                if not (c.source == symbol.file and symbol.source in c.text)
            ]

        # For a tool-driven question the retrieved excerpts are noise: the
        # answer is entirely determined by the tool output, and unrelated
        # code in the prompt is what makes a small model wander off and
        # cite tests that have nothing to do with the question.
        if facts and intent.kind in FACTS_ONLY:
            chunks = []

        return Context(
            question=question,
            intent=intent,
            chunks=chunks,
            missing_files=missing,
            facts=facts,
            symbols=symbols,
            findings_clean=clean,
        )

    def stream(self, question: str) -> tuple[list[Retrieved], Iterator[str]]:
        """Return the sources immediately, plus a token stream.

        Returning sources up front lets the UI show what it is reading from
        while the model is still generating.
        """
        context = self.build_context(question)
        stream = llm.stream_answer(
            llm.build_prompt_for(context),
            model=self.settings.model,
            temperature=self.settings.temperature,
        )
        return context.chunks, stream

    def stream_context(self, question: str) -> tuple[Context, Iterator[str]]:
        """Like `stream`, but hands back the full context.

        The UI shows the deterministic findings next to the model's prose,
        so the user can see that the syntax error came from Python and not
        from a 1.7B model's guess.
        """
        context = self.build_context(question)
        stream = llm.stream_answer(
            llm.build_prompt_for(context),
            model=self.settings.model,
            temperature=self.settings.temperature,
        )
        return context, stream

    def ask(self, question: str) -> Answer:
        """Blocking convenience call - used by tests and scripts."""
        context = self.build_context(question)
        text = "".join(
            llm.stream_answer(
                llm.build_prompt_for(context),
                model=self.settings.model,
                temperature=self.settings.temperature,
            )
        )
        return Answer(
            question=question,
            text=text,
            sources=context.chunks,
            missing_files=context.missing_files,
            facts=context.facts,
            intent=context.intent.kind,
        )
