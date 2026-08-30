"""The four coding tasks, end to end but with the model stubbed out.

The point of these is that the *facts* handed to the model are correct and
complete before it ever runs - that is what makes a 1.7B model usable for
code questions.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from assistant import config, intents, llm
from assistant.pipeline import Assistant
from assistant.store import build_index

from .conftest import FakeStore

BROKEN = 'print(" HAMDI"\n'


@pytest.fixture
def indexed(sample_project: Path):
    """A FakeStore whose indexed_root points at a real folder on disk."""
    store = FakeStore()
    build_index(sample_project, store)
    return Assistant(store, config.Settings(top_k=2)), sample_project


class TestSyntaxTask:
    def test_reports_the_error_with_file_and_line(self, indexed):
        assistant, root = indexed
        (root / "src" / "bad.py").write_text(BROKEN, encoding="utf-8")
        build_index(root, assistant.store)

        context = assistant.build_context("what is wrong with src/bad.py?")
        assert context.intent.kind == intents.SYNTAX
        assert "src/bad.py" in context.facts
        assert "never closed" in context.facts

    def test_clean_file_is_reported_as_clean(self, indexed):
        assistant, _ = indexed
        context = assistant.build_context("is there a syntax error in src/calc.py?")
        assert "no syntax errors" in context.facts

    def test_whole_project_is_checked_when_no_file_is_named(self, indexed):
        assistant, root = indexed
        (root / "src" / "bad.py").write_text(BROKEN, encoding="utf-8")
        build_index(root, assistant.store)

        context = assistant.build_context("are there any errors in my project?")
        assert "src/bad.py" in context.facts

    def test_facts_reach_the_prompt_as_authoritative(self, indexed):
        assistant, root = indexed
        (root / "src" / "bad.py").write_text(BROKEN, encoding="utf-8")
        build_index(root, assistant.store)

        prompt = llm.build_prompt_for(
            assistant.build_context("what is wrong with src/bad.py?")
        )
        assert "never closed" in prompt
        assert "takes precedence" in prompt


class TestStyleTask:
    def test_reports_what_ruff_would_change(self, indexed):
        assistant, root = indexed
        (root / "src" / "messy.py").write_text(
            "def  f( a,b ):\n  return a+b\n", encoding="utf-8"
        )
        build_index(root, assistant.store)

        context = assistant.build_context("how should src/messy.py be formatted?")
        assert context.intent.kind == intents.STYLE
        if "not installed" in context.facts:
            pytest.skip("ruff unavailable")
        assert "def f(a, b):" in context.facts


class TestExplainTask:
    def test_uses_the_exact_definition_not_a_chunk(self, indexed):
        assistant, _ = indexed
        context = assistant.build_context("explain the function add")
        assert context.intent.kind == intents.EXPLAIN
        assert "EXACT DEFINITION" in context.facts
        assert "def add(a, b):" in context.facts
        assert context.symbols and context.symbols[0].name == "add"

    def test_unknown_symbol_is_admitted_not_invented(self, indexed):
        assistant, _ = indexed
        context = assistant.build_context("explain the function frobnicate")
        assert "no function, class or variable named" in context.facts
        assert context.symbols == []

    def test_a_plain_question_runs_no_tools(self, indexed):
        assistant, _ = indexed
        context = assistant.build_context("how does the project work?")
        assert context.facts == ""


class TestUnitTestTask:
    def test_definition_and_house_style_are_supplied(self, indexed):
        assistant, root = indexed
        (root / "tests").mkdir(exist_ok=True)
        (root / "tests" / "test_calc.py").write_text(
            "class TestAdd:\n    def test_adds(self):\n        assert add(1, 2) == 3\n",
            encoding="utf-8",
        )
        build_index(root, assistant.store)

        context = assistant.build_context("write a unit test for add")
        assert context.intent.kind == intents.TEST
        assert "def add(a, b):" in context.facts
        assert "HOUSE TEST STYLE" in context.facts
        assert "class TestAdd" in context.facts

    def test_works_without_any_existing_tests(self, indexed):
        assistant, _ = indexed
        context = assistant.build_context("write a unit test for add")
        assert "def add(a, b):" in context.facts
        assert "HOUSE TEST STYLE" not in context.facts

    def test_prompt_asks_for_a_pytest_test(self, indexed):
        assistant, _ = indexed
        prompt = llm.build_prompt_for(
            assistant.build_context("write a unit test for add")
        )
        assert "pytest test" in prompt


class TestAnswerCarriesTheTask:
    def test_answer_records_intent_and_facts(self, indexed, monkeypatch):
        assistant, _ = indexed
        monkeypatch.setattr(llm, "stream_answer", lambda *a, **k: iter(["ok"]))
        answer = assistant.ask("explain the function add")
        assert answer.intent == intents.EXPLAIN
        assert "def add(a, b):" in answer.facts

    def test_stream_context_exposes_the_same_context(self, indexed, monkeypatch):
        assistant, _ = indexed
        monkeypatch.setattr(llm, "stream_answer", lambda *a, **k: iter(["ok"]))
        context, stream = assistant.stream_context("explain the function add")
        assert context.intent.kind == intents.EXPLAIN
        assert "".join(stream) == "ok"


class TestToolsAreSkippedWithoutARoot:
    """A FakeStore with no indexed_root must not crash the pipeline."""

    def test_no_root_means_no_facts(self):
        assistant = Assistant(FakeStore(), config.Settings(top_k=1))
        context = assistant.build_context("what is wrong with app.py?")
        assert context.facts == ""
