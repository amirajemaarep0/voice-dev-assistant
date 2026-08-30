"""Whole-project checks: lint scan and the real test runner.

These shell out to ruff and pytest, so they are slower than the rest of
the suite but still hermetic - everything runs against a tmp_path project,
never the network.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from assistant import config, intents, tools
from assistant.pipeline import Assistant
from assistant.store import build_index

from .conftest import FakeStore

BROKEN = 'print(" HAMDI"\n'


@pytest.fixture
def indexed(sample_project: Path):
    store = FakeStore()
    build_index(sample_project, store)
    return Assistant(store, config.Settings(top_k=2)), sample_project


class TestLintProject:
    def test_finds_an_unused_import(self, sample_project: Path):
        (sample_project / "src" / "dead.py").write_text(
            "import os\n\nx = 1\n", encoding="utf-8"
        )
        issues, available, _ = tools.lint_project(sample_project)
        if not available:
            pytest.skip("ruff unavailable")
        assert any("F401" in i and "src/dead.py" in i for i in issues)

    def test_paths_are_relative_to_the_project(self, sample_project: Path):
        (sample_project / "src" / "dead.py").write_text(
            "import os\n", encoding="utf-8"
        )
        issues, available, _ = tools.lint_project(sample_project)
        if not available:
            pytest.skip("ruff unavailable")
        assert all(not Path(i.split(":")[0]).is_absolute() for i in issues)

    def test_clean_project_reports_nothing(self, tmp_path: Path):
        (tmp_path / "ok.py").write_text("x = 1\n", encoding="utf-8")
        issues, available, _ = tools.lint_project(tmp_path)
        if not available:
            pytest.skip("ruff unavailable")
        assert issues == []

    def test_respects_the_cap(self, tmp_path: Path):
        for n in range(12):
            (tmp_path / f"m{n}.py").write_text("import os\n", encoding="utf-8")
        issues, available, _ = tools.lint_project(tmp_path, max_issues=5)
        if not available:
            pytest.skip("ruff unavailable")
        assert len(issues) <= 5


class TestRunTests:
    def test_green_suite(self, tmp_path: Path):
        (tmp_path / "test_ok.py").write_text(
            "def test_one():\n    assert 1 == 1\n", encoding="utf-8"
        )
        run = tools.run_tests(tmp_path)
        assert run.ran and run.is_green
        assert run.passed == 1
        assert "GREEN" in run.as_text()

    def test_red_suite_reports_the_failure(self, tmp_path: Path):
        (tmp_path / "test_bad.py").write_text(
            "def test_one():\n    assert 1 == 2\n", encoding="utf-8"
        )
        run = tools.run_tests(tmp_path)
        assert run.ran and not run.is_green
        assert run.failed == 1
        assert "test_one" in run.failures
        assert "RED" in run.as_text()

    def test_runtime_error_is_surfaced(self, tmp_path: Path):
        """A NameError no static check would reach."""
        (tmp_path / "test_boom.py").write_text(
            "def test_one():\n    assert undefined_thing() == 1\n", encoding="utf-8"
        )
        run = tools.run_tests(tmp_path)
        assert not run.is_green
        assert "NameError" in run.failures

    def test_no_tests_is_not_a_pass(self, tmp_path: Path):
        (tmp_path / "notatest.py").write_text("x = 1\n", encoding="utf-8")
        run = tools.run_tests(tmp_path)
        assert not run.is_green
        assert "no tests were collected" in run.as_text()

    def test_project_addopts_do_not_hide_the_summary(self, tmp_path: Path):
        """A pytest.ini with its own -q must not suppress the counts."""
        (tmp_path / "pytest.ini").write_text(
            "[pytest]\naddopts = -q\n", encoding="utf-8"
        )
        (tmp_path / "test_ok.py").write_text(
            "def test_one():\n    assert True\n", encoding="utf-8"
        )
        run = tools.run_tests(tmp_path)
        assert run.passed == 1

    def test_missing_folder_is_reported(self, tmp_path: Path):
        run = tools.run_tests(tmp_path / "nope")
        assert not run.ran and "not a folder" in run.error

    def test_timeout_is_reported_not_raised(self, tmp_path: Path):
        (tmp_path / "test_slow.py").write_text(
            "import time\n\n\ndef test_slow():\n    time.sleep(30)\n",
            encoding="utf-8",
        )
        run = tools.run_tests(tmp_path, timeout=2.0)
        assert not run.ran
        assert "exceeded" in run.error


class TestHealthIntent:
    def test_reports_syntax_and_lint_together(self, indexed):
        assistant, root = indexed
        (root / "src" / "bad.py").write_text(BROKEN, encoding="utf-8")
        build_index(root, assistant.store)

        context = assistant.build_context("check the whole project for mistakes")
        assert context.intent.kind == intents.HEALTH
        assert "SYNTAX ERRORS" in context.facts
        assert "src/bad.py" in context.facts
        assert "LINT" in context.facts

    def test_clean_project_says_so(self, indexed):
        assistant, _ = indexed
        context = assistant.build_context("check every file in the project")
        assert "every Python file in the project parses" in context.facts


class TestTestRunIntent:
    def test_runs_the_suite_and_reports(self, indexed):
        assistant, root = indexed
        (root / "test_demo.py").write_text(
            "def test_demo():\n    assert True\n", encoding="utf-8"
        )
        context = assistant.build_context("run the tests")
        assert context.intent.kind == intents.TESTRUN
        assert "TEST RUN" in context.facts
        assert "GREEN" in context.facts

    def test_failing_suite_is_reported_as_red(self, indexed):
        assistant, root = indexed
        (root / "test_demo.py").write_text(
            "def test_demo():\n    assert False\n", encoding="utf-8"
        )
        context = assistant.build_context("do my tests pass?")
        assert "RED" in context.facts


class TestFactsOnlyTasks:
    """Tool-driven answers must not carry unrelated retrieved excerpts."""

    @pytest.mark.parametrize("question", [
        "what is wrong with src/calc.py?",
        "how should src/calc.py be formatted?",
        "check the whole project",
        "run the tests",
    ])
    def test_no_excerpts_are_attached(self, indexed, question):
        assistant, _ = indexed
        context = assistant.build_context(question)
        if not context.facts:
            pytest.skip("tool produced no facts in this environment")
        assert context.chunks == []

    def test_explain_and_test_writing_keep_their_excerpts(self):
        """Only tool-driven tasks are stripped.

        Explaining code and writing a test both benefit from surrounding
        context, so they must not be in the facts-only set.
        """
        from assistant.pipeline import FACTS_ONLY

        assert intents.EXPLAIN not in FACTS_ONLY
        assert intents.TEST not in FACTS_ONLY
        assert intents.GENERAL not in FACTS_ONLY


class TestCleanBranchInstructions:
    """A clean result and a broken one must get different instructions.

    A single instruction covering both has to name the clean outcome, and a
    small model then copies "the file parses cleanly" into an answer that
    has just reported a syntax error - observed in evaluate.py.
    """

    def test_broken_file_is_flagged_unclean(self, indexed):
        assistant, root = indexed
        (root / "src" / "bad.py").write_text(BROKEN, encoding="utf-8")
        build_index(root, assistant.store)
        context = assistant.build_context("what is wrong with src/bad.py?")
        assert context.findings_clean is False

    def test_clean_file_is_flagged_clean(self, indexed):
        assistant, _ = indexed
        context = assistant.build_context("syntax errors in src/calc.py?")
        assert context.findings_clean is True

    def test_no_tool_run_leaves_it_unknown(self, indexed):
        assistant, _ = indexed
        context = assistant.build_context("how does this project work?")
        assert context.findings_clean is None

    def test_green_suite_is_clean(self, indexed):
        assistant, root = indexed
        (root / "test_demo.py").write_text(
            "def test_demo():\n    assert True\n", encoding="utf-8"
        )
        assert assistant.build_context("run the tests").findings_clean is True

    def test_red_suite_is_not_clean(self, indexed):
        assistant, root = indexed
        (root / "test_demo.py").write_text(
            "def test_demo():\n    assert False\n", encoding="utf-8"
        )
        assert assistant.build_context("run the tests").findings_clean is False

    def test_the_error_prompt_never_offers_the_clean_wording(self, indexed):
        from assistant import llm

        assistant, root = indexed
        (root / "src" / "bad.py").write_text(BROKEN, encoding="utf-8")
        build_index(root, assistant.store)
        prompt = llm.build_prompt_for(
            assistant.build_context("what is wrong with src/bad.py?")
        )
        assert "parses cleanly" not in prompt
        assert "BROKEN" in prompt

    def test_instruction_key_falls_back(self):
        from assistant import llm

        assert llm.instruction_key("explain", True) == "explain"
        assert llm.instruction_key("syntax", True) == "syntax_clean"
        assert llm.instruction_key("syntax", False) == "syntax"
        assert llm.instruction_key("syntax", None) == "syntax"
