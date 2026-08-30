"""The deterministic half of the assistant: ast, symbol lookup, ruff.

These are the parts that must be exactly right, because the model is told
to trust them over its own reading of the code.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from assistant import tools

BROKEN = 'print(" HAMDI"\n'
GOOD = '''
def average(numbers):
    """Mean of a sequence."""
    return sum(numbers) / len(numbers)


TOTAL = 0
LIMIT: int = 10


class Stats:
    """Running statistics."""

    def __init__(self, values):
        self.values = values

    def mean(self):
        return average(self.values)
'''


class TestCheckSyntax:
    def test_clean_source_returns_none(self):
        assert tools.check_syntax(GOOD, "good.py") is None

    def test_finds_the_unclosed_bracket(self):
        issue = tools.check_syntax(BROKEN, "broken.py")
        assert issue is not None
        assert issue.line == 1
        assert "never closed" in issue.message

    def test_reports_the_filename_it_was_given(self):
        issue = tools.check_syntax(BROKEN, "samples/broken_function.py")
        assert issue.file == "samples/broken_function.py"

    def test_as_text_shows_the_offending_line_and_a_caret(self):
        text = tools.check_syntax(BROKEN, "broken.py").as_text()
        assert 'print(" HAMDI"' in text
        assert "^" in text
        assert "line 1" in text

    def test_empty_source_is_valid_python(self):
        assert tools.check_syntax("", "empty.py") is None


class TestCheckFile:
    def test_non_python_files_are_skipped(self, tmp_path: Path):
        md = tmp_path / "notes.md"
        md.write_text("# not python (", encoding="utf-8")
        assert tools.check_file(md) is None

    def test_reads_a_broken_file(self, tmp_path: Path):
        path = tmp_path / "bad.py"
        path.write_text(BROKEN, encoding="utf-8")
        assert tools.check_file(path) is not None

    def test_missing_file_reports_rather_than_raises(self, tmp_path: Path):
        issue = tools.check_file(tmp_path / "nope.py")
        assert issue is not None and issue.line == 0


class TestCheckProject:
    def test_clean_project_has_no_issues(self, sample_project: Path):
        assert tools.check_project(sample_project) == []

    def test_finds_the_one_broken_file(self, sample_project: Path):
        (sample_project / "src" / "bad.py").write_text(BROKEN, encoding="utf-8")
        issues = tools.check_project(sample_project)
        assert [i.file for i in issues] == ["src/bad.py"]

    def test_ignored_directories_are_not_checked(self, sample_project: Path):
        (sample_project / ".venv" / "bad.py").write_text(BROKEN, encoding="utf-8")
        assert tools.check_project(sample_project) == []


class TestIterSymbols:
    def test_finds_functions_classes_methods_and_variables(self):
        found = {(s.qualname, s.kind) for s in tools.iter_symbols(GOOD, "m.py")}
        assert ("average", "function") in found
        assert ("Stats", "class") in found
        assert ("Stats.mean", "method") in found
        assert ("TOTAL", "variable") in found

    def test_annotated_assignment_is_a_variable(self):
        found = {s.qualname for s in tools.iter_symbols(GOOD, "m.py")}
        assert "LIMIT" in found

    def test_source_is_the_whole_definition(self):
        average = next(
            s for s in tools.iter_symbols(GOOD, "m.py") if s.name == "average"
        )
        assert average.source.startswith("def average(numbers):")
        assert "sum(numbers)" in average.source

    def test_docstring_is_captured(self):
        average = next(
            s for s in tools.iter_symbols(GOOD, "m.py") if s.name == "average"
        )
        assert average.doc == "Mean of a sequence."

    def test_unparseable_source_yields_nothing_instead_of_raising(self):
        assert tools.iter_symbols(BROKEN, "broken.py") == []


class TestFindSymbol:
    def test_locates_a_function_across_the_project(self, sample_project: Path):
        found = tools.find_symbol(sample_project, "add")
        assert found and found[0].file == "src/calc.py"
        assert "def add(a, b)" in found[0].source

    def test_locates_a_method_by_qualified_name(self, sample_project: Path):
        found = tools.find_symbol(sample_project, "Calculator.accumulate")
        assert found and found[0].kind == "method"

    def test_unknown_name_returns_empty(self, sample_project: Path):
        assert tools.find_symbol(sample_project, "does_not_exist") == []

    def test_is_case_insensitive(self, sample_project: Path):
        assert tools.find_symbol(sample_project, "CALCULATOR")

    def test_trailing_parens_are_tolerated(self, sample_project: Path):
        assert tools.find_symbol(sample_project, "add()")


class TestResolveInProject:
    def test_returns_the_real_path(self, sample_project: Path):
        assert tools.resolve_in_project(sample_project, "src/calc.py") is not None

    def test_returns_none_for_a_missing_file(self, sample_project: Path):
        assert tools.resolve_in_project(sample_project, "src/nope.py") is None

    def test_returns_none_without_a_root(self):
        assert tools.resolve_in_project("", "src/calc.py") is None


class TestStyleReport:
    """Requires ruff, which is a declared dependency."""

    def test_messy_file_produces_a_diff(self, tmp_path: Path):
        path = tmp_path / "messy.py"
        path.write_text("def  f( a,b ):\n  return a+b\n", encoding="utf-8")
        report = tools.style_report(path, display_name="messy.py")
        if not report.available:
            pytest.skip(f"ruff unavailable: {report.error}")
        assert not report.is_clean
        assert "def f(a, b):" in report.diff

    def test_unused_import_is_reported(self, tmp_path: Path):
        path = tmp_path / "unused.py"
        path.write_text("import os\n\nx = 1\n", encoding="utf-8")
        report = tools.style_report(path)
        if not report.available:
            pytest.skip(f"ruff unavailable: {report.error}")
        assert any("F401" in issue for issue in report.issues)

    def test_clean_file_says_so(self, tmp_path: Path):
        path = tmp_path / "clean.py"
        path.write_text("x = 1\n", encoding="utf-8")
        report = tools.style_report(path, display_name="clean.py")
        if not report.available:
            pytest.skip(f"ruff unavailable: {report.error}")
        assert report.is_clean
        assert "already formatted" in report.as_text()
