"""Phase 1d - deterministic code intelligence, no model involved.

Everything in this module produces facts rather than opinions: `ast` knows
exactly where a syntax error is, and ruff knows exactly what it would
reformat. That matters more here than it would with a frontier model - a
1.7B model asked to find the bug in `print(" HAMDI"` will cheerfully quote
it back with the bracket closed and report the file as fine.

So the model is never asked to *find* these things. It is handed the exact
finding and asked to explain and fix it, which is what a small model is
actually good at.
"""
from __future__ import annotations

import ast
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Symbols that would drown the answer if we returned every match.
MAX_SYMBOL_MATCHES = 3
RUFF_SELECT = "E,W,F,I"
RUFF_TIMEOUT = 20.0


# --------------------------------------------------------------- syntax
@dataclass
class SyntaxIssue:
    """A syntax error, located precisely."""

    file: str
    line: int
    column: int
    message: str
    source_line: str = ""

    def as_text(self) -> str:
        where = f"{self.file}, line {self.line}"
        if self.column:
            where += f", column {self.column}"
        out = f"{where}: {self.message}"
        if self.source_line.strip():
            caret = " " * max(self.column - 1, 0) + "^"
            out += f"\n    {self.source_line.rstrip()}\n    {caret}"
        return out


def check_syntax(source: str, filename: str = "<unknown>") -> SyntaxIssue | None:
    """Parse Python source. Returns the first syntax error, or None.

    `ast.parse` is the same parser CPython uses, so a file that passes here
    is a file that will import.
    """
    try:
        ast.parse(source, filename=filename)
    except SyntaxError as exc:
        return SyntaxIssue(
            file=filename,
            line=exc.lineno or 0,
            column=exc.offset or 0,
            message=exc.msg or "invalid syntax",
            source_line=exc.text or "",
        )
    except ValueError as exc:  # e.g. source containing null bytes
        return SyntaxIssue(file=filename, line=0, column=0, message=str(exc))
    return None


def check_file(path: Path | str, display_name: str | None = None) -> SyntaxIssue | None:
    """Syntax-check one file on disk. Non-Python files are skipped."""
    path = Path(path)
    if path.suffix.lower() != ".py":
        return None
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return SyntaxIssue(
            file=display_name or path.name, line=0, column=0, message=str(exc)
        )
    return check_syntax(source, filename=display_name or path.name)


def check_project(root: Path | str) -> list[SyntaxIssue]:
    """Syntax-check every Python file under `root`."""
    from .indexer import iter_source_files

    root = Path(root)
    issues: list[SyntaxIssue] = []
    for path in iter_source_files(root, extensions={".py"}):
        rel = str(path.relative_to(root)).replace("\\", "/")
        issue = check_file(path, display_name=rel)
        if issue is not None:
            issues.append(issue)
    return issues


# --------------------------------------------------------------- symbols
@dataclass
class Symbol:
    """A function, class or module-level variable, with its exact source."""

    name: str
    kind: str            # "function" | "method" | "class" | "variable"
    file: str
    line: int
    source: str
    doc: str = ""
    qualname: str = ""

    def __post_init__(self) -> None:
        if not self.qualname:
            self.qualname = self.name

    def as_text(self) -> str:
        return (
            f"--- {self.kind.upper()} {self.qualname} "
            f"({self.file}, line {self.line}) ---\n{self.source}"
        )


def _segment(source: str, node: ast.AST) -> str:
    segment = ast.get_source_segment(source, node)
    if segment:
        return segment
    # get_source_segment needs end positions; fall back to a line slice.
    start = getattr(node, "lineno", 1) - 1
    end = getattr(node, "end_lineno", start + 1)
    return "\n".join(source.splitlines()[start:end])


def iter_symbols(source: str, filename: str = "<unknown>") -> list[Symbol]:
    """Every top-level function, class, method and module variable.

    Returns [] rather than raising when the file does not parse - a broken
    file should still be answerable about, just not introspectable.
    """
    try:
        tree = ast.parse(source, filename=filename)
    except (SyntaxError, ValueError):
        return []

    symbols: list[Symbol] = []

    def add(node, kind: str, qualname: str) -> None:
        symbols.append(
            Symbol(
                name=node.name,
                kind=kind,
                file=filename,
                line=node.lineno,
                source=_segment(source, node),
                doc=ast.get_docstring(node) or "",
                qualname=qualname,
            )
        )

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            add(node, "function", node.name)
        elif isinstance(node, ast.ClassDef):
            add(node, "class", node.name)
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    add(child, "method", f"{node.name}.{child.name}")
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    symbols.append(
                        Symbol(
                            name=target.id,
                            kind="variable",
                            file=filename,
                            line=node.lineno,
                            source=_segment(source, node),
                            qualname=target.id,
                        )
                    )
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            symbols.append(
                Symbol(
                    name=node.target.id,
                    kind="variable",
                    file=filename,
                    line=node.lineno,
                    source=_segment(source, node),
                    qualname=node.target.id,
                )
            )
    return symbols


def find_symbol(
    root: Path | str,
    name: str,
    limit: int = MAX_SYMBOL_MATCHES,
) -> list[Symbol]:
    """Locate a named function/class/method/variable across the project.

    Retrieval by embedding gets you a chunk that may or may not contain the
    whole definition. This gets you the definition.
    """
    from .indexer import iter_source_files, read_text

    root = Path(root)
    wanted = name.strip().lstrip("@").rstrip("()").lower()
    if not wanted:
        return []

    exact: list[Symbol] = []
    partial: list[Symbol] = []
    for path in iter_source_files(root, extensions={".py"}):
        source = read_text(path)
        if not source:
            continue
        rel = str(path.relative_to(root)).replace("\\", "/")
        for symbol in iter_symbols(source, filename=rel):
            if symbol.qualname.lower() == wanted or symbol.name.lower() == wanted:
                exact.append(symbol)
            elif wanted in symbol.qualname.lower():
                partial.append(symbol)
        if len(exact) >= limit:
            break
    return (exact or partial)[:limit]


# --------------------------------------------------------------- ruff
@dataclass
class StyleReport:
    """What ruff would change about a file."""

    file: str
    diff: str = ""
    issues: list[str] = field(default_factory=list)
    available: bool = True
    error: str = ""

    @property
    def is_clean(self) -> bool:
        return self.available and not self.diff and not self.issues

    def as_text(self) -> str:
        if not self.available:
            return f"(ruff is not installed: {self.error})"
        if self.is_clean:
            return f"{self.file} is already formatted and lint-clean."
        parts = []
        if self.issues:
            parts.append("Lint findings:\n" + "\n".join(self.issues))
        if self.diff:
            parts.append("Formatting diff ruff would apply:\n" + self.diff)
        return "\n\n".join(parts)


def _run_ruff(args: list[str]) -> tuple[int, str, str]:
    proc = subprocess.run(
        [sys.executable, "-m", "ruff", *args],
        capture_output=True,
        text=True,
        timeout=RUFF_TIMEOUT,
    )
    return proc.returncode, proc.stdout, proc.stderr


def style_report(
    path: Path | str,
    display_name: str | None = None,
    max_issues: int = 20,
) -> StyleReport:
    """Run ruff's linter and formatter over one file, without changing it."""
    path = Path(path)
    name = display_name or path.name
    report = StyleReport(file=name)

    try:
        _, diff_out, _ = _run_ruff(["format", "--diff", str(path)])
        report.diff = diff_out.strip()

        _, check_out, _ = _run_ruff(
            ["check", "--select", RUFF_SELECT, "--output-format", "json", str(path)]
        )
        for item in json.loads(check_out or "[]"):
            location = item.get("location") or {}
            report.issues.append(
                f"line {location.get('row', '?')}: "
                f"{item.get('code', '?')} {item.get('message', '')}"
            )
        report.issues = report.issues[:max_issues]
    except FileNotFoundError as exc:  # pragma: no cover - python without ruff
        report.available = False
        report.error = str(exc)
    except (subprocess.SubprocessError, json.JSONDecodeError, OSError) as exc:
        report.available = False
        report.error = str(exc)
    return report


def resolve_in_project(root: Path | str, source: str) -> Path | None:
    """Turn an indexed `source` path back into a real file on disk."""
    if not root or not source:
        return None
    candidate = Path(root) / source
    return candidate if candidate.is_file() else None
