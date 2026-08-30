"""What is the developer actually asking for?

The assistant does four things a generic RAG chat cannot: check syntax,
report formatting, explain a named symbol, and write a unit test. Each one
needs different context assembled before the model is called - the exact
`ast` error, the ruff diff, the full definition - so the question has to be
classified first.

Kept as pure string matching: it is inspectable, it is instant, and it does
not spend a second model call to decide what the first one should do.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from . import config

SYNTAX = "syntax"
STYLE = "style"
EXPLAIN = "explain"
TEST = "test"
GENERAL = "general"

# Ordered: the first pattern that matches wins, so the more specific
# intents are listed before the ones whose wording overlaps them.
_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (TEST, re.compile(
        r"\b(unit\s*test|write\s+(a\s+)?test|generate\s+(a\s+)?test|"
        r"create\s+(a\s+)?test|test\s+case|pytest)\b", re.I)),
    (SYNTAX, re.compile(
        r"\b(syntax|compile[sd]?|parse[sd]?|valid\s+python|"
        r"what(?:'s|s|\s+is|\s+are)?\s+wrong|any\s+errors?|"
        r"error[s]?\s+in|broken|bug|does\s+it\s+run)\b", re.I)),
    (STYLE, re.compile(
        r"\b(re)?format(s|ted|ting)?\b|\bstyl(e|ing|ed)\b|\bpep\s*-?\s*8\b|"
        r"\blint(er|ing)?\b|\bclean\s*up\b|\btidy\b|\bindentation\b|"
        r"\bconventions?\b", re.I)),
    (EXPLAIN, re.compile(
        r"\b(explain|what\s+does|what\s+is|how\s+does|describe|"
        r"walk\s+me\s+through|purpose\s+of|meaning\s+of)\b", re.I)),
]

# A quoted or back-ticked name, or a bare identifier after a cue word.
_QUOTED = re.compile(r"[`'\"]([A-Za-z_][\w.]*)[`'\"]")
_CUED = re.compile(
    r"\b(?:function|method|class|variable|symbol|def|test\s+for)\s+"
    r"([A-Za-z_][\w.]*)",
    re.I,
)
_CALL = re.compile(r"\b([A-Za-z_][\w.]*)\s*\(\s*\)")

# Words that look like identifiers but never are, in these questions.
_STOPWORDS = {
    "the", "this", "that", "it", "a", "an", "of", "for", "in", "to", "is",
    "does", "do", "what", "how", "explain", "file", "code", "project",
    "function", "method", "class", "variable", "test", "tests", "unit",
    "py", "python", "me", "my", "please", "you", "can",
}


@dataclass
class Intent:
    """A classified question."""

    kind: str = GENERAL
    symbol: str = ""

    @property
    def is_general(self) -> bool:
        return self.kind == GENERAL

    @property
    def needs_symbol(self) -> bool:
        return self.kind in (EXPLAIN, TEST)


def _is_identifier_like(candidate: str) -> bool:
    """True for things worth looking up as a symbol.

    Filenames are excluded: they are resolved by the file-reference path in
    `indexer.extract_file_references`, and treating "app.py" as a symbol
    named `py` would send the whole answer off course.
    """
    if not candidate or candidate.lower() in _STOPWORDS:
        return False
    suffix = candidate[candidate.rfind("."):].lower() if "." in candidate else ""
    return suffix not in config.SOURCE_EXTENSIONS


def extract_symbol(question: str) -> str:
    """Pull the identifier a question is about, if it names one.

    Prefers explicit signals - back-ticks, quotes, "the function X", "X()" -
    over guessing, because a wrong guess is worse than no guess at all.
    """
    if not question:
        return ""
    for pattern in (_QUOTED, _CUED, _CALL):
        for match in pattern.finditer(question):
            candidate = match.group(1).strip(".")
            if _is_identifier_like(candidate):
                return candidate
    # Dotted names, then snake_case, then CamelCase. The lookahead keeps
    # "broken_function" inside "broken_function.py" from being read as a
    # symbol - that is a filename, and a different code path handles it.
    for pattern in (
        r"\b[A-Za-z_]\w*\.[A-Za-z_]\w*(?!\.?\w)",
        r"\b\w+_\w+\b(?!\.[A-Za-z])",
        r"\b[A-Z][a-z]+[A-Z]\w*\b(?!\.[A-Za-z])",
    ):
        for word in re.findall(pattern, question):
            if _is_identifier_like(word):
                return word
    return ""


def detect_intent(question: str) -> Intent:
    """Classify a question into one of the assistant's coding tasks."""
    if not question or not question.strip():
        return Intent()
    for kind, pattern in _PATTERNS:
        if pattern.search(question):
            symbol = extract_symbol(question) if kind in (EXPLAIN, TEST) else ""
            return Intent(kind=kind, symbol=symbol)
    return Intent(kind=GENERAL, symbol=extract_symbol(question))
