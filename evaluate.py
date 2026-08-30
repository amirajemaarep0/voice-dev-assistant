"""Measure how accurate the assistant's answers actually are.

    python evaluate.py                 # default model
    python evaluate.py llama3:latest   # compare another model
    python evaluate.py --facts-only    # score the tool layer alone, no LLM

Each case states what a correct answer must contain and what it must not.
"Must not" is the important half: the failures worth catching are not
missing detail, they are confident statements that contradict the tools -
"the file parses cleanly" printed directly under a syntax error.

The tool layer is scored separately from the model, because they fail for
different reasons and only one of them is fixable by prompting.
"""
from __future__ import annotations

import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from assistant import config, intents
from assistant.pipeline import Assistant
from assistant.store import ProjectStore

ROOT = Path(__file__).parent


@dataclass
class Case:
    question: str
    intent: str
    facts_must_contain: list[str] = field(default_factory=list)
    must_contain: list[str] = field(default_factory=list)
    must_not_contain: list[str] = field(default_factory=list)
    # A regex with one group, matched against the tool findings. Whatever it
    # captures must then appear verbatim in the answer. Used for numbers the
    # model must copy rather than estimate - hard-coding the expected value
    # here would just go stale the next time a test is added.
    echo_from_facts: str = ""


CASES = [
    Case(
        question="what is wrong with samples/broken_function.py?",
        intent=intents.SYNTAX,
        facts_must_contain=["broken_function.py", "never closed", "line 5"],
        must_contain=["broken_function.py", r"\)"],
        must_not_contain=[r"parses cleanly", r"no syntax error", r"is fine"],
    ),
    Case(
        question="is there a syntax error in assistant/store.py?",
        intent=intents.SYNTAX,
        facts_must_contain=["no syntax errors"],
        must_contain=["store.py"],
        must_not_contain=[r"never closed", r"invalid syntax"],
    ),
    Case(
        question="check the whole project for syntax mistakes",
        intent=intents.HEALTH,
        facts_must_contain=["samples/broken_function.py"],
        must_contain=["broken_function.py"],
        must_not_contain=[r"no syntax error", r"every .{0,20}file .{0,20}parses"],
    ),
    Case(
        question="run the tests",
        intent=intents.TESTRUN,
        facts_must_contain=["TEST RUN", "GREEN"],
        must_contain=["pass"],
        must_not_contain=[r"\bfailed\b(?!,? 0)", r"\bred\b"],
    ),
    Case(
        question="how should samples/messy_style.py be formatted?",
        intent=intents.STYLE,
        facts_must_contain=["F401", "def average(numbers):"],
        must_contain=["import"],
        must_not_contain=[r"already formatted", r"no issues"],
    ),
    Case(
        question="explain the function normalize_project_path",
        intent=intents.EXPLAIN,
        facts_must_contain=["def normalize_project_path", "expanduser"],
        must_contain=["path"],
        must_not_contain=[r"not in the (excerpts|index)", r"cannot find"],
    ),
    Case(
        question="explain the function frobnicate_widgets",
        intent=intents.EXPLAIN,
        facts_must_contain=["no function, class or variable named"],
        must_contain=["frobnicate_widgets"],
        must_not_contain=[r"```python"],   # must not invent an implementation
    ),
    Case(
        question="write a unit test for average",
        intent=intents.TEST,
        facts_must_contain=["def  average( numbers )"],
        must_contain=["def test", "average"],
        must_not_contain=[r"I cannot", r"not in the excerpts"],
    ),
    Case(
        question="can you check /tests/test_function.py file?",
        intent=intents.GENERAL,
        must_contain=["test_function.py"],
        must_not_contain=[r"```python\s*\ndef ", r"the file contains"],
    ),
    Case(
        question="do my tests pass?",
        intent=intents.TESTRUN,
        facts_must_contain=["TEST RUN"],
        # The count must be copied from the tools, not estimated - and not
        # hard-coded here, or it goes stale the next time a test is added.
        echo_from_facts=r"TEST RUN: (\d+) passed",
        must_not_contain=[r"\bfailed\b(?!,? 0)"],
    ),
    Case(
        question="are there any errors in the whole codebase?",
        intent=intents.HEALTH,
        facts_must_contain=["broken_function.py", "F401"],
        must_contain=["broken_function.py"],
        # "the project is **not clean**" is a correct answer, so the check
        # has to be for the positive claim, not the bare word.
        must_not_contain=[r"project is clean", r"no (errors|issues|findings)"],
    ),
    Case(
        question="what does the Retrieved class do?",
        intent=intents.EXPLAIN,
        facts_must_contain=["class Retrieved", "citation"],
        must_contain=["chunk|retriev"],
        must_not_contain=[r"not (in|found)", r"cannot"],
    ),
    Case(
        question="explain iter_source_files",
        intent=intents.EXPLAIN,
        facts_must_contain=["def iter_source_files", "os.walk"],
        must_contain=["director|folder|file"],
        # It must not claim the traversal does something it does not.
        must_not_contain=[r"recursively deletes", r"modifies the files"],
    ),
]


def _missing(text: str, patterns: list[str], regex: bool) -> list[str]:
    out = []
    for pattern in patterns:
        found = (
            re.search(pattern, text, re.I) if regex
            else pattern.lower() in text.lower()
        )
        if not found:
            out.append(pattern)
    return out


def _present(text: str, patterns: list[str]) -> list[str]:
    return [p for p in patterns if re.search(p, text, re.I)]


def main() -> int:
    argv = [a for a in sys.argv[1:]]
    facts_only = "--facts-only" in argv
    with_excerpts = "--with-excerpts" in argv
    argv = [a for a in argv if not a.startswith("--")]
    model = argv[0] if argv else config.DEFAULT_MODEL

    if with_excerpts:
        # Restore the pre-fix behaviour: attach retrieved excerpts even to
        # tool-driven questions. Kept as a flag so the improvement from
        # dropping them is reproducible rather than asserted.
        import assistant.pipeline as pipeline_module

        pipeline_module.FACTS_ONLY = frozenset()

    store = ProjectStore()
    if store.count() == 0:
        print("Index this project first (streamlit run app.py, or smoke_test.py).")
        return 1

    assistant = Assistant(store, config.Settings(model=model, top_k=4))
    print("=" * 72)
    print(f"Evaluating: {'tool layer only' if facts_only else model}")
    if with_excerpts:
        print("Mode: retrieved excerpts attached to tool-driven questions")
    print(f"Indexed from: {store.indexed_root}")
    print("=" * 72)

    intent_ok = facts_ok = answer_ok = 0
    started = time.perf_counter()

    for case in CASES:
        context = assistant.build_context(case.question)

        got_intent = context.intent.kind == case.intent
        intent_ok += got_intent

        lost = _missing(context.facts, case.facts_must_contain, regex=False)
        got_facts = not lost
        facts_ok += got_facts

        print(f"\nQ: {case.question}")
        print(f"   intent  {'OK ' if got_intent else 'BAD'} "
              f"(want {case.intent}, got {context.intent.kind})")
        print(f"   facts   {'OK ' if got_facts else 'BAD'}"
              + (f" missing {lost}" if lost else ""))

        if facts_only:
            continue

        answer = "".join(assistant.stream_context(case.question)[1])
        absent = _missing(answer, case.must_contain, regex=True)
        forbidden = _present(answer, case.must_not_contain)

        # A number the model was told to copy must survive into the answer.
        if case.echo_from_facts:
            wanted = re.search(case.echo_from_facts, context.facts)
            if not wanted:
                absent.append(f"(facts lacked {case.echo_from_facts})")
            elif wanted.group(1) not in answer:
                absent.append(f"(did not echo {wanted.group(1)} from the tools)")

        good = not absent and not forbidden
        answer_ok += good
        print(f"   answer  {'OK ' if good else 'BAD'}"
              + (f" missing {absent}" if absent else "")
              + (f" CONTRADICTS {forbidden}" if forbidden else ""))
        if not good:
            print("   ---")
            for line in answer.strip().splitlines()[:6]:
                print(f"   | {line}")

    total = len(CASES)
    elapsed = time.perf_counter() - started
    print("\n" + "=" * 72)
    print(f"intent classification : {intent_ok}/{total}")
    print(f"tool findings         : {facts_ok}/{total}")
    if not facts_only:
        print(f"model answers         : {answer_ok}/{total}")
    print(f"took {elapsed:.0f}s")
    print("=" * 72)
    return 0 if facts_ok == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
