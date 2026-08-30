"""Phase 1c - talk to the local Ollama server and assemble prompts.

Uses the plain HTTP API rather than the `ollama` python package: one less
dependency, and the request/response shape is stable enough to mock in
tests without pinning a client version.
"""
from __future__ import annotations

import json
from typing import Iterable, Iterator

import requests

from . import config
from .store import Retrieved

SYSTEM_PROMPT = """You are a senior developer assistant running entirely on the \
user's machine. You answer questions about THEIR project using only the code \
excerpts provided below.

Rules:
- Ground every claim in the excerpts. If they do not contain the answer, say \
so plainly and name what file you would need to see.
- Cite the file you are drawing on, like [src/app.py].
- Quote the user's code exactly as it appears, including anything that looks \
like a mistake. Never silently repair a typo, a missing bracket or a broken \
line while quoting it: reporting the defect IS the answer they asked for.
- Be concise. The answer is read aloud or skimmed while the user codes.
- When asked to write code (a test, a refactor), match the style and imports \
already visible in the excerpts."""


class OllamaError(RuntimeError):
    """Raised when the local Ollama server is unreachable or errors out."""


def _is_local(model_name: str, entry: dict | None = None) -> bool:
    """Cloud-hosted Ollama models would send code off-device. Exclude them.

    Two independent checks, because either alone can be fooled:
    the ``*-cloud`` naming convention, and the ``remote_host`` /
    ``remote_model`` fields Ollama sets on proxied models.
    """
    if entry:
        if entry.get("remote_host") or entry.get("remote_model"):
            return False
    return "-cloud" not in model_name and not model_name.endswith("cloud")


def list_local_models(
    host: str = config.OLLAMA_HOST, timeout: float = 5.0
) -> list[str]:
    """Return installed, genuinely on-device model names."""
    try:
        resp = requests.get(f"{host}/api/tags", timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise OllamaError(
            f"Cannot reach Ollama at {host}. Is `ollama serve` running?"
        ) from exc
    entries = resp.json().get("models", [])
    return sorted(
        e.get("name", "")
        for e in entries
        if e.get("name") and _is_local(e["name"], e)
    )


_CAPS_CACHE: dict[str, set[str]] = {}


def model_capabilities(model: str, host: str = config.OLLAMA_HOST) -> set[str]:
    """Capabilities Ollama reports for a model (e.g. {'completion','thinking'}).

    Cached: this is a local call, but it happens on every question.
    """
    if model in _CAPS_CACHE:
        return _CAPS_CACHE[model]
    try:
        resp = requests.get(f"{host}/api/tags", timeout=5.0)
        resp.raise_for_status()
        for entry in resp.json().get("models", []):
            _CAPS_CACHE[entry.get("name", "")] = set(entry.get("capabilities") or [])
    except requests.RequestException:
        return set()
    return _CAPS_CACHE.get(model, set())


def is_available(host: str = config.OLLAMA_HOST) -> bool:
    try:
        list_local_models(host)
        return True
    except OllamaError:
        return False


def format_context(chunks: Iterable[Retrieved]) -> str:
    """Render retrieved chunks as a labelled, citable block."""
    parts = []
    for c in chunks:
        parts.append(f"--- FILE: {c.source} (chunk {c.position}) ---\n{c.text}")
    return "\n\n".join(parts)


def build_prompt(
    question: str,
    chunks: Iterable[Retrieved],
    missing_files: Iterable[str] | None = None,
    facts: str = "",
    task: str = "general",
) -> str:
    """Assemble the final user prompt. Pure function - unit tested."""
    chunks = list(chunks)
    missing = list(missing_files or [])

    # Deterministic findings go first and are labelled as measured, not
    # inferred: the model must not "correct" them from the excerpts.
    head = ""
    if facts:
        head = (
            "The following was produced by running real tools on the "
            "user's files. It is factual and takes precedence over "
            f"anything you infer from the excerpts.\n\n{facts}\n\n"
        )
    instruction = TASK_INSTRUCTIONS.get(task, "")
    tail = f"\n\n{instruction}" if instruction else ""

    # A named file that is not in the index has a specific, actionable cause:
    # it was added after the last index, or it lives outside the indexed
    # folder. Saying that beats "the excerpts do not mention it".
    note = ""
    if missing:
        names = ", ".join(missing)
        note = (
            f"\n\nNOTE: the question names {names}, which is NOT in the "
            "indexed project. Say so directly and tell the user to re-index "
            "if they have just created or moved it. Do not guess its contents."
        )

    if not chunks:
        if facts:
            return f"{head}Question: {question}{note}{tail}"
        return (
            f"{question}\n\n"
            "(No relevant code was found in the indexed project. Say so, and "
            f"ask for the file or symbol name.){note}{tail}"
        )
    return (
        f"{head}"
        "Here are the most relevant excerpts from the user's project:\n\n"
        f"{format_context(chunks)}\n\n"
        f"Question: {question}{note}{tail}"
    )


# What to do with the deterministic findings, per task. The findings are
# already correct; these say how to present them.
TASK_INSTRUCTIONS = {
    "syntax": (
        "The syntax check above came from Python's own parser and is "
        "authoritative. If it lists errors, report each one at the exact "
        "line given, explain in one sentence why Python rejects it, and "
        "show the corrected line - do NOT also say the file is fine. Only "
        "if it reports no errors may you say the file parses cleanly."
    ),
    "health": (
        "The report above is a full scan of the project: Python's parser "
        "for syntax, ruff for lint. Give a short verdict first (is the "
        "project clean or not), then list the findings grouped by file, "
        "using the exact paths and line numbers given. Do not add findings "
        "that are not listed, and do not omit any that are."
    ),
    "testrun": (
        "The test suite above was actually executed just now. Your first "
        "sentence MUST state whether it is green or red AND quote the exact "
        "counts from the report (how many passed, failed, errored, "
        "skipped) - copy those digits, never round or estimate them. If it "
        "is red, then name each failing test and explain the cause from the "
        "failure output. Never claim a test passed or failed unless the "
        "output says so."
    ),
    "style": (
        "The report above is exactly what `ruff` would change - it is not a "
        "suggestion of yours. Summarise the findings in a short list, "
        "grouped by what kind of problem they are, and show the corrected "
        "code for the most important one. Do not invent findings."
    ),
    "explain": (
        "Explain what the definition above does: its purpose in one "
        "sentence, then its parameters, its return value, and anything "
        "non-obvious about how it works. Refer only to code shown."
    ),
    "test": (
        "Write a pytest test for the definition above. Match the house "
        "style shown: same import layout, same class/function naming, same "
        "assertion style. Cover the normal case and at least one edge case. "
        "Output the test code in one ```python block, then one short line "
        "saying what each test covers. Do not test behaviour the code does "
        "not have."
    ),
}


def build_prompt_for(context) -> str:
    """Assemble the prompt for a classified question.

    Takes a `pipeline.Context`. Kept here so prompt assembly - the part
    worth inspecting when an answer goes wrong - lives in one file.
    """
    return build_prompt(
        context.question,
        context.chunks,
        missing_files=context.missing_files,
        facts=context.facts,
        task=context.intent.kind,
    )


def stream_answer(
    prompt: str,
    model: str = config.DEFAULT_MODEL,
    temperature: float = 0.1,
    host: str = config.OLLAMA_HOST,
    timeout: float = 300.0,
    think: bool = False,
) -> Iterator[str]:
    """Yield answer tokens as they arrive from Ollama.

    Reasoning models (qwen3 and friends) otherwise emit a long hidden
    "thinking" pass before the first visible token - about 100 s on this
    hardware. We turn it off: for grounded Q&A over retrieved code it buys
    nothing and destroys the interaction.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "system": SYSTEM_PROMPT,
        "stream": True,
        "keep_alive": config.KEEP_ALIVE,
        "options": {
            "temperature": temperature,
            "num_ctx": config.NUM_CTX,
        },
    }
    if "thinking" in model_capabilities(model, host):
        payload["think"] = think
    try:
        with requests.post(
            f"{host}/api/generate", json=payload, stream=True, timeout=timeout
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if data.get("error"):
                    raise OllamaError(data["error"])
                token = data.get("response")
                if token:
                    yield token
                if data.get("done"):
                    break
    except requests.RequestException as exc:
        raise OllamaError(f"Ollama request failed: {exc}") from exc


def answer(prompt: str, **kwargs) -> str:
    """Non-streaming convenience wrapper."""
    return "".join(stream_answer(prompt, **kwargs))
