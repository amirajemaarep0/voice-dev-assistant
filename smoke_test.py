"""End-to-end smoke check against the real stack.

Indexes this project into a real Chroma store, retrieves for a sample
question, then (if Ollama is up) generates one grounded answer.

    python smoke_test.py
"""
from __future__ import annotations

import time
from pathlib import Path

from assistant import config, llm
from assistant.pipeline import Assistant
from assistant.store import ProjectStore, build_index

ROOT = Path(__file__).parent
QUESTION = "How does the project decide which files to index?"


def main() -> int:
    print("=" * 62)
    print("1. INDEXING")
    store = ProjectStore(persist_dir=ROOT / ".chroma")
    t0 = time.perf_counter()
    stats = build_index(ROOT, store)
    print(f"   {stats.as_dict()}")
    print(f"   took {time.perf_counter() - t0:.1f}s, store holds {store.count()}")

    print("\n2. RETRIEVAL")
    print(f"   Q: {QUESTION}")
    t0 = time.perf_counter()
    chunks = store.search(QUESTION, top_k=3)
    print(f"   took {time.perf_counter() - t0:.2f}s")
    for c in chunks:
        print(f"   - {c.citation}  distance={c.distance:.3f}")
    if not chunks:
        print("   !! nothing retrieved")
        return 1

    print("\n3. GENERATION")
    try:
        models = llm.list_local_models()
    except llm.OllamaError as exc:
        print(f"   SKIPPED: {exc}")
        return 0

    print(f"   local models: {models}")
    model = config.DEFAULT_MODEL if config.DEFAULT_MODEL in models else models[0]
    print(f"   using: {model}\n")

    assistant = Assistant(store, config.Settings(model=model, top_k=3))
    t0 = time.perf_counter()
    _, stream = assistant.stream(QUESTION)
    first_token = None
    parts = []
    for token in stream:
        if first_token is None:
            first_token = time.perf_counter() - t0
        parts.append(token)
        print(token, end="", flush=True)
    total = time.perf_counter() - t0

    print(f"\n\n   time to first token: {first_token:.1f}s")
    print(f"   total: {total:.1f}s for {len(''.join(parts))} chars")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
