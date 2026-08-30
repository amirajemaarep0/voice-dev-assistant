# Local Voice AI Dev Assistant with Project Context

End-of-year project — Amira Jemaa, Pristini School of AI, 2025–2026
Supervisor: Dr. Soumaya Trabelsi

A desktop assistant that answers spoken questions about your own codebase.
Speech recognition, embeddings, retrieval and generation all run **on this
machine** — no source code is ever sent to a third-party server.

---

## Architecture

```
voice ──► Whisper (STT) ──► text ──┐
                                    ├─► retrieve top-k chunks ──► prompt ──► Ollama ──► answer
project files ──► chunk ──► embed ──┘        (Chroma)                         (LLM)
```

| Module | File | Responsibility |
|---|---|---|
| Config | `assistant/config.py` | Models, paths, chunking and filter constants |
| Indexing | `assistant/indexer.py` | Walk the tree, read files, language-aware chunking |
| Vector store | `assistant/store.py` | Chroma persistence, embedding, similarity search |
| LLM | `assistant/llm.py` | Ollama HTTP client, system prompt, prompt assembly |
| STT | `assistant/stt.py` | Whisper transcription (faster-whisper / CTranslate2) |
| Orchestration | `assistant/pipeline.py` | Question → retrieve → prompt → streamed answer |
| Interface | `app.py` | Streamlit UI: folder picker, mic, chat, sources |

## Design decisions worth defending

- **faster-whisper over openai-whisper.** Identical Whisper weights, run on
  CTranslate2 instead of PyTorch: ~3–4× faster on CPU and far less RAM.
- **Chroma's ONNX `all-MiniLM-L6-v2` for embeddings.** Same model the
  literature uses, but no PyTorch dependency — which keeps ~3 GB of RAM free
  for the LLM on an 8 GB machine.
- **Language-aware chunking** (`RecursiveCharacterTextSplitter.from_language`).
  Naive character splitting cuts functions in half; splitting on syntactic
  boundaries measurably improves retrieval relevance.
- **Cloud models are filtered out** (`llm._is_local`). An Ollama `*-cloud`
  model would send code off-device and invalidate the project's core claim.
- **Ollama over HTTP** rather than the Python client: one less dependency and
  a response shape that is trivial to mock in tests.

## Setup

```powershell
cd voice-dev-assistant
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# start the model server (leave running in its own terminal)
ollama serve
ollama pull qwen3:1.7b
```

## Run

```powershell
streamlit run app.py
```

1. Paste a project folder path in the sidebar and press **Index project**.
2. Record a question with the mic, or type it.
3. The answer streams back with the source excerpts it was grounded in.

## Tests

```powershell
pytest
```

The suite runs offline: Ollama, Whisper and the vector store are all faked,
so the tests need no model weights and no network.

Two extra checks that exercise the real stack:

```powershell
python smoke_test.py    # real indexing + retrieval + one generated answer
python verify_app.py    # executes app.py outside Streamlit to catch UI errors
```

### Measured on the dev machine

| Step | Cold | Warm |
|---|---|---|
| Index 18 files → 100 chunks | 150 s (incl. 79 MB embedding-model download) | **5.0 s** |
| Retrieval (top-3) | — | **0.05 s** |
| Answer, `qwen3:1.7b`, time to first token | 104 s (reasoning pass enabled) | **12.2 s** |

The generation figure is the single most important tuning result in the
project: `qwen3` is a reasoning model and spends a long hidden "thinking"
pass before its first visible token. Sending `"think": false` to Ollama
removes it, cutting time-to-first-token by roughly 8×. Reasoning buys
nothing here because the answer is already grounded in retrieved code.

## Hardware notes

Developed on: Ryzen 5 6600H, 8 GB RAM, RTX 3050 Laptop (4 GB VRAM), Windows 11.

`qwen3:1.7b` is the default because it fits in 4 GB of VRAM. `llama3:latest`
(8B) runs but spills into system RAM on an 8 GB machine and is noticeably
slower — a useful comparison for the evaluation chapter.
