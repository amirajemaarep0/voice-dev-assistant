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
- **Filenames are resolved before similarity search** (`extract_file_references`).
  Pure vector retrieval is close to useless for questions that name a file:
  *"what is in assistant/stt.py?"* shares almost no vocabulary with the
  contents of `stt.py`, and measurably returned **zero** chunks from that
  file — the top-4 were `README.md`, `conftest.py`, `test_indexer.py` and
  `test_stt.py`. The question is now scanned for filenames, any that exist
  in the index are pulled in whole by metadata filter, and similarity hits
  fill the remaining budget. This matters because two of the three example
  questions in the project brief — *"Explain what this function does"*,
  *"Generate a unit test for this file"* — are file-scoped by nature.
- **A named file that is not indexed is reported as such.** Otherwise the
  model answers "that is not in the excerpts", which is true, useless, and
  indistinguishable from the file not existing.
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
   Quotes around the path (what Explorer's *Copy as path* gives you) and
   stray whitespace are stripped for you.
2. Record a question with the mic, or type it.
3. The answer streams back with the source excerpts it was grounded in.

The sidebar always shows **which folder the current index was built from**.
If that line does not match the folder in the box, the answers are still
coming from the previous project — press **Index project** to rebuild.

The index is a snapshot, not a live view: a file created after the last
index is invisible to the assistant until you re-index. The sidebar watches
the folder and says *"Files have changed on disk since this index was
built"* when that happens, so a missing file is never a silent failure.

`samples/broken_function.py` is a deliberately broken file kept as a demo
target — ask the assistant what is wrong with it. It lives outside `tests/`
on purpose: pytest collects `tests/test_*.py` by importing them, so a syntax
error in that directory aborts collection for the entire suite.

### Streamlit state, and why the folder box is written the way it is

Two bugs in the first version of the sidebar both came from the same place,
and they are worth knowing about before writing any more Streamlit:

- **The path could not be changed.** The box was written as
  `st.text_input(..., value=st.session_state["project_dir"])` with no `key`.
  Streamlit derives a widget's identity from its arguments, so every time
  that `value` changed the widget was rebuilt from scratch and whatever the
  user had typed was thrown away — the box snapped back to the old path.
  The fix is to give the widget a `key` and let it own its own state.
- **Indexing appeared to do nothing.** The button was
  `disabled=not valid_dir`, and `valid_dir` was computed from the *previous*
  run's value. Typing a path and clicking took two presses: the first click
  only committed the text and enabled the button. The button is now never
  disabled and the folder is validated when it is clicked.

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
