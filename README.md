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
| Code tools | `assistant/tools.py` | Syntax check (`ast`), symbol lookup, ruff report |
| Intent | `assistant/intents.py` | Classify the question into a coding task |
| Folder picker | `assistant/picker.py` | Native dialog + in-app directory browser |
| Orchestration | `assistant/pipeline.py` | Question → classify → tools + retrieve → answer |
| Interface | `app.py` | Streamlit UI: folder picker, mic, chat, sources |

## What it does

Beyond answering questions about the code, it performs four developer tasks
locally, so none of them require a third-party AI service:

| Ask | What actually runs | Why it is reliable |
|---|---|---|
| *what is wrong with `samples/broken_function.py`?* | `ast.parse` | Python's own parser — exact line and column |
| *are there errors in the whole codebase?* | `ast.parse` on every file + `ruff check` | Full scan, syntax and lint in one report |
| *do my tests pass?* | `pytest` in a subprocess | The suite is really executed; exit code is the verdict |
| *how should `samples/messy_style.py` be formatted?* | `ruff format --diff`, `ruff check` | The real diff, not a guess |
| *explain the function `normalize_project_path`* | `ast` symbol lookup | The whole definition, not a nearby chunk |
| *write a unit test for `average`* | symbol lookup + an existing test file | Matches your project's test conventions |

Running the tests **executes your project's own code** — that is the only
way to surface a genuine runtime error, such as a `NameError` in a branch
no static check reaches. It runs only when you ask, with a timeout.

One limitation worth knowing: the suite is run with *this* project's Python
interpreter. If the project you indexed has its own virtual environment and
dependencies, its imports will fail and you will see that in the failure
output rather than a real test result. Syntax, lint and explanation work on
any project; running the tests assumes a shared environment.

The key design decision is that **the model is never asked to find these
things** — only to explain findings that are already correct. This matters
because the model is small: asked directly what was wrong with
`print(" HAMDI"`, `qwen3:1.7b` quoted the line back with the bracket closed
and declared the file fine. Handed the parser's output, it reports the error
at the right line and shows the fix. The UI labels tool output *"exact, not
generated"* so the two are never confused.

`samples/` holds two deliberately faulty files kept as demo targets — one
with a syntax error, one badly formatted.

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

1. Choose the project folder, three ways — whichever suits:
   - **📂 Browse…** opens the normal Windows folder dialog. The Streamlit
     server *is* your machine, so a native `tkinter` dialog is legitimate
     here. It opens on the desktop, in front of the browser.
   - **🗂 Folder list** browses directories inside the app instead. Always
     works, including where no desktop dialog can open.
   - **Paste the path** into the box. Quotes around it (what Explorer's
     *Copy as path* gives you) and stray whitespace are stripped for you.
2. Press **Index project**.
3. Record a question with the mic, or type it.
4. The answer streams back, with any tool findings shown separately from the
   model's prose and the source excerpts it was grounded in.

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

### Measuring answer accuracy

`evaluate.py` scores the assistant against cases that state what a correct
answer must contain **and what it must not**. The second half is the one
that matters: the failures worth catching are not missing detail, they are
confident statements that contradict the tools — "the file parses cleanly"
printed directly underneath a syntax error.

```powershell
python evaluate.py                    # score the default model
python evaluate.py llama3:latest      # compare another model
python evaluate.py --facts-only       # score the tool layer alone, no LLM
python evaluate.py --with-excerpts    # restore the pre-fix prompt, to compare
```

It scores three layers separately, because they fail for different reasons:
intent classification, tool findings, and the model's prose. Only the last
is affected by prompting.

**The measured result that drove a design change.** Answers were noticeably
unreliable, and separating the layers showed why: the tool layer was already
perfect, so the fault was entirely in the prompt. Tool-driven questions were
still being sent the top-k retrieved excerpts, and that unrelated code was
what the small model latched onto — in one run it justified a claim by
citing a test that had nothing to do with the question.

| Prompt for tool-driven questions | Correct answers (`qwen3:1.7b`, 13 cases) |
|---|---|
| Tool findings **+ retrieved excerpts** (original) | 9 / 13 |
| Tool findings **only** | 12 / 13 |
| Tool findings only, **branch-specific instructions** | **13 / 13** |

Reproduce the first two with `python evaluate.py --with-excerpts` versus
plain `python evaluate.py`. Retrieval still runs for questions that need
it — explaining code and writing tests keep their excerpts; only the tasks
whose answer is fully determined by tool output have them stripped.

The last row is a smaller but instructive fix. One instruction covering
both outcomes had to describe the clean one — *"only if it reports no
errors may you say the file parses cleanly"* — and the model copied that
phrase into an answer that had just correctly reported a syntax error.
There are now separate instructions for the clean and unclean branches,
chosen from what the tools actually found, so the model never sees wording
for an outcome that did not happen. **Not showing a small model the words
is more reliable than forbidding them.**

The project passes its own linter — the one it offers to run on your code:

```powershell
python -m ruff check --select E,W,F,I assistant/ app.py tests/
```

`samples/` is deliberately excluded from that claim; those files are broken
on purpose.

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
