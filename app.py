"""Phase 3 - Streamlit interface for the Local Voice AI Dev Assistant.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import hashlib

import streamlit as st

from assistant import config, llm, picker, stt
from assistant.indexer import normalize_project_path, project_fingerprint
from assistant.intents import TASK_LABELS
from assistant.pipeline import Assistant
from assistant.store import ProjectStore, build_index

st.set_page_config(
    page_title="Local Voice AI Dev Assistant",
    page_icon="🎙️",
    layout="wide",
    # The folder picker lives in the sidebar; collapsed by default it is easy
    # to miss that the app needs a project pointed at it at all.
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner=False)
def get_store(persist_dir: str) -> ProjectStore:
    """One Chroma client per persist directory, reused across reruns."""
    return ProjectStore(persist_dir=persist_dir)


@st.cache_data(ttl=5, show_spinner=False)
def current_fingerprint(root: str) -> str:
    """Signature of the files on disk right now.

    Cached briefly because Streamlit reruns the whole script on every
    interaction and this walks the project tree.
    """
    try:
        return project_fingerprint(root)
    except OSError:
        return ""


@st.cache_data(ttl=30, show_spinner=False)
def available_models() -> list[str]:
    try:
        return llm.list_local_models()
    except llm.OllamaError:
        return []


def init_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("project_dir", "")
    st.session_state.setdefault("last_audio_hash", "")
    st.session_state.setdefault("index_stats", None)
    st.session_state.setdefault("browse_at", "")
    st.session_state.setdefault("browsing", False)
    st.session_state.setdefault("pending_dir", "")


def apply_pending_dir() -> None:
    """Move a folder chosen by the picker into the text input.

    Streamlit forbids writing to a widget's session_state key after that
    widget has been created, so the picker parks its result in
    `pending_dir` and it is applied here - before the input is built.
    """
    pending = st.session_state.get("pending_dir")
    if pending:
        st.session_state["project_dir"] = pending
        st.session_state["pending_dir"] = ""


init_state()
apply_pending_dir()
store = get_store(str(config.CHROMA_DIR))
models = available_models()

# =========================================================== sidebar
with st.sidebar:
    st.title("🎙️ Dev Assistant")
    st.caption("Fully on-device. No code leaves this machine.")

    # --- 1. model ---------------------------------------------------
    st.subheader("1 · Model")
    if models:
        default_idx = (
            models.index(config.DEFAULT_MODEL)
            if config.DEFAULT_MODEL in models
            else 0
        )
        model = st.selectbox("Ollama model", models, index=default_idx)
        st.success("Ollama is running", icon="✅")
    else:
        model = config.DEFAULT_MODEL
        st.error("Ollama not reachable — run `ollama serve`", icon="⚠️")

    with st.expander("Retrieval settings"):
        top_k = st.slider("Chunks retrieved (top-k)", 1, 10, config.TOP_K)
        chunk_size = st.slider("Chunk size", 300, 2000, config.CHUNK_SIZE, step=100)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.1, step=0.1)

    settings = config.Settings(
        model=model,
        top_k=top_k,
        chunk_size=chunk_size,
        chunk_overlap=min(config.CHUNK_OVERLAP, chunk_size // 4),
        temperature=temperature,
    )

    # --- 2. project -------------------------------------------------
    st.subheader("2 · Project")
    # The widget owns its value through `key`. Passing `value=` from
    # session_state instead makes Streamlit rebuild the widget every time that
    # value changes, discarding whatever the user was typing: the box snaps
    # back to the previous path and the project folder cannot be changed.
    st.text_input(
        "Project directory",
        key="project_dir",
        placeholder=r"C:\Users\you\PycharmProjects\my-project",
        help="The folder whose code the assistant should read. Paste the path "
             "and press Index project — surrounding quotes are fine.",
    )
    project_path = normalize_project_path(st.session_state["project_dir"])

    # --- pick a folder instead of typing one ------------------------
    pick_col, browse_col = st.columns(2)
    with pick_col:
        if st.button("📂 Browse…", width="stretch",
                     help="Open the Windows folder picker"):
            chosen = picker.choose_directory_dialog(
                initial=str(project_path) if project_path else None
            )
            if chosen:
                st.session_state["pending_dir"] = chosen
                st.session_state["browsing"] = False
                st.rerun()
            else:
                # Cancelled, or no native dialog available on this session.
                st.session_state["browsing"] = True
                st.rerun()
    with browse_col:
        label = "✖ Close list" if st.session_state["browsing"] else "🗂 Folder list"
        if st.button(label, width="stretch",
                     help="Browse folders inside the app"):
            st.session_state["browsing"] = not st.session_state["browsing"]
            if st.session_state["browsing"] and not st.session_state["browse_at"]:
                st.session_state["browse_at"] = str(
                    picker.default_browse_root(st.session_state["project_dir"])
                )
            st.rerun()

    if st.session_state["browsing"]:
        listing = picker.list_subdirectories(
            st.session_state["browse_at"]
            or picker.default_browse_root(st.session_state["project_dir"])
        )
        st.caption(f"📍 `{listing.current}`")
        if listing.error:
            st.error(listing.error)

        if st.button("✅ Use this folder", width="stretch"):
            st.session_state["pending_dir"] = str(listing.current)
            st.session_state["browsing"] = False
            st.rerun()
        if listing.parent is not None and st.button("⬆️ Up one level",
                                                    width="stretch"):
            st.session_state["browse_at"] = str(listing.parent)
            st.rerun()

        if listing.subdirectories:
            with st.container(height=220):
                for sub in listing.subdirectories:
                    if st.button(f"📁 {sub.name}", key=f"cd::{sub}",
                                 width="stretch"):
                        st.session_state["browse_at"] = str(sub)
                        st.rerun()
        elif not listing.error:
            st.caption("No sub-folders here.")

    # Deliberately never disabled. A disabled button swallows the very click
    # that enables it, so a freshly typed path needed two presses before it
    # indexed anything — which reads as "the button does nothing". Validate
    # on click instead.
    if st.button("📚 Index project", width="stretch"):
        if project_path is None:
            st.error("Type the folder you want indexed first.")
        elif not project_path.is_dir():
            st.error(f"Not a folder: {project_path}")
        else:
            bar = st.progress(0.0, text="Starting…")
            status = st.empty()

            def on_progress(rel_path, stats):
                status.caption(
                    f"{stats.files_indexed} files · {stats.chunks} chunks"
                )
                bar.progress(min(0.99, stats.files_indexed / 200.0),
                             text=rel_path[-46:])

            with st.spinner("Indexing…"):
                stats = build_index(project_path, store, settings=settings,
                                    progress=on_progress)
            bar.progress(1.0, text="Done")
            st.session_state["index_stats"] = stats.as_dict()
            if stats.files_indexed:
                st.success(
                    f"Indexed {stats.files_indexed} files → {stats.chunks} chunks"
                )
            else:
                st.warning(
                    "No indexable files found there. Check the folder, or add "
                    "the extensions you need to SOURCE_EXTENSIONS."
                )

    # --- what the store actually holds, and where it came from -------
    indexed_root = store.indexed_root
    if store.count():
        st.caption(
            f"Vector store holds **{store.count()}** chunks "
            f"from **{len(store.sources())}** files"
        )
        if indexed_root:
            st.caption(f"Indexed from `{indexed_root}`")

        if project_path and indexed_root and (
            str(project_path.resolve()) != indexed_root
        ):
            st.warning(
                "The index was built from a different folder. Press "
                "**Index project** to rebuild it for this one.",
                icon="⚠️",
            )
        elif indexed_root and current_fingerprint(indexed_root) != (
            store.indexed_fingerprint
        ):
            # Files were added, edited or deleted since the index was built.
            # Without this the assistant simply cannot see a new file, and
            # says it is "not in the excerpts" with no hint why.
            st.warning(
                "Files have changed on disk since this index was built. "
                "Press **Index project** to pick them up.",
                icon="🔄",
            )
    else:
        st.caption("Vector store is empty — nothing indexed yet.")

    if st.button("🗑️ Clear conversation", width="stretch"):
        st.session_state["messages"] = []
        st.rerun()

# =========================================================== main pane
assistant = Assistant(store, settings=settings)

st.title("Local Voice AI Dev Assistant")
st.caption(
    "Ask about your project by voice or text. Answers are grounded in your "
    "own files and generated by a model running on this machine."
)

with st.expander("What it can do"):
    st.markdown(
        "- **Check syntax** — *what is wrong with `samples/broken_function.py`?* "
        "Uses Python's own parser, so the line and column are exact.\n"
        "- **Scan the whole project** — *are there any errors in the whole "
        "codebase?* Every file, syntax plus lint, in one report.\n"
        "- **Run the tests** — *do my tests pass?* Actually executes pytest "
        "and reports the real result, including runtime errors.\n"
        "- **Review formatting** — *how should `samples/messy_style.py` be "
        "formatted?* Reports what `ruff` would change, including the diff.\n"
        "- **Explain code** — *explain the function `normalize_project_path`* "
        "Looks up the real definition rather than a nearby chunk.\n"
        "- **Write unit tests** — *write a unit test for `average`* Matches "
        "the conventions of the tests already in your project.\n\n"
        "Running the tests executes your project's own code."
    )

if store.count() == 0:
    st.info(
        "No project indexed yet. Point the sidebar at a project folder and "
        "press **Index project**.",
        icon="👈",
    )

# --- replay history ---------------------------------------------------
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"📄 Sources ({len(msg['sources'])})"):
                for src in msg["sources"]:
                    st.markdown(f"**{src['citation']}**")
                    st.code(src["text"], language="python")

# --- voice input ------------------------------------------------------
question: str | None = None

st.markdown("##### 🎤 Ask by voice")
audio = st.audio_input("Record your question", label_visibility="collapsed")

if audio is not None:
    audio_bytes = audio.getvalue()
    digest = hashlib.sha1(audio_bytes).hexdigest()
    if digest != st.session_state["last_audio_hash"]:
        st.session_state["last_audio_hash"] = digest
        with st.spinner("Transcribing with Whisper…"):
            try:
                transcript = stt.transcribe_bytes(audio_bytes)
            except stt.TranscriptionError as exc:
                st.error(str(exc))
                transcript = None
        if transcript and transcript.text:
            question = transcript.text
            st.caption(f"Heard: _{transcript.text}_")
        elif transcript is not None:
            st.warning("Nothing was picked up — try again, a bit closer to the mic.")

# --- text input -------------------------------------------------------
typed = st.chat_input("…or type your question")
if typed:
    question = typed

# --- answer -----------------------------------------------------------
if question:
    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if not models:
            st.error("Ollama is not running. Start it with `ollama serve`.")
        else:
            try:
                # The tool phase runs before the first token: a full test
                # run can take a while, so say what is happening rather
                # than showing a blank bubble.
                with st.spinner("Analysing your project…"):
                    context, stream = assistant.stream_context(question)
                chunks = context.chunks

                # Tool output first, and labelled: these findings come from
                # Python's parser and ruff, not from the model, and the user
                # should be able to tell the difference at a glance.
                if context.facts:
                    st.caption(
                        "🔧 Local analysis · "
                        f"{TASK_LABELS[context.intent.kind]}"
                    )
                    with st.expander("Tool findings (exact, not generated)",
                                     expanded=True):
                        st.code(context.facts, language="text")
                if chunks:
                    with st.expander(f"📄 Reading {len(chunks)} excerpt(s)"):
                        for c in chunks:
                            label = (
                                "named file"
                                if c.distance == 0.0
                                else f"distance {c.distance:.3f}"
                            )
                            st.markdown(f"**{c.citation}** · {label}")
                            st.code(c.text, language="python")
                answer_text = st.write_stream(stream)
            except llm.OllamaError as exc:
                answer_text = f"⚠️ {exc}"
                chunks = []
                st.error(answer_text)

            st.session_state["messages"].append(
                {
                    "role": "assistant",
                    "content": answer_text,
                    "sources": [
                        {"citation": c.citation, "text": c.text} for c in chunks
                    ],
                }
            )
