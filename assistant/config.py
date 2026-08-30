"""Central configuration.

Values are plain constants so they can be imported by tests without side
effects. Anything a user might reasonably change is exposed in the UI.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# --- Ollama -----------------------------------------------------------------
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

# Models that run fully on-device. Cloud-suffixed Ollama models (e.g.
# "*-cloud") are intentionally excluded: sending code to a remote endpoint
# would defeat the confidentiality guarantee this project is built on.
LOCAL_MODELS = ["qwen3:1.7b", "llama3:latest"]
DEFAULT_MODEL = "qwen3:1.7b"

# Context window for generation. The prompt is a system prompt plus top_k
# code chunks, so ~8k tokens is comfortable headroom without wasting RAM.
NUM_CTX = 8192

# Keep the model resident between questions - a cold load costs ~10-20 s.
KEEP_ALIVE = "10m"

# --- Whisper ----------------------------------------------------------------
# "base" is the sweet spot on an 8 GB CPU machine; "small" is ~2x slower.
WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL_SIZE", "base")
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE_TYPE = "int8"

# --- Indexing ---------------------------------------------------------------
# Anchored to the repository root, not the working directory: `streamlit run`
# can be launched from anywhere, and a relative path would silently create a
# second, empty store instead of reusing the one that was already built.
APP_ROOT = Path(__file__).resolve().parent.parent
CHROMA_DIR = Path(os.environ.get("CHROMA_DIR") or APP_ROOT / ".chroma")
COLLECTION_NAME = "project_context"

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
TOP_K = 4

# When a question names a file, that file's chunks are pulled in whole rather
# than left to similarity search. Capped so a large file cannot crowd the
# context window on its own.
FILE_CHUNK_LIMIT = 8

# Files worth indexing, mapped to the LangChain Language used for
# syntax-aware splitting (None = plain recursive splitting).
SOURCE_EXTENSIONS: dict[str, str | None] = {
    ".py": "python",
    ".js": "js",
    ".ts": "ts",
    ".java": "java",
    ".c": "c",
    ".cpp": "cpp",
    ".cs": "csharp",
    ".go": "go",
    ".rs": "rust",
    ".php": "php",
    ".rb": "ruby",
    ".md": "markdown",
    ".html": "html",
    ".txt": None,
    ".json": None,
    ".yaml": None,
    ".yml": None,
    ".toml": None,
}

# Directories never worth indexing.
IGNORED_DIRS = {
    ".git", ".svn", ".hg",
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    "node_modules", "venv", ".venv", "env",
    "dist", "build", "target", "out", "bin", "obj",
    ".idea", ".vscode", ".chroma", "chroma_db",
    "site-packages", ".next", ".cache",
}

MAX_FILE_BYTES = 400_000  # skip generated/minified monsters


@dataclass
class Settings:
    """Runtime settings, overridable from the UI."""

    model: str = DEFAULT_MODEL
    top_k: int = TOP_K
    chunk_size: int = CHUNK_SIZE
    chunk_overlap: int = CHUNK_OVERLAP
    temperature: float = 0.1
    extensions: set[str] = field(
        default_factory=lambda: set(SOURCE_EXTENSIONS.keys())
    )
