"""Phase 1a - walk a project directory, chunk it, embed it into Chroma.

Split into small pure functions (`iter_source_files`, `read_text`,
`split_text`) so the traversal and chunking logic can be unit-tested
without a vector store or a model.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

from . import config


@dataclass
class IndexStats:
    files_seen: int = 0
    files_indexed: int = 0
    files_skipped: int = 0
    chunks: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "files_seen": self.files_seen,
            "files_indexed": self.files_indexed,
            "files_skipped": self.files_skipped,
            "chunks": self.chunks,
        }


def iter_source_files(
    root: Path,
    extensions: set[str] | None = None,
    ignored_dirs: set[str] | None = None,
    max_bytes: int = config.MAX_FILE_BYTES,
) -> Iterator[Path]:
    """Yield indexable source files under `root`.

    Prunes ignored directories in-place so we never descend into
    node_modules or .venv, which is what makes this usable on a real repo.
    """
    root = Path(root)
    exts = extensions if extensions is not None else set(config.SOURCE_EXTENSIONS)
    ignored = ignored_dirs if ignored_dirs is not None else config.IGNORED_DIRS

    import os

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in ignored and not d.startswith(".")]
        for name in sorted(filenames):
            path = Path(dirpath) / name
            if path.suffix.lower() not in exts:
                continue
            try:
                if path.stat().st_size > max_bytes:
                    continue
            except OSError:
                continue
            yield path


def read_text(path: Path) -> str | None:
    """Read a source file, tolerating the encodings found in the wild."""
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return Path(path).read_text(encoding=encoding)
        except (UnicodeDecodeError, ValueError):
            continue
        except OSError:
            return None
    return None


def _splitter(suffix: str, chunk_size: int, chunk_overlap: int):
    """Return a syntax-aware splitter when LangChain knows the language."""
    lang_name = config.SOURCE_EXTENSIONS.get(suffix.lower())
    if lang_name:
        language = getattr(Language, lang_name.upper(), None)
        if language is not None:
            return RecursiveCharacterTextSplitter.from_language(
                language=language,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )


def split_text(
    text: str,
    suffix: str = ".py",
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
) -> list[str]:
    """Split source text into chunks, respecting language structure."""
    if not text or not text.strip():
        return []
    chunks = _splitter(suffix, chunk_size, chunk_overlap).split_text(text)
    return [c for c in chunks if c.strip()]


def chunk_id(rel_path: str, position: int, content: str) -> str:
    """Stable, content-addressed id so re-indexing overwrites cleanly."""
    digest = hashlib.sha1(content.encode("utf-8", "replace")).hexdigest()[:10]
    return f"{rel_path}::{position}::{digest}"
