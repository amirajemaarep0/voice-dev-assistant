"""Phase 1a - walk a project directory, chunk it, embed it into Chroma.

Split into small pure functions (`iter_source_files`, `read_text`,
`split_text`) so the traversal and chunking logic can be unit-tested
without a vector store or a model.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator

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


# A path-ish token: word characters, dots, slashes and dashes, ending in a
# short alphabetic extension. The extension must start with a letter so that
# version numbers and "e.g." are not mistaken for filenames.
_FILE_TOKEN = re.compile(r"[\w./\\-]*\w\.[A-Za-z][A-Za-z0-9]{0,5}\b")


def _normalize_source(token: str) -> str:
    """Reduce a written path to the form used in chunk metadata."""
    return token.replace("\\", "/").lstrip("./").lower()


def extract_file_references(
    question: str,
    known_sources: Iterable[str],
) -> tuple[list[str], list[str]]:
    """Split the filenames a question mentions into indexed and missing.

    Embedding similarity is close to useless for "what is in stt.py?" - the
    question shares almost no vocabulary with the file's contents, so the
    named file often does not appear in the top-k at all. Resolving the name
    against the index first is what makes file-scoped questions work.

    Returns (sources present in the index, names that are not).
    """
    sources = list(known_sources)
    by_normalized = {_normalize_source(s): s for s in sources}
    by_basename: dict[str, list[str]] = {}
    for source in sources:
        by_basename.setdefault(source.rsplit("/", 1)[-1].lower(), []).append(source)

    matched: list[str] = []
    missing: list[str] = []
    for raw in _FILE_TOKEN.findall(question or ""):
        token = _normalize_source(raw)
        if not token:
            continue
        hits: list[str] = []
        if token in by_normalized:
            hits = [by_normalized[token]]
        else:
            # "test_store.py" should find "tests/test_store.py"; a written
            # sub-path like "tests/test_store.py" must still match exactly.
            suffix_hits = [s for s in sources if _normalize_source(s).endswith("/" + token)]
            hits = suffix_hits or by_basename.get(token.rsplit("/", 1)[-1], [])

        if hits:
            matched.extend(h for h in hits if h not in matched)
        elif Path(token).suffix in config.SOURCE_EXTENSIONS and raw not in missing:
            # Only claim a file is missing when it is the kind of file this
            # project would have indexed - otherwise every "vs." is a report.
            missing.append(raw)
    return matched, missing


def normalize_project_path(raw: str | None) -> Path | None:
    """Turn what a user pastes into the UI into a usable path.

    Windows' "Copy as path" wraps the path in double quotes, and a pasted
    path routinely carries trailing whitespace or a newline. Left as-is,
    those characters make an existing folder look like it does not exist.
    Returns None when there is nothing usable to work with.
    """
    if not raw:
        return None
    text = raw.strip().strip('"').strip("'").strip()
    if not text:
        return None
    try:
        return Path(text).expanduser()
    except (OSError, ValueError):
        return None


def project_fingerprint(
    root: Path | str,
    extensions: set[str] | None = None,
) -> str:
    """A cheap signature of the indexable files under `root`.

    File count plus the newest modification time. Creating, deleting or
    editing a source file changes it, which is enough for the UI to notice
    that the index no longer matches the disk - the failure that otherwise
    looks like "the assistant cannot see my new file".
    """
    count = 0
    latest = 0.0
    for path in iter_source_files(root, extensions=extensions):
        count += 1
        try:
            latest = max(latest, path.stat().st_mtime)
        except OSError:
            continue
    return f"{count}:{latest:.0f}"


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
