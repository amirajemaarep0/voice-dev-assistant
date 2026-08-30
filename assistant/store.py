"""Phase 1b - persistent vector store (Chroma) and index building.

Embeddings use Chroma's bundled all-MiniLM-L6-v2 running on onnxruntime.
That is the same model named in the report, but it avoids a PyTorch
install - which matters on an 8 GB machine that also has to hold an LLM.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from . import config
from .indexer import (
    IndexStats,
    chunk_id,
    iter_source_files,
    project_fingerprint,
    read_text,
    split_text,
)

# Records which folder the persisted collection was built from. Without it
# the UI cannot tell a freshly indexed project from a stale index left over
# from a different one - which looks exactly like "it ignores my files".
INDEX_META_FILE = "index_meta.json"


@dataclass
class Retrieved:
    """One retrieved chunk, with enough metadata to cite it."""

    text: str
    source: str
    position: int
    distance: float

    @property
    def citation(self) -> str:
        return f"{self.source} (chunk {self.position})"


class ProjectStore:
    """Thin wrapper over a persistent Chroma collection."""

    def __init__(self, persist_dir: Path | str = config.CHROMA_DIR,
                 collection_name: str = config.COLLECTION_NAME) -> None:
        import logging
        import os

        # Chroma's bundled posthog client is version-mismatched and spams
        # "Failed to send telemetry event" on every call. We disable
        # telemetry anyway; this silences the failed attempts too.
        os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
        logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)

        import chromadb
        from chromadb.config import Settings as ChromaSettings
        from chromadb.utils import embedding_functions

        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._embedder = embedding_functions.DefaultEmbeddingFunction()
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=self._embedder,
            metadata={"hnsw:space": "cosine"},
        )

    # --- write side ---------------------------------------------------
    def reset(self) -> None:
        """Drop and recreate the collection (used when re-indexing)."""
        name = self._collection.name
        self._client.delete_collection(name)
        self._collection = self._client.get_or_create_collection(
            name=name,
            embedding_function=self._embedder,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, ids, documents, metadatas) -> None:
        if not ids:
            return
        self._collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

    def count(self) -> int:
        return self._collection.count()

    # --- provenance ---------------------------------------------------
    @property
    def _meta_path(self) -> Path:
        return self.persist_dir / INDEX_META_FILE

    def _meta(self) -> dict:
        try:
            return json.loads(self._meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    @property
    def indexed_root(self) -> str:
        """Absolute path this collection was last built from ("" if unknown)."""
        return str(self._meta().get("root", ""))

    @property
    def indexed_fingerprint(self) -> str:
        """The project fingerprint captured when the index was built."""
        return str(self._meta().get("fingerprint", ""))

    def set_indexed_root(
        self, root: Path | str | None, fingerprint: str = ""
    ) -> None:
        payload = {
            "root": str(Path(root).resolve()) if root else "",
            "fingerprint": fingerprint if root else "",
        }
        try:
            self._meta_path.write_text(json.dumps(payload), encoding="utf-8")
        except OSError:  # pragma: no cover - read-only persist dir
            pass

    # --- read side ----------------------------------------------------
    def sources(self) -> list[str]:
        """Every distinct file path currently in the collection."""
        if self.count() == 0:
            return []
        got = self._collection.get(include=["metadatas"])
        return sorted({str(m.get("source", "")) for m in got["metadatas"] if m})

    def chunks_for_source(
        self, source: str, limit: int = config.FILE_CHUNK_LIMIT
    ) -> list[Retrieved]:
        """Every chunk of one file, in order.

        Distance is 0.0: this is an exact metadata match, not a similarity
        hit, and the UI should not present it as one.
        """
        got = self._collection.get(
            where={"source": source}, include=["documents", "metadatas"]
        )
        out = [
            Retrieved(
                text=doc,
                source=str(meta.get("source", source)),
                position=int(meta.get("position", 0)),
                distance=0.0,
            )
            for doc, meta in zip(got["documents"], got["metadatas"])
        ]
        out.sort(key=lambda r: r.position)
        return out[:limit]

    def search(self, query: str, top_k: int = config.TOP_K) -> list[Retrieved]:
        if not query.strip() or self.count() == 0:
            return []
        n = min(top_k, self.count())
        res = self._collection.query(query_texts=[query], n_results=n)
        out: list[Retrieved] = []
        for doc, meta, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            out.append(
                Retrieved(
                    text=doc,
                    source=str(meta.get("source", "?")),
                    position=int(meta.get("position", 0)),
                    distance=float(dist),
                )
            )
        return out


def build_index(
    root: Path | str,
    store: ProjectStore,
    settings: config.Settings | None = None,
    progress: Callable[[str, IndexStats], None] | None = None,
    batch_size: int = 128,
) -> IndexStats:
    """Index every source file under `root` into `store`."""
    settings = settings or config.Settings()
    root = Path(root)
    stats = IndexStats()
    store.reset()
    # Cleared first, restored on success: a run that dies halfway must not
    # leave the UI claiming the new project is fully indexed.
    store.set_indexed_root(None)

    ids: list[str] = []
    docs: list[str] = []
    metas: list[dict] = []

    def flush() -> None:
        store.add(ids, docs, metas)
        ids.clear()
        docs.clear()
        metas.clear()

    for path in iter_source_files(root, extensions=settings.extensions):
        stats.files_seen += 1
        text = read_text(path)
        if text is None:
            stats.files_skipped += 1
            continue

        chunks = split_text(
            text,
            suffix=path.suffix,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        if not chunks:
            stats.files_skipped += 1
            continue

        rel = str(path.relative_to(root)).replace("\\", "/")
        for position, chunk in enumerate(chunks):
            ids.append(chunk_id(rel, position, chunk))
            docs.append(chunk)
            metas.append({"source": rel, "position": position})

        stats.files_indexed += 1
        stats.chunks += len(chunks)

        if progress is not None:
            progress(rel, stats)
        if len(ids) >= batch_size:
            flush()

    flush()
    store.set_indexed_root(
        root, fingerprint=project_fingerprint(root, settings.extensions)
    )
    return stats
