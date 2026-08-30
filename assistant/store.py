"""Phase 1b - persistent vector store (Chroma) and index building.

Embeddings use Chroma's bundled all-MiniLM-L6-v2 running on onnxruntime.
That is the same model named in the report, but it avoids a PyTorch
install - which matters on an 8 GB machine that also has to hold an LLM.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from . import config
from .indexer import IndexStats, chunk_id, iter_source_files, read_text, split_text


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

    # --- read side ----------------------------------------------------
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
    return stats
