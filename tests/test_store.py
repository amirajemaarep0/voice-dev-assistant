"""build_index against an in-memory store (no chromadb needed)."""
from __future__ import annotations

from pathlib import Path

from assistant import config
from assistant.store import Retrieved, build_index


class TestBuildIndex:
    def test_indexes_sample_project(self, sample_project: Path, fake_store):
        stats = build_index(sample_project, fake_store)
        assert stats.files_indexed >= 3
        assert stats.chunks >= stats.files_indexed
        assert fake_store.count() == stats.chunks

    def test_resets_before_indexing(self, sample_project: Path, fake_store):
        build_index(sample_project, fake_store)
        build_index(sample_project, fake_store)
        assert fake_store.reset_calls == 2

    def test_reindex_is_not_cumulative(self, sample_project: Path, fake_store):
        first = build_index(sample_project, fake_store).chunks
        second = build_index(sample_project, fake_store).chunks
        assert first == second == fake_store.count()

    def test_metadata_uses_forward_slashes(self, sample_project: Path, fake_store):
        build_index(sample_project, fake_store)
        sources = {m["source"] for m in fake_store.metadatas}
        assert any("src/calc.py" == s for s in sources)
        assert not any("\\" in s for s in sources)

    def test_ids_are_unique(self, sample_project: Path, fake_store):
        build_index(sample_project, fake_store)
        assert len(fake_store.ids) == len(set(fake_store.ids))

    def test_progress_callback_is_called(self, sample_project: Path, fake_store):
        seen = []
        build_index(sample_project, fake_store,
                    progress=lambda rel, stats: seen.append(rel))
        assert seen

    def test_extension_filter(self, sample_project: Path, fake_store):
        settings = config.Settings(extensions={".py"})
        build_index(sample_project, fake_store, settings=settings)
        assert all(m["source"].endswith(".py") for m in fake_store.metadatas)

    def test_empty_project(self, tmp_path: Path, fake_store):
        stats = build_index(tmp_path, fake_store)
        assert stats.files_indexed == 0 and stats.chunks == 0

    def test_batching_flushes_everything(self, sample_project: Path, fake_store):
        stats = build_index(sample_project, fake_store, batch_size=1)
        assert fake_store.count() == stats.chunks


class TestRetrieved:
    def test_citation_format(self):
        r = Retrieved(text="x", source="src/a.py", position=3, distance=0.1)
        assert r.citation == "src/a.py (chunk 3)"
