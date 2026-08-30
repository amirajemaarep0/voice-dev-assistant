"""The sidebar's folder handling - where both reported UI bugs lived.

`normalize_project_path` is the pure half of "did the user give us a real
folder?", and `indexed_root` is what lets the UI say which project the
persisted store was actually built from.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from assistant.indexer import normalize_project_path
from assistant.store import build_index


class TestNormalizeProjectPath:
    @pytest.mark.parametrize("raw", ["", "   ", None, '""', "'"])
    def test_nothing_usable_returns_none(self, raw):
        assert normalize_project_path(raw) is None

    def test_plain_path(self, tmp_path: Path):
        assert normalize_project_path(str(tmp_path)) == tmp_path

    def test_strips_surrounding_whitespace(self, tmp_path: Path):
        assert normalize_project_path(f"  {tmp_path}\n") == tmp_path

    def test_strips_quotes_from_copy_as_path(self, tmp_path: Path):
        """Windows Explorer's "Copy as path" wraps the path in quotes."""
        assert normalize_project_path(f'"{tmp_path}"') == tmp_path

    def test_expands_home(self):
        result = normalize_project_path("~/projects/demo")
        assert result is not None
        assert "~" not in str(result)

    def test_result_is_checkable_as_a_directory(self, sample_project: Path):
        assert normalize_project_path(f'  "{sample_project}"  ').is_dir()
        assert not normalize_project_path(str(sample_project / "nope")).is_dir()


class TestIndexedRoot:
    def test_build_index_records_the_root(self, sample_project: Path, fake_store):
        build_index(sample_project, fake_store)
        assert fake_store.indexed_root == str(sample_project.resolve())

    def test_root_is_cleared_before_it_is_rewritten(
        self, sample_project: Path, fake_store
    ):
        """A crash mid-index must not leave a root the store never finished."""
        build_index(sample_project, fake_store)
        assert fake_store.root_writes[0] == ""
        assert fake_store.root_writes[-1] == str(sample_project.resolve())

    def test_reindexing_elsewhere_replaces_the_root(
        self, sample_project: Path, tmp_path: Path, fake_store
    ):
        other = tmp_path / "other"
        other.mkdir()
        (other / "main.py").write_text("x = 1\n", encoding="utf-8")

        build_index(sample_project, fake_store)
        build_index(other, fake_store)
        assert fake_store.indexed_root == str(other.resolve())


class TestPersistedIndexedRoot:
    """Same contract, but against the real Chroma-backed store."""

    def test_round_trips_through_the_persist_dir(self, tmp_path: Path):
        from assistant.store import ProjectStore

        persist = tmp_path / "store"
        store = ProjectStore(persist_dir=persist, collection_name="roundtrip")
        assert store.indexed_root == ""

        store.set_indexed_root(tmp_path)
        assert store.indexed_root == str(tmp_path.resolve())

        # A fresh handle on the same directory sees it too - this is what the
        # sidebar reads after a restart.
        reopened = ProjectStore(persist_dir=persist, collection_name="roundtrip")
        assert reopened.indexed_root == str(tmp_path.resolve())
