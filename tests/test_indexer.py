"""Traversal and chunking - the part that decides answer quality."""
from __future__ import annotations

from pathlib import Path

from assistant.indexer import (
    chunk_id,
    iter_source_files,
    read_text,
    split_text,
)


class TestIterSourceFiles:
    def test_finds_source_files(self, sample_project: Path):
        found = {p.name for p in iter_source_files(sample_project)}
        assert "calc.py" in found
        assert "notes.md" in found
        assert "README.md" in found

    def test_skips_ignored_directories(self, sample_project: Path):
        paths = [str(p) for p in iter_source_files(sample_project)]
        assert not any("node_modules" in p for p in paths)
        assert not any(".venv" in p for p in paths)

    def test_skips_unknown_extensions(self, sample_project: Path):
        assert not any(p.suffix == ".png" for p in iter_source_files(sample_project))

    def test_extension_filter_is_respected(self, sample_project: Path):
        found = list(iter_source_files(sample_project, extensions={".py"}))
        assert found and all(p.suffix == ".py" for p in found)

    def test_skips_oversized_files(self, tmp_path: Path):
        big = tmp_path / "big.py"
        big.write_text("x = 1\n" * 50_000, encoding="utf-8")
        assert list(iter_source_files(tmp_path, max_bytes=1000)) == []

    def test_empty_directory(self, tmp_path: Path):
        assert list(iter_source_files(tmp_path)) == []


class TestReadText:
    def test_reads_utf8(self, tmp_path: Path):
        f = tmp_path / "a.py"
        f.write_text("# héllo\n", encoding="utf-8")
        assert "héllo" in read_text(f)

    def test_missing_file_returns_none(self, tmp_path: Path):
        assert read_text(tmp_path / "nope.py") is None

    def test_binary_does_not_raise(self, tmp_path: Path):
        f = tmp_path / "b.py"
        f.write_bytes(b"\x00\x01\x02\xff")
        read_text(f)  # latin-1 fallback: must not raise


class TestSplitText:
    def test_empty_input(self):
        assert split_text("") == []
        assert split_text("   \n  ") == []

    def test_short_file_is_one_chunk(self):
        assert len(split_text("def f():\n    return 1\n", ".py")) == 1

    def test_long_file_is_split(self):
        text = "\n\n".join(f"def f{i}():\n    return {i}" for i in range(200))
        chunks = split_text(text, ".py", chunk_size=400, chunk_overlap=50)
        assert len(chunks) > 1
        assert all(c.strip() for c in chunks)

    def test_python_splitter_keeps_definitions_intact(self):
        text = "\n\n".join(f"def f{i}():\n    return {i}" for i in range(40))
        chunks = split_text(text, ".py", chunk_size=300, chunk_overlap=0)
        # A language-aware splitter should not leave a chunk starting mid-body.
        assert not any(c.lstrip().startswith("return") for c in chunks)

    def test_unknown_extension_still_splits(self):
        assert split_text("hello world " * 200, ".txt", chunk_size=200)


class TestChunkId:
    def test_is_deterministic(self):
        assert chunk_id("a.py", 0, "x = 1") == chunk_id("a.py", 0, "x = 1")

    def test_differs_by_content(self):
        assert chunk_id("a.py", 0, "x = 1") != chunk_id("a.py", 0, "x = 2")

    def test_differs_by_position(self):
        assert chunk_id("a.py", 0, "x = 1") != chunk_id("a.py", 1, "x = 1")
