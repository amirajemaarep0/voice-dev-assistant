"""Answering questions that name a file.

Pure similarity search is unreliable here: "what is in stt.py?" shares
almost no vocabulary with stt.py's contents, so the named file routinely
fails to appear in the top-k at all. These tests pin the behaviour that
replaces it.
"""
from __future__ import annotations

from assistant import config, llm
from assistant.indexer import extract_file_references, project_fingerprint
from assistant.pipeline import Assistant
from assistant.store import Retrieved

from .conftest import FakeStore

INDEXED = [
    "app.py",
    "assistant/stt.py",
    "assistant/store.py",
    "tests/test_store.py",
    "README.md",
]


class TestExtractFileReferences:
    def test_exact_path(self):
        matched, missing = extract_file_references("explain assistant/stt.py", INDEXED)
        assert matched == ["assistant/stt.py"]
        assert missing == []

    def test_bare_filename_resolves_to_its_path(self):
        matched, _ = extract_file_references("what is in stt.py?", INDEXED)
        assert matched == ["assistant/stt.py"]

    def test_leading_slash_and_backslashes(self):
        """Users paste "/tests/test_store.py" and "tests\\test_store.py"."""
        for written in ("/tests/test_store.py", r"tests\test_store.py"):
            matched, missing = extract_file_references(f"check {written}", INDEXED)
            assert matched == ["tests/test_store.py"], written
            assert missing == []

    def test_case_insensitive(self):
        matched, _ = extract_file_references("open README.MD", INDEXED)
        assert matched == ["README.md"]

    def test_unknown_file_is_reported_missing(self):
        matched, missing = extract_file_references(
            "can you check /tests/test_function.py file?", INDEXED
        )
        assert matched == []
        assert missing == ["/tests/test_function.py"]

    def test_several_files_at_once(self):
        matched, _ = extract_file_references("compare app.py and stt.py", INDEXED)
        assert matched == ["app.py", "assistant/stt.py"]

    def test_prose_is_not_mistaken_for_a_filename(self):
        """Only extensions this project indexes may be called missing."""
        for question in ("what does it do, e.g. on startup?",
                         "is version 1.5 supported?",
                         "explain the flow"):
            matched, missing = extract_file_references(question, INDEXED)
            assert (matched, missing) == ([], []), question

    def test_no_index_means_nothing_matches(self):
        matched, missing = extract_file_references("check app.py", [])
        assert matched == []
        assert missing == ["app.py"]


class TestFileScopedRetrieval:
    def _store(self) -> FakeStore:
        store = FakeStore(results=[
            Retrieved(text="# unrelated top hit", source="README.md",
                      position=0, distance=0.51),
        ])
        store.add(
            ids=["stt::0", "stt::1"],
            documents=["def transcribe_file(path):", "def transcribe_bytes(b):"],
            metadatas=[{"source": "assistant/stt.py", "position": 0},
                       {"source": "assistant/stt.py", "position": 1}],
        )
        return store

    def test_named_file_is_retrieved_even_when_similarity_misses_it(self):
        assistant = Assistant(self._store(), config.Settings(top_k=1))
        chunks = assistant.retrieve("what is in assistant/stt.py?")
        assert "assistant/stt.py" in {c.source for c in chunks}

    def test_named_file_chunks_come_first_and_in_order(self):
        assistant = Assistant(self._store(), config.Settings(top_k=1))
        chunks = assistant.retrieve("explain assistant/stt.py")
        named = [c for c in chunks if c.source == "assistant/stt.py"]
        assert [c.position for c in named] == [0, 1]
        assert chunks[0].source == "assistant/stt.py"

    def test_similarity_hits_still_fill_the_rest(self):
        assistant = Assistant(self._store(), config.Settings(top_k=1))
        chunks = assistant.retrieve("explain assistant/stt.py")
        assert "README.md" in {c.source for c in chunks}

    def test_no_duplicate_chunks(self):
        assistant = Assistant(self._store(), config.Settings(top_k=4))
        chunks = assistant.retrieve("assistant/stt.py and README.md")
        keys = [(c.source, c.position) for c in chunks]
        assert len(keys) == len(set(keys))

    def test_plain_question_is_unaffected(self):
        assistant = Assistant(self._store(), config.Settings(top_k=1))
        chunks = assistant.retrieve("how does transcription work?")
        assert [c.source for c in chunks] == ["README.md"]

    def test_missing_file_is_reported(self):
        assistant = Assistant(self._store(), config.Settings(top_k=1))
        _, missing = assistant.gather("check tests/test_function.py")
        assert missing == ["tests/test_function.py"]


class TestMissingFileReachesTheModel:
    def test_prompt_names_the_missing_file(self, sample_chunks):
        prompt = llm.build_prompt("check test_function.py", sample_chunks,
                                  missing_files=["test_function.py"])
        assert "test_function.py" in prompt
        assert "NOT in the" in prompt

    def test_prompt_is_unchanged_when_nothing_is_missing(self, sample_chunks):
        with_arg = llm.build_prompt("q", sample_chunks, missing_files=[])
        without = llm.build_prompt("q", sample_chunks)
        assert with_arg == without

    def test_empty_index_still_flags_the_missing_file(self):
        prompt = llm.build_prompt("check nope.py", [], missing_files=["nope.py"])
        assert "nope.py" in prompt

    def test_answer_carries_the_missing_files(self, monkeypatch):
        monkeypatch.setattr(llm, "stream_answer", lambda *a, **k: iter(["no"]))
        answer = Assistant(FakeStore()).ask("check tests/test_function.py")
        assert answer.missing_files == ["tests/test_function.py"]


class TestProjectFingerprint:
    def test_stable_when_nothing_changes(self, sample_project):
        assert project_fingerprint(sample_project) == project_fingerprint(
            sample_project
        )

    def test_changes_when_a_file_is_added(self, sample_project):
        before = project_fingerprint(sample_project)
        (sample_project / "extra.py").write_text("x = 1\n", encoding="utf-8")
        assert project_fingerprint(sample_project) != before

    def test_changes_when_a_file_is_deleted(self, sample_project):
        before = project_fingerprint(sample_project)
        (sample_project / "README.md").unlink()
        assert project_fingerprint(sample_project) != before

    def test_ignored_directories_do_not_affect_it(self, sample_project):
        before = project_fingerprint(sample_project)
        (sample_project / "node_modules" / "more.js").write_text(
            "var y=2;", encoding="utf-8"
        )
        assert project_fingerprint(sample_project) == before

    def test_build_index_records_it(self, sample_project, fake_store):
        from assistant.store import build_index

        build_index(sample_project, fake_store)
        assert fake_store.indexed_fingerprint == project_fingerprint(sample_project)
