"""Choosing a project folder: the in-app browser and the native dialog."""
from __future__ import annotations

from pathlib import Path

from assistant import picker


class TestListSubdirectories:
    def test_lists_child_folders_only(self, sample_project: Path):
        listing = picker.list_subdirectories(sample_project)
        names = {p.name for p in listing.subdirectories}
        assert "src" in names
        assert "README.md" not in names

    def test_hidden_folders_are_off_by_default(self, sample_project: Path):
        names = {p.name for p in
                 picker.list_subdirectories(sample_project).subdirectories}
        assert ".venv" not in names

    def test_hidden_folders_can_be_shown(self, sample_project: Path):
        names = {p.name for p in picker.list_subdirectories(
            sample_project, show_hidden=True).subdirectories}
        assert ".venv" in names

    def test_noise_folders_are_skipped(self, sample_project: Path):
        names = {p.name for p in
                 picker.list_subdirectories(sample_project).subdirectories}
        assert "node_modules" not in names

    def test_sorted_case_insensitively(self, tmp_path: Path):
        for name in ("zebra", "Apple", "mango"):
            (tmp_path / name).mkdir()
        names = [p.name for p in
                 picker.list_subdirectories(tmp_path).subdirectories]
        assert names == ["Apple", "mango", "zebra"]

    def test_parent_is_offered(self, sample_project: Path):
        assert picker.list_subdirectories(sample_project).parent is not None

    def test_missing_folder_reports_instead_of_raising(self, tmp_path: Path):
        listing = picker.list_subdirectories(tmp_path / "nope")
        assert listing.subdirectories == []
        assert "Not a folder" in listing.error

    def test_a_file_is_not_a_folder(self, sample_project: Path):
        listing = picker.list_subdirectories(sample_project / "README.md")
        assert listing.error


class TestDefaultBrowseRoot:
    def test_uses_the_current_folder_when_valid(self, sample_project: Path):
        assert picker.default_browse_root(str(sample_project)) == sample_project

    def test_falls_back_to_the_parent_of_a_partial_path(self, sample_project: Path):
        partial = str(sample_project / "not-created-yet")
        assert picker.default_browse_root(partial) == sample_project

    def test_falls_back_to_home_when_empty(self):
        assert picker.default_browse_root("") == Path.home()


class TestChooseDirectoryDialog:
    def test_returns_none_when_tkinter_is_missing(self, monkeypatch):
        """A headless or tkinter-less session must fall back, not crash."""
        import builtins

        real_import = builtins.__import__

        def blocked(name, *args, **kwargs):
            if name.startswith("tkinter"):
                raise ImportError("no tkinter")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", blocked)
        assert picker.choose_directory_dialog() is None

    def test_returns_none_when_the_dialog_fails(self, monkeypatch):
        import tkinter

        def explode(*args, **kwargs):
            raise RuntimeError("no display")

        monkeypatch.setattr(tkinter, "Tk", explode)
        assert picker.choose_directory_dialog() is None
