"""Phase 3a - choosing a project folder.

Two ways, because neither alone is enough:

- `choose_directory_dialog` opens the real Windows folder picker. The
  Streamlit server *is* the user's machine here, so a native tkinter dialog
  is legitimate and is what the brief's "select the project directory"
  asks for. It can fail (no display, tkinter absent, a headless run), so it
  never raises - it returns None and the caller falls back.
- `list_subdirectories` powers an in-app browser that works everywhere,
  including over a network, and needs no native toolkit at all.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Folders nobody wants to browse into while picking a project.
HIDDEN_PREFIXES = (".",)
SKIP_DIRS = {
    "__pycache__", "node_modules", "site-packages", "System Volume Information",
    "$RECYCLE.BIN", "AppData",
}


class DialogUnavailable(RuntimeError):
    """Raised internally when no native folder dialog can be opened."""


def choose_directory_dialog(initial: str | None = None) -> str | None:
    """Open the OS folder picker and return the chosen path.

    Returns None if the user cancels, and None if no dialog can be shown -
    the caller cannot tell the difference and does not need to, because a
    cancelled dialog and an unavailable one both mean "keep what you had".
    """
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:  # pragma: no cover - tkinter missing from the build
        return None

    root = None
    try:
        root = tk.Tk()
        root.withdraw()
        # Without this the dialog opens behind the browser window and looks
        # like nothing happened.
        root.attributes("-topmost", True)
        root.update()
        chosen = filedialog.askdirectory(
            title="Select the project folder to index",
            initialdir=initial or str(Path.home()),
            mustexist=True,
        )
        return chosen or None
    except Exception:  # pragma: no cover - depends on the desktop session
        return None
    finally:
        if root is not None:
            try:
                root.destroy()
            except Exception:  # pragma: no cover
                pass


@dataclass
class DirectoryListing:
    """One level of the in-app folder browser."""

    current: Path
    parent: Path | None
    subdirectories: list[Path]
    error: str = ""


def list_subdirectories(
    path: Path | str,
    show_hidden: bool = False,
) -> DirectoryListing:
    """List the folders inside `path`, for the in-app browser.

    Never raises: an unreadable folder comes back as an empty listing with
    `error` set, so the browser can say why instead of crashing the app.
    """
    current = Path(path).expanduser()
    parent = current.parent if current.parent != current else None

    if not current.is_dir():
        return DirectoryListing(
            current=current, parent=parent, subdirectories=[],
            error=f"Not a folder: {current}",
        )

    subdirs: list[Path] = []
    try:
        for entry in current.iterdir():
            if not entry.is_dir():
                continue
            name = entry.name
            if name in SKIP_DIRS:
                continue
            if not show_hidden and name.startswith(HIDDEN_PREFIXES):
                continue
            subdirs.append(entry)
    except PermissionError:
        return DirectoryListing(
            current=current, parent=parent, subdirectories=[],
            error="No permission to read this folder.",
        )
    except OSError as exc:
        return DirectoryListing(
            current=current, parent=parent, subdirectories=[], error=str(exc)
        )

    subdirs.sort(key=lambda p: p.name.lower())
    return DirectoryListing(current=current, parent=parent, subdirectories=subdirs)


def default_browse_root(current: str = "") -> Path:
    """Where the in-app browser should open."""
    if current:
        candidate = Path(current).expanduser()
        if candidate.is_dir():
            return candidate
        if candidate.parent.is_dir():
            return candidate.parent
    return Path.home()
