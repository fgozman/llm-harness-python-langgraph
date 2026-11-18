"""Sandboxed filesystem operations, confined to a single root folder.

Every path received from the model is interpreted relative to the root and
rejected if it resolves (symlinks included) to a location outside of it.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from langchain_core.tools import ToolException

log = logging.getLogger(__name__)

DEFAULT_MAX_READ_CHARS = 4_000
DEFAULT_MAX_LIST_ENTRIES = 500
MAX_DIRECTORY_DEPTH = 3  # how deep create_directory may nest category folders
MAX_PATH_CHARS = 1_024
_BINARY_SNIFF_BYTES = 1_024


class WorkspaceError(ToolException):
    """A filesystem operation was invalid; the message is safe to show the model.

    It subclasses ToolException so LangChain tools report it to the model as an error
    result instead of raising.
    """


class Workspace:
    def __init__(
        self,
        root: str | Path,
        *,
        max_read_chars: int = DEFAULT_MAX_READ_CHARS,
        max_list_entries: int = DEFAULT_MAX_LIST_ENTRIES,
    ):
        root = Path(root).expanduser()
        if not root.exists():
            raise WorkspaceError(f"Folder '{root}' does not exist.")
        if not root.is_dir():
            raise WorkspaceError(f"'{root}' is not a directory.")
        self.root = root.resolve()
        self.max_read_chars = max_read_chars
        self.max_list_entries = max_list_entries

    # ------------------------------------------------------------------ helpers
    def _resolve(self, path: str) -> Path:
        """Map a model-supplied path to an absolute path inside the root.

        Rejected: paths outside the root, hidden entries (dotfiles such as .env, .git or
        the agent's own state/log files), and symlinks (acting on a link would act on its
        target, which may be elsewhere).
        """
        if not isinstance(path, str) or len(path) > MAX_PATH_CHARS or "\x00" in path:
            raise WorkspaceError("Invalid path.")
        try:
            candidate = (self.root / path).resolve()
        except (OSError, ValueError, RuntimeError) as exc:  # e.g. symlink loops
            raise WorkspaceError(f"Invalid path '{path}'.") from exc
        if candidate != self.root and self.root not in candidate.parents:
            log.warning("Blocked path outside the workspace: %r", path)
            raise WorkspaceError(f"Path '{path}' is outside the folder being organized.")
        if any(part.startswith(".") for part in candidate.relative_to(self.root).parts):
            log.warning("Blocked hidden path: %r", path)
            raise WorkspaceError(f"Path '{path}' is hidden; hidden files are off-limits.")
        # Walk the unresolved path: any symlink along it is refused.
        current = self.root
        for part in Path(path).parts:
            if part in (".", "/") or part == Path(path).anchor:
                continue
            current = current.parent if part == ".." else current / part
            if current.is_symlink():
                log.warning("Blocked symlink: %r", path)
                raise WorkspaceError(f"Path '{path}' goes through a symlink; symlinks are off-limits.")
        return candidate

    def _rel(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix() or "."

    # ------------------------------------------------------------------- tools
    def list_files(self, directory: str = ".") -> list[str]:
        target = self._resolve(directory)
        if not target.is_dir():
            raise WorkspaceError(f"Directory '{directory}' not found.")
        entries = sorted(
            p for p in target.iterdir() if not p.name.startswith(".") and not p.is_symlink()
        )
        listing = [self._rel(p) + ("/" if p.is_dir() else "") for p in entries[: self.max_list_entries]]
        if len(entries) > self.max_list_entries:
            listing.append(f"[... {len(entries) - self.max_list_entries} more entries not shown]")
        return listing

    def read_file(self, path: str) -> str:
        target = self._resolve(path)
        if not target.exists():
            raise WorkspaceError(f"File '{path}' not found.")
        if not target.is_file():
            raise WorkspaceError(f"'{path}' is not a file.")

        with target.open("rb") as fh:
            head = fh.read(_BINARY_SNIFF_BYTES)
        if b"\x00" in head:
            size = target.stat().st_size
            return (
                f"[binary file '{self._rel(target)}', {size} bytes - content not shown; "
                "classify it by its name and extension]"
            )

        # Read a bounded prefix only (UTF-8 is at most 4 bytes per char), never the whole file.
        size = target.stat().st_size
        with target.open("rb") as fh:
            raw = fh.read(self.max_read_chars * 4)
        text = raw.decode("utf-8", errors="replace")
        if len(text) > self.max_read_chars or len(raw) < size:
            text = text[: self.max_read_chars] + f"\n[... truncated, file is {size} bytes]"
        return text

    def create_directory(self, path: str) -> str:
        target = self._resolve(path)
        if target == self.root:
            raise WorkspaceError("Refusing to create the root folder itself.")
        depth = len(target.relative_to(self.root).parts)
        if depth > MAX_DIRECTORY_DEPTH:
            raise WorkspaceError(
                f"'{path}' is nested {depth} levels deep; the limit is {MAX_DIRECTORY_DEPTH}."
            )
        if target.exists():
            if not target.is_dir():
                raise WorkspaceError(f"'{path}' exists and is not a directory.")
            return f"Directory '{self._rel(target)}' already exists."
        target.mkdir(parents=True)
        log.info("Created directory %s", target)
        return f"Created directory '{self._rel(target)}'."

    def move_file(self, source: str, destination: str) -> str:
        """Move a file into an existing directory, keeping its name (files are never renamed)."""
        src = self._resolve(source)
        target_dir = self._resolve(destination)
        if not src.exists():
            raise WorkspaceError(f"File '{source}' not found.")
        if not src.is_file():
            raise WorkspaceError(f"'{source}' is not a file.")
        if not target_dir.exists():
            raise WorkspaceError(f"Destination directory '{destination}' does not exist; create it first.")
        if not target_dir.is_dir():
            raise WorkspaceError(
                f"Destination '{destination}' is not a directory; files can only be moved into "
                "a directory and are never renamed."
            )

        dst = target_dir / src.name
        if dst.exists():
            raise WorkspaceError(f"Destination '{self._rel(dst)}' already exists; refusing to overwrite.")

        shutil.move(src, dst)
        log.info("Moved %s -> %s", src, dst)
        return f"Moved '{self._rel(src)}' to '{self._rel(dst)}'."
