"""Sandboxed filesystem operations, confined to a single root folder.

Every path received from the model is interpreted relative to the root and
rejected if it resolves (symlinks included) to a location outside of it.
"""

from __future__ import annotations

import logging
import shutil
from collections.abc import Collection
from pathlib import Path
from typing import Any, NamedTuple

from langchain_core.tools import ToolException

log = logging.getLogger(__name__)

DEFAULT_MAX_READ_CHARS = 4_000
DEFAULT_MAX_LIST_ENTRIES = 500
MAX_DIRECTORY_DEPTH = 3  # how deep create_directory may nest category folders
MAX_PATH_CHARS = 1_024
MAX_RENAME_ATTEMPTS = 1_000  # "name (2).txt" ... "name (1000).txt"
_BINARY_SNIFF_BYTES = 1_024


class Change(NamedTuple):
    """The result of a filesystem change: the message for the model, and how to undo it.

    ``undo`` is JSON-serializable (it is kept in the decision log) or None when nothing
    changed on disk (e.g. the directory already existed):
    * ``{"op": "move", "from": "a.txt", "to": "finance/a.txt"}``
    * ``{"op": "mkdir", "paths": ["docs", "docs/finance"]}`` (every directory created,
      parents first)
    """

    message: str
    undo: dict[str, Any] | None


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
        rename_on_clash: bool = False,
    ):
        """``rename_on_clash``: when a file with the same name is already in the destination
        directory, move the file there as "name (2).ext" (the first free number) instead
        of failing. Off by default; existing files are never overwritten either way."""
        root = Path(root).expanduser()
        if not root.exists():
            raise WorkspaceError(f"Folder '{root}' does not exist.")
        if not root.is_dir():
            raise WorkspaceError(f"'{root}' is not a directory.")
        self.root = root.resolve()
        self.max_read_chars = max_read_chars
        self.max_list_entries = max_list_entries
        self.rename_on_clash = rename_on_clash

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

    def create_directory_change(self, path: str) -> Change:
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
            return Change(f"Directory '{self._rel(target)}' already exists.", None)
        missing = [p for p in (target, *target.parents) if p != self.root and self.root in p.parents
                   and not p.exists()]
        target.mkdir(parents=True)
        log.info("Created directory %s", target)
        created = [self._rel(p) for p in reversed(missing)]  # parents first
        return Change(f"Created directory '{self._rel(target)}'.", {"op": "mkdir", "paths": created})

    def _move_target(self, source: str, destination: str,
                     claimed: Collection[str] = ()) -> tuple[Path, Path]:
        """Validate a move (read-only) and return (source file, destination file path).
        ``claimed``: relative paths to treat as taken although nothing is there yet."""
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
        if src.parent == target_dir:
            raise WorkspaceError(f"'{self._rel(src)}' is already in '{self._rel(target_dir)}'.")

        def taken(path: Path) -> bool:
            return path.exists() or path.is_symlink() or self._rel(path) in claimed

        dst = target_dir / src.name
        if not taken(dst):
            return src, dst
        if not self.rename_on_clash:
            raise WorkspaceError(f"Destination '{self._rel(dst)}' already exists; refusing to overwrite.")
        for n in range(2, MAX_RENAME_ATTEMPTS + 1):
            candidate = target_dir / f"{src.stem} ({n}){src.suffix}"
            if not taken(candidate):
                return src, candidate
        raise WorkspaceError(f"Too many files named like '{src.name}' in '{self._rel(target_dir)}'.")

    def planned_destination(self, source: str, destination: str,
                            claimed: Collection[str] = ()) -> str | None:
        """Where ``move_file_change(source, destination)`` would put the file now (relative
        path), or None if the move would fail. ``claimed`` are destinations of earlier moves
        of the same batch, not made yet. Changes nothing."""
        try:
            return self._rel(self._move_target(source, destination, claimed)[1])
        except WorkspaceError:
            return None

    def move_file_change(self, source: str, destination: str) -> Change:
        """Move a file into an existing directory, keeping its name (unless renaming on a
        name clash is enabled). Never overwrites."""
        src, dst = self._move_target(source, destination)
        src_rel = self._rel(src)
        shutil.move(src, dst)
        log.info("Moved %s -> %s", src, dst)
        message = f"Moved '{src_rel}' to '{self._rel(dst)}'."
        if dst.name != src.name:
            message += f" A file named '{src.name}' was already there, so it was renamed to '{dst.name}'."
        return Change(message, {"op": "move", "from": src_rel, "to": self._rel(dst)})

    # -------------------------------------------------------------------- undo
    def undo_move(self, moved_to: str, original: str) -> str:
        """Move a file back to where it was before ``move_file``. Never overwrites."""
        current = self._resolve(moved_to)
        back = self._resolve(original)
        if not current.is_file():
            raise WorkspaceError(f"'{moved_to}' is no longer there.")
        if back.exists() or back.is_symlink():
            raise WorkspaceError(f"'{original}' exists again; refusing to overwrite it.")
        if not back.parent.is_dir():
            raise WorkspaceError(f"The original directory of '{original}' no longer exists.")
        shutil.move(current, back)
        log.info("Undo: moved %s back to %s", current, back)
        return f"Moved '{self._rel(current)}' back to '{self._rel(back)}'."

    def remove_empty_directory(self, path: str) -> str:
        """Remove a directory, but only if it is empty (hidden files count)."""
        target = self._resolve(path)
        if target == self.root:
            raise WorkspaceError("Refusing to remove the root folder itself.")
        if not target.exists():  # already gone (e.g. a partly finished earlier undo)
            return f"Directory '{self._rel(target)}' is already gone."
        if not target.is_dir():
            raise WorkspaceError(f"'{path}' is not a directory.")
        if any(target.iterdir()):
            raise WorkspaceError(f"Directory '{path}' is not empty; kept it.")
        target.rmdir()
        log.info("Undo: removed directory %s", target)
        return f"Removed directory '{self._rel(target)}'."
