"""Sandboxed filesystem operations, confined to a single root folder.

Every path received from the model is interpreted relative to the root and
rejected if it resolves (symlinks included) to a location outside of it.
"""

from __future__ import annotations

import logging
import os
import shutil
from collections.abc import Collection
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NamedTuple

from langchain_core.tools import ToolException

from . import extract

log = logging.getLogger(__name__)

#: Characters of content returned per read_file call. By default a third of this is shown
#: from the beginning, the middle and the end of long text (~2K each); offset/limit pages
#: through the rest, at most this many characters at a time.
DEFAULT_MAX_READ_CHARS = 6_000
#: Documents (PDF/DOCX) larger than this are not parsed, and paging through a text file
#: never reads more than this many bytes of it.
DEFAULT_MAX_PARSE_BYTES = 32 * 1024 * 1024
#: Text extracted from a document at most (what offset/limit can page through).
MAX_EXTRACT_CHARS = 1_000_000
DEFAULT_MAX_LIST_ENTRIES = 500
MAX_DIRECTORY_DEPTH = 3  # how deep create_directory may nest category folders
MAX_PATH_CHARS = 1_024
MAX_RENAME_ATTEMPTS = 1_000  # "name (2).txt" ... "name (1000).txt"


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
        max_parse_bytes: int = DEFAULT_MAX_PARSE_BYTES,
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
        self.max_parse_bytes = max_parse_bytes
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

    def read_file(self, path: str, offset: int | None = None, limit: int | None = None) -> str:
        """Metadata of a file, then (a bounded part of) its content.

        Always first: name, extension, MIME type, size and modification time. Then, by
        format: the text of PDF/DOCX files, the header and first rows of a CSV, the EXIF
        tags of an image, the text of a text file; other binary files get metadata only.

        Without ``offset``/``limit``, text longer than ``max_read_chars`` is shown as its
        beginning, middle and end (a third each), with truncation markers in between. With them, characters
        ``[offset, offset + limit)`` of the text are returned (limit is capped at
        ``max_read_chars``); for a CSV this pages through its raw text.
        """
        if offset is not None and offset < 0:
            raise WorkspaceError("offset must be 0 or more.")
        if limit is not None and limit < 1:
            raise WorkspaceError("limit must be 1 or more.")
        target = self._resolve(path)
        if not target.exists():
            raise WorkspaceError(f"File '{path}' not found.")
        if not target.is_file():
            raise WorkspaceError(f"'{path}' is not a file.")

        stat = target.stat()
        with target.open("rb") as fh:
            head = fh.read(extract.SNIFF_BYTES)
        kind, mime = extract.detect(target, head)
        metadata = "\n".join([
            "[metadata]",
            f"name: {extract.one_line(target.name, 255)}",
            f"extension: {extract.one_line(target.suffix) or '(none)'}",
            f"mime type: {mime}",
            f"size: {stat.st_size} bytes",
            f"modified: {datetime.fromtimestamp(stat.st_mtime, UTC).isoformat(timespec='seconds')}",
        ])
        paging = offset is not None or limit is not None
        page = (offset or 0, min(limit or self.max_read_chars, self.max_read_chars))
        try:
            body = self._content(target, kind, stat.st_size, page if paging else None)
        except extract.ExtractionError as exc:
            body = f"[content: not available - {exc}; classify the file by its metadata]"
        if paging and limit is not None and limit > self.max_read_chars:
            body += f"\n[limit was capped at {self.max_read_chars} characters per read]"
        return f"{metadata}\n{body}"

    # ------------------------------------------------------- read_file content
    def _content(self, target: Path, kind: str, size: int, page: tuple[int, int] | None) -> str:
        if kind == "binary":
            return "[content: none - binary file; classify it by its name, extension and metadata]"
        if kind == "image":
            lines = extract.image_details(target)
            if page:
                lines.append("[offset/limit ignored: an image has no text to page through]")
            return "[content: image metadata]\n" + "\n".join(lines)
        if kind in ("pdf", "docx"):
            if size > self.max_parse_bytes:
                raise extract.ExtractionError(
                    f"the document is larger than {self.max_parse_bytes} bytes, so it was not parsed")
            extractor = extract.pdf_text if kind == "pdf" else extract.docx_text
            text, details, cut = extractor(target, MAX_EXTRACT_CHARS)
            label = f"{kind.upper()} text" + (f" (extraction stopped after {len(text)} characters)" if cut else "")
            prefix = "".join(line + "\n" for line in details)
            if not text.strip():
                return prefix + f"[content: {kind.upper()} has no extractable text (e.g. a scan)]"
            return prefix + self._view(text, page, label)
        if kind == "csv" and page is None:
            return self._csv_preview(target, size)
        return self._text(target, size, page)

    def _view(self, text: str, page: tuple[int, int] | None, label: str) -> str:
        """Default head/tail view or one page of an in-memory text."""
        total = len(text)
        if page is not None:
            offset, limit = page
            if offset > 0 and offset >= total:
                return f"[content: {label}; offset {offset} is past the end ({total} characters)]"
            end = min(offset + limit, total)
            more = f"more follows, next offset={end}" if end < total else "end of text"
            return f"[content: {label}, characters {offset}-{end} of {total}; {more}]\n{text[offset:end]}"
        if total <= self.max_read_chars:
            return f"[content: {label}, {total} characters]\n{text}"
        head, mid, tail = self._thirds()
        m0 = (total - mid) // 2  # the middle sample is centred in the text
        m1, t0 = m0 + mid, total - tail
        return (
            f"[content: {label}, {total} characters; showing characters 0-{head}, {m0}-{m1} "
            f"and {t0}-{total}]\n"
            f"{text[:head]}\n{self._marker(f'characters {head}-{m0}', head)}\n"
            f"{text[m0:m1]}\n{self._marker(f'characters {m1}-{t0}', m1)}\n{text[t0:]}"
        )

    def _thirds(self) -> tuple[int, int, int]:
        """How many characters of the beginning, middle and end the default view shows."""
        part = max(1, self.max_read_chars // 3)
        return max(1, self.max_read_chars - 2 * part), part, part

    @staticmethod
    def _marker(omitted: str, next_offset: int | None) -> str:
        how = (f"Call read_file with offset={next_offset} (and optionally limit) to read them"
               if next_offset is not None else "Page with read_file offset/limit to read them")
        return f"[... truncated: {omitted} omitted. {how} ...]"

    def _text(self, target: Path, size: int, page: tuple[int, int] | None) -> str:
        """A UTF-8 text file. Never reads the whole file: the default view reads at most
        ``max_read_chars * 4`` bytes (UTF-8 is at most 4 bytes per character); paging
        decodes from the start up to ``max_parse_bytes`` bytes."""
        cap = self.max_read_chars * 4
        if page is None and size > cap:  # large file: read the beginning, the middle and the end
            head_chars, mid_chars, tail_chars = self._thirds()
            chunk_start = size // 2 - mid_chars * 2  # a chunk centred in the file
            with target.open("rb") as fh:
                head = fh.read(head_chars * 4).decode("utf-8", errors="replace")[:head_chars]
                fh.seek(chunk_start)
                chunk = fh.read(mid_chars * 4)
                fh.seek(-tail_chars * 4, os.SEEK_END)
                tail = extract.decode_tail(fh.read())[-tail_chars:]
            # Keep the characters at the centre of the chunk (its ends may hold split characters).
            decoded = extract.decode_tail(chunk)
            skip = (len(decoded) - mid_chars) // 2
            mid = decoded[skip:skip + mid_chars]
            mid_start = chunk_start + (len(chunk) - len(decoded[skip:].encode("utf-8")))
            mid_end = mid_start + len(mid.encode("utf-8"))
            # Character offsets of the middle and end are unknown without decoding the whole
            # file, so the markers give byte positions (equal to characters for ASCII text).
            return (
                f"[content: text, {size} bytes; showing the first {len(head)} characters, "
                f"{len(mid)} from about byte {mid_start} and the last {len(tail)}]\n"
                f"{head}\n{self._marker(f'the text from character {len(head)} to about byte {mid_start}', len(head))}\n"
                f"{mid}\n{self._marker(f'the text from about byte {mid_end} to the last part', None)}\n{tail}"
            )
        if page is None or size <= cap:
            with target.open("rb") as fh:
                text = fh.read(cap).decode("utf-8", errors="replace")
            return self._view(text, page, "text")
        offset, limit = page
        text, more, hit_cap = extract.stream_slice(target, offset, limit, self.max_parse_bytes)
        if not text and not more:
            return f"[content: text; offset {offset} is past the end of the file]"
        if hit_cap and len(text) < limit:
            note = f"paging stops after the first {self.max_parse_bytes} bytes of the file"
        else:
            note = f"more follows, next offset={offset + len(text)}" if more else "end of file"
        return f"[content: text, characters {offset}-{offset + len(text)}; {note}]\n{text}"

    def _csv_preview(self, target: Path, size: int) -> str:
        cap = self.max_read_chars * 4
        with target.open("rb") as fh:
            raw = fh.read(cap)
        preview, more = extract.csv_preview(
            raw.decode("utf-8", errors="replace"), complete=len(raw) >= size,
            tsv=target.suffix.lower() == ".tsv")
        if len(preview) > self.max_read_chars:
            preview = preview[: self.max_read_chars] + "\n[... preview truncated]"
        footer = ("\n[... more rows follow; call read_file with offset/limit to page through the raw text]"
                  if more else "")
        return f"[content: CSV header and first {extract.CSV_PREVIEW_ROWS} rows]\n{preview.rstrip()}{footer}"

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
