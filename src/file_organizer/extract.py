"""Format detection and bounded content extraction for ``Workspace.read_file``.

The format is detected from the file's first bytes (magic numbers), not just its name,
so a misleadingly named file is still read correctly. Every extractor is bounded: it
stops after ``max_chars`` characters of text, and the caller refuses to parse documents
larger than its byte cap.

PDF text and image EXIF need optional libraries (``pypdf``, ``pillow``; the
``documents`` extra). Without them, ``MissingDependency`` is raised and the caller
reports metadata only. DOCX is a zip of XML, read with the standard library.
"""

from __future__ import annotations

import codecs
import csv
import io
import mimetypes
import zipfile
from pathlib import Path
from typing import Literal, NamedTuple
from xml.etree import ElementTree

Kind = Literal["pdf", "docx", "image", "csv", "text", "binary"]

SNIFF_BYTES = 2_048
MAX_PDF_PAGES = 500
MAX_DOCX_XML_BYTES = 64 * 1024 * 1024  # uncompressed document.xml read at most (zip bombs)
MAX_META_VALUE_CHARS = 120
MAX_EXIF_TAGS = 40
CSV_PREVIEW_ROWS = 5
MAX_CSV_CELL_CHARS = 200

_DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
_IMAGE_MAGIC: tuple[tuple[bytes, str], ...] = (
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"II*\x00", "image/tiff"),
    (b"MM\x00*", "image/tiff"),
)
_FALLBACK_MIME = {"pdf": "application/pdf", "docx": _DOCX_MIME, "csv": "text/csv",
                  "text": "text/plain", "binary": "application/octet-stream"}
# A fresh MimeTypes() uses only Python's built-in table, not the OS's (same result everywhere).
_MIME_TYPES = mimetypes.MimeTypes()
_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


class ExtractionError(Exception):
    """The content could not be extracted (corrupt, encrypted, ...); metadata still works."""


class MissingDependency(ExtractionError):
    """The optional library needed for this format is not installed."""


class Detected(NamedTuple):
    kind: Kind
    mime: str


def detect(path: Path, head: bytes) -> Detected:
    """Classify a file from its first bytes (``head``), falling back to its extension."""
    by_ext = _MIME_TYPES.guess_type(path.name, strict=False)[0]
    if head.startswith(b"%PDF-"):
        return Detected("pdf", "application/pdf")
    if head.startswith(b"PK\x03\x04") and _is_docx(path):
        return Detected("docx", _DOCX_MIME)
    for magic, mime in _IMAGE_MAGIC:
        if head.startswith(magic):
            return Detected("image", mime)
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return Detected("image", "image/webp")
    if b"\x00" in head:
        return Detected("binary", by_ext or _FALLBACK_MIME["binary"])
    if path.suffix.lower() in (".csv", ".tsv"):
        return Detected("csv", by_ext or _FALLBACK_MIME["csv"])
    return Detected("text", by_ext if by_ext and _is_textual(by_ext) else _FALLBACK_MIME["text"])


def _is_textual(mime: str) -> bool:
    return mime.startswith("text/") or mime in ("application/json", "application/xml", "message/rfc822",
                                                "application/javascript", "application/x-sh")


def _is_docx(path: Path) -> bool:
    try:
        with zipfile.ZipFile(path) as zf:  # reads only the central directory
            return "word/document.xml" in zf.namelist()
    except (zipfile.BadZipFile, OSError, ValueError):
        return False


def one_line(value: object, max_chars: int = MAX_META_VALUE_CHARS) -> str:
    """Render an untrusted metadata value (file name, EXIF, PDF title) on one bounded line."""
    text = "".join(ch if ch.isprintable() else " " for ch in str(value)).strip()
    return text if len(text) <= max_chars else text[: max_chars - 1] + "…"


# ------------------------------------------------------------------------ text
def decode_tail(raw: bytes) -> str:
    """Decode bytes read from the middle of a UTF-8 file: skip a split leading character."""
    start = 0
    while start < min(len(raw), 3) and 0x80 <= raw[start] <= 0xBF:  # continuation bytes
        start += 1
    return raw[start:].decode("utf-8", errors="replace")


def stream_slice(path: Path, offset: int, limit: int, max_bytes: int) -> tuple[str, bool, bool]:
    """Characters ``[offset, offset + limit)`` of a UTF-8 text file, decoding as it goes.

    Reads at most ``max_bytes`` bytes. Returns (text, more_follows, hit_byte_cap).
    """
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    skipped, parts, collected, read = 0, [], 0, 0
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(min(64 * 1024, max_bytes - read))
            read += len(chunk)
            text = decoder.decode(chunk, final=not chunk)
            if skipped < offset:
                drop = min(offset - skipped, len(text))
                skipped += drop
                text = text[drop:]
            if text and skipped >= offset:
                parts.append(text[: limit - collected])
                collected += len(parts[-1])
                if collected >= limit:
                    more = len(text) > len(parts[-1]) or bool(fh.read(1))
                    return "".join(parts), more, False
            if not chunk:
                return "".join(parts), False, False
            if read >= max_bytes:
                more = bool(fh.read(1))
                return "".join(parts), more, more


def csv_preview(sample: str, *, complete: bool, tsv: bool, rows: int = CSV_PREVIEW_ROWS) -> tuple[str, bool]:
    """Header plus the first ``rows`` rows of CSV text, re-serialized with bounded cells.

    ``complete``: ``sample`` is the whole file (otherwise its last line may be cut off and
    is not shown). Returns (preview, more_rows_follow).
    """
    lines = sample.splitlines(keepends=True)
    if not complete and len(lines) > 1:
        lines = lines[:-1]
    try:
        dialect = csv.Sniffer().sniff("".join(lines[:20]), delimiters="\t" if tsv else ",;\t|")
        delimiter = dialect.delimiter
    except csv.Error:
        delimiter = "\t" if tsv else ","
    out = io.StringIO()
    writer = csv.writer(out, delimiter=delimiter, lineterminator="\n")
    reader = csv.reader(lines, delimiter=delimiter)
    shown = 0
    try:
        for row in reader:
            if shown > rows:  # header + `rows` rows already written
                return out.getvalue(), True
            writer.writerow([one_line(cell, MAX_CSV_CELL_CHARS) for cell in row])
            shown += 1
    except csv.Error:  # malformed: show the raw lines instead
        return "".join(lines[: rows + 1]), len(lines) > rows + 1 or not complete
    return out.getvalue(), not complete


# -------------------------------------------------------------------- documents
def pdf_text(path: Path, max_chars: int) -> tuple[str, list[str], bool]:
    """(text, detail lines, truncated) of a PDF; stops after ``max_chars`` characters."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise MissingDependency("PDF text extraction needs pypdf (install the 'documents' extra)") from exc
    try:
        reader = PdfReader(path)
        if reader.is_encrypted and not reader.decrypt(""):
            raise ExtractionError("the PDF is password-protected")
        details = [f"pages: {len(reader.pages)}"]
        info = reader.metadata or {}
        for key, label in (("/Title", "title"), ("/Author", "author"), ("/Subject", "subject")):
            if info.get(key):
                details.append(f"{label}: {one_line(info[key])}")
        parts, total = [], 0
        for i, page in enumerate(reader.pages):
            if total >= max_chars or i >= MAX_PDF_PAGES:
                return "".join(parts)[:max_chars], details, True
            text = (page.extract_text() or "").strip()
            if text:
                parts.append(text + "\n\n")
                total += len(parts[-1])
        text = "".join(parts).rstrip()
        return text[:max_chars], details, len(text) > max_chars
    except ExtractionError:
        raise
    except Exception as exc:  # pypdf raises many types on malformed input
        raise ExtractionError(f"could not parse the PDF ({type(exc).__name__})") from exc


def docx_text(path: Path, max_chars: int) -> tuple[str, list[str], bool]:
    """(text, detail lines, truncated) of a .docx, parsed incrementally with bounded reads."""
    parts: list[str] = []
    total = 0
    try:
        with zipfile.ZipFile(path) as zf, zf.open("word/document.xml") as fh:
            parser = ElementTree.XMLPullParser(events=("end",))
            read = 0
            while chunk := fh.read(64 * 1024):
                if read == 0 and b"<!DOCTYPE" in chunk:
                    raise ExtractionError("the document declares a DTD, which is not allowed")
                read += len(chunk)
                parser.feed(chunk)
                for _, elem in parser.read_events():
                    if elem.tag == f"{_W}t" and elem.text:
                        parts.append(elem.text)
                    elif elem.tag == f"{_W}tab":
                        parts.append("\t")
                    elif elem.tag in (f"{_W}br", f"{_W}cr"):
                        parts.append("\n")
                    elif elem.tag == f"{_W}p":
                        parts.append("\n")
                        elem.clear()
                    else:
                        continue
                    total += len(parts[-1])
                if total > max_chars or read >= MAX_DOCX_XML_BYTES:
                    return "".join(parts)[:max_chars], [], True
    except ExtractionError:
        raise
    except (zipfile.BadZipFile, KeyError, ElementTree.ParseError, OSError, RuntimeError, ValueError) as exc:
        raise ExtractionError(f"could not parse the DOCX ({type(exc).__name__})") from exc
    return "".join(parts).strip(), [], False


def image_details(path: Path) -> list[str]:
    """Format, dimensions and EXIF tags of an image (header only; pixels are not decoded)."""
    try:
        from PIL import ExifTags, Image
    except ImportError as exc:
        raise MissingDependency("reading image EXIF needs pillow (install the 'documents' extra)") from exc
    try:
        with Image.open(path) as im:
            lines = [f"image: {im.format}, {im.width}x{im.height} pixels"]
            exif = im.getexif()
            tags: dict[str, object] = {ExifTags.TAGS.get(k, f"tag {k}"): v for k, v in exif.items()}
            for ifd_id in (ExifTags.IFD.Exif, ExifTags.IFD.GPSInfo):
                names = ExifTags.GPSTAGS if ifd_id == ExifTags.IFD.GPSInfo else ExifTags.TAGS
                for k, v in exif.get_ifd(ifd_id).items():
                    tags[names.get(k, f"tag {k}")] = v
    except Exception as exc:  # corrupt files, decompression-bomb guards, unknown formats
        raise ExtractionError(f"could not read the image metadata ({type(exc).__name__})") from exc
    shown = 0
    for name, value in tags.items():
        if name in ("ExifOffset", "GPSInfo", "MakerNote") or isinstance(value, bytes):
            continue  # pointers and opaque binary blobs
        if shown == MAX_EXIF_TAGS:
            lines.append("[... more EXIF tags not shown]")
            break
        lines.append(f"exif {one_line(name, 40)}: {one_line(value)}")
        shown += 1
    if shown == 0:
        lines.append("exif: none")
    return lines
