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
import contextlib
import csv
import io
import mimetypes
import re
import struct
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal, NamedTuple
from xml.etree import ElementTree

Kind = Literal["pdf", "docx", "xlsx", "image", "csv", "text", "ole", "binary"]

SNIFF_BYTES = 2_048
MAX_PDF_PAGES = 500
MAX_DOCX_XML_BYTES = 64 * 1024 * 1024  # uncompressed XML read at most per zip member (zip bombs)
MAX_XLSX_SHEETS = 10  # sheets previewed
MAX_XLSX_COLUMNS = 50  # columns shown per row
MAX_META_VALUE_CHARS = 120
MAX_EXIF_TAGS = 40
CSV_PREVIEW_ROWS = 5
MAX_CSV_CELL_CHARS = 200

_DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
_XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
_OLE_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"  # legacy Office: .doc, .xls, .ppt, .msg
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
for _ext, _mime in ((".docx", _DOCX_MIME), (".xlsx", _XLSX_MIME),
                    (".pptx", "application/vnd.openxmlformats-officedocument.presentationml.presentation"),
                    (".odt", "application/vnd.oasis.opendocument.text"),
                    (".ods", "application/vnd.oasis.opendocument.spreadsheet"),
                    (".odp", "application/vnd.oasis.opendocument.presentation")):
    _MIME_TYPES.add_type(_mime, _ext)  # missing from Python's built-in table
_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_S = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_REL_ID = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
_PKG_REL = "{http://schemas.openxmlformats.org/package/2006/relationships}Relationship"


class ExtractionError(Exception):
    """The content could not be extracted (corrupt, encrypted, ...); metadata still works."""


class MissingDependency(ExtractionError):
    """The optional library needed for this format is not installed."""


class Detected(NamedTuple):
    kind: Kind
    mime: str


class Extracted(NamedTuple):
    """Text extracted from a document, with lines of document details shown before it."""

    text: str
    details: list[str]
    truncated: bool  # extraction stopped at max_chars
    label: str  # what the text is, e.g. "PDF text"
    empty: str = "it has no extractable text"  # shown when there is no text


def detect(path: Path, head: bytes) -> Detected:
    """Classify a file from its first bytes (``head``), falling back to its extension."""
    by_ext = _MIME_TYPES.guess_type(path.name, strict=False)[0]
    if head.startswith(b"%PDF-"):
        return Detected("pdf", "application/pdf")
    if head.startswith(b"PK\x03\x04"):
        office = _office_zip_kind(path)
        if office == "docx":
            return Detected("docx", _DOCX_MIME)
        if office == "xlsx":
            return Detected("xlsx", _XLSX_MIME)
    if head.startswith(_OLE_MAGIC):
        return Detected("ole", by_ext or "application/x-ole-storage")
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


def _office_zip_kind(path: Path) -> Literal["docx", "xlsx"] | None:
    try:
        with zipfile.ZipFile(path) as zf:  # reads only the central directory
            names = set(zf.namelist())
    except (zipfile.BadZipFile, OSError, ValueError):
        return None
    if "word/document.xml" in names:
        return "docx"
    if "xl/workbook.xml" in names:
        return "xlsx"
    return None


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
def pdf_text(path: Path, max_chars: int) -> Extracted:
    """Text and details of a PDF; stops after ``max_chars`` characters."""
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
                return Extracted("".join(parts)[:max_chars], details, True, "PDF text")
            text = (page.extract_text() or "").strip()
            if text:
                parts.append(text + "\n\n")
                total += len(parts[-1])
        text = "".join(parts).rstrip()
        return Extracted(text[:max_chars], details, len(text) > max_chars, "PDF text",
                         empty="it has no extractable text (e.g. a scanned PDF)")
    except ExtractionError:
        raise
    except Exception as exc:  # pypdf raises many types on malformed input
        raise ExtractionError(f"could not parse the PDF ({type(exc).__name__})") from exc


def docx_text(path: Path, max_chars: int) -> Extracted:
    """Text of a .docx, parsed incrementally with bounded reads."""
    parts: list[str] = []
    total = 0
    try:
        with zipfile.ZipFile(path) as zf, zf.open("word/document.xml") as fh:
            for elem in _xml_elements(fh):
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
                if total > max_chars:
                    return Extracted("".join(parts)[:max_chars], [], True, "DOCX text")
    except _XmlPartTooLarge:
        return Extracted("".join(parts)[:max_chars], [], True, "DOCX text")
    except ExtractionError:
        raise
    except (zipfile.BadZipFile, KeyError, ElementTree.ParseError, OSError, RuntimeError, ValueError) as exc:
        raise ExtractionError(f"could not parse the DOCX ({type(exc).__name__})") from exc
    return Extracted("".join(parts).strip(), [], False, "DOCX text")


class _XmlPartTooLarge(ExtractionError):
    """A zipped XML part is longer than MAX_DOCX_XML_BYTES uncompressed."""


def _xml_elements(fh):
    """Yield the elements of a zipped XML part as they end; DTDs are refused, and
    _XmlPartTooLarge is raised after MAX_DOCX_XML_BYTES (zip bombs)."""
    parser = ElementTree.XMLPullParser(events=("end",))
    read = 0
    while chunk := fh.read(64 * 1024):
        if read == 0 and b"<!DOCTYPE" in chunk:
            raise ExtractionError("the document declares a DTD, which is not allowed")
        read += len(chunk)
        parser.feed(chunk)
        for _, elem in parser.read_events():
            yield elem
        if read >= MAX_DOCX_XML_BYTES:
            raise _XmlPartTooLarge(f"an XML part is larger than {MAX_DOCX_XML_BYTES} bytes uncompressed")
    parser.close()  # raises ParseError if the XML is truncated
    for _, elem in parser.read_events():
        yield elem


class _Shared(int):
    """A cell holding an index into the workbook's shared strings."""


def _column(ref: str | None, default: int) -> int:
    """0-based column of a cell reference like "AB12"."""
    letters = "".join(ch for ch in (ref or "") if ch.isalpha())
    if not letters:
        return default
    index = 0
    for ch in letters.upper():
        index = index * 26 + ord(ch) - ord("A") + 1
    return index - 1


# Built-in number formats that show a date and/or time (ECMA-376 18.8.30, plus CJK locales).
_DATE_FORMAT_IDS = frozenset([*range(14, 23), *range(27, 37), *range(45, 48), *range(50, 59)])


def _date_format_kind(code: str) -> Literal["date", "elapsed"] | None:
    """Whether a custom number format shows a date/time ("dd/mm/yyyy", "hh:mm") or an
    elapsed duration ("[h]:mm:ss"); None for other formats."""
    plain, i, elapsed = [], 0, False
    while i < len(code):  # drop quoted text, escaped characters and [colors]/[conditions]
        ch = code[i]
        if ch == '"':
            end = code.find('"', i + 1)
            i = end + 1 if end != -1 else len(code)
        elif ch in "\\_*":
            i += 2
        elif ch == "[":
            end = code.find("]", i)
            if end == -1:
                break
            if code[i + 1:end].lower() in ("h", "hh", "m", "mm", "s", "ss"):  # elapsed time
                elapsed = True
            i = end + 1
        else:
            plain.append(ch.lower())
            i += 1
    text = "".join(plain)
    if elapsed:
        return "elapsed"
    return "date" if text != "general" and any(ch in text for ch in "dmyhs") else None


def _xlsx_date_styles(zf: zipfile.ZipFile, names: set[str]) -> dict[int, str]:
    """Cell styles (``<c s="...">`` indexes) whose number format is a date/time ("date")
    or a duration ("elapsed")."""
    if "xl/styles.xml" not in names:
        return {}
    custom: dict[int, str] = {}
    with zf.open("xl/styles.xml") as fh:
        for elem in _xml_elements(fh):
            if elem.tag == f"{_S}numFmt" and (elem.get("numFmtId") or "").isdigit():
                custom[int(elem.get("numFmtId"))] = elem.get("formatCode", "")
            elif elem.tag == f"{_S}cellXfs":  # (cellStyleXfs also holds <xf>, not used by cells)
                styles: dict[int, str] = {}
                for index, xf in enumerate(elem.findall(f"{_S}xf")):
                    fmt_id = xf.get("numFmtId", "0")
                    fmt = int(fmt_id) if fmt_id.isdigit() else 0
                    if fmt in custom:
                        kind = _date_format_kind(custom[fmt])
                    else:
                        kind = "elapsed" if fmt == 46 else "date" if fmt in _DATE_FORMAT_IDS else None
                    if kind:
                        styles[index] = kind
                return styles
    return {}


def _excel_date(serial: str, date1904: bool, kind: str = "date") -> str | None:
    """An Excel date serial number (or duration in days) as text, or None if it isn't one."""
    try:
        days = float(serial)
        if kind == "elapsed":
            seconds = round(abs(days) * 86_400)
            sign = "-" if days < 0 else ""
            return f"{sign}{seconds // 3600}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"
        if date1904:
            epoch = datetime(1904, 1, 1)
        else:  # Excel counts a nonexistent 1900-02-29 (serial 60), so earlier serials shift by one
            epoch = datetime(1899, 12, 31) if days < 60 else datetime(1899, 12, 30)
        # Rounding to whole seconds can also overflow past datetime.max.
        return _format_excel_moment(epoch + timedelta(days=days), days)
    except (ValueError, OverflowError):
        return None


def _format_excel_moment(moment: datetime, days: float) -> str:
    """A spreadsheet date as Excel shows it by default, for XLSX and XLS alike: a time
    of day for serials below 1, a date for whole serials, else both (whole seconds)."""
    moment = moment.replace(microsecond=0) + timedelta(seconds=round(moment.microsecond / 1e6))
    if 0 <= days < 1:  # a time of day without a date
        return moment.strftime("%H:%M:%S")
    return moment.strftime("%Y-%m-%d") if days == int(days) else moment.strftime("%Y-%m-%d %H:%M:%S")


def _xlsx_sheets(zf: zipfile.ZipFile, names: set[str]) -> tuple[list[tuple[str, str]], bool]:
    """(sheet name, zip member) of every worksheet in workbook order, and whether the
    workbook uses the 1904 date system."""
    targets: dict[str, str] = {}
    if "xl/_rels/workbook.xml.rels" in names:
        with zf.open("xl/_rels/workbook.xml.rels") as fh:
            for elem in _xml_elements(fh):
                if elem.tag == _PKG_REL and elem.get("Id") and elem.get("Target"):
                    target = elem.get("Target")
                    targets[elem.get("Id")] = target.lstrip("/") if target.startswith("/") else f"xl/{target}"
    sheets, date1904 = [], False
    with zf.open("xl/workbook.xml") as fh:
        for elem in _xml_elements(fh):
            if elem.tag == f"{_S}workbookPr":
                date1904 = elem.get("date1904", "").lower() in ("1", "true")
            elif elem.tag == f"{_S}sheet":
                member = targets.get(elem.get(_REL_ID, ""))
                if member in names:
                    sheets.append((elem.get("name", "?"), member))
    return sheets, date1904


def _xlsx_rows(zf: zipfile.ZipFile, member: str, count: int, date_styles: dict[int, str] | None = None,
               date1904: bool = False) -> list[list[object]]:
    """The first ``count`` non-empty rows of a worksheet (cells by column position)."""
    rows: list[list[object]] = []
    with zf.open(member) as fh:
        for elem in _xml_elements(fh):
            if elem.tag != f"{_S}row":
                continue
            row: list[object] = []
            for cell in elem.iter(f"{_S}c"):
                col = _column(cell.get("r"), len(row))
                if col >= MAX_XLSX_COLUMNS:
                    continue
                kind, value = cell.get("t"), cell.findtext(f"{_S}v")
                if kind == "s" and value is not None and value.strip().isdigit():
                    value = _Shared(int(value))
                elif kind == "inlineStr":
                    value = "".join(t.text or "" for t in cell.iter(f"{_S}t"))
                elif kind == "b":
                    value = {"1": "TRUE", "0": "FALSE"}.get(value or "", value)
                elif (kind in (None, "n") and value and date_styles and cell.get("s", "").isdigit()
                      and int(cell.get("s")) in date_styles):
                    value = _excel_date(value, date1904, date_styles[int(cell.get("s"))]) or value
                row.extend([""] * (col - len(row)))
                row[col:col + 1] = ["" if value is None else value]
            elem.clear()
            if any(v != "" for v in row):
                rows.append(row)
                if len(rows) == count:
                    break
    return rows


def _xlsx_shared_strings(zf: zipfile.ZipFile, count: int) -> list[str]:
    """The first ``count`` shared strings (only as many as the previewed cells need)."""
    strings: list[str] = []
    with zf.open("xl/sharedStrings.xml") as fh:
        for elem in _xml_elements(fh):
            if elem.tag != f"{_S}si":
                continue
            # Text runs, skipping phonetic hints (<rPh>), which repeat the text in kana.
            runs = [elem.find(f"{_S}t")] + [r.find(f"{_S}t") for r in elem.findall(f"{_S}r")]
            strings.append("".join(t.text or "" for t in runs if t is not None))
            elem.clear()
            if len(strings) >= count:
                break
    return strings


def xlsx_preview(path: Path, max_chars: int, rows: int = CSV_PREVIEW_ROWS) -> Extracted:
    """Header plus the first ``rows`` rows of each sheet, as CSV.

    Reads only what the preview needs: the first rows of each sheet and as many shared
    strings as those rows reference. Formula cells show their cached value, and date-formatted
    numbers are shown as ISO dates.
    """
    try:
        with zipfile.ZipFile(path) as zf:
            names = set(zf.namelist())
            sheets, date1904 = _xlsx_sheets(zf, names)
            date_styles = _xlsx_date_styles(zf, names)
            tables = [(name, _xlsx_rows(zf, member, rows + 2, date_styles, date1904))
                      for name, member in sheets[:MAX_XLSX_SHEETS]]
            needed = [v for _, table in tables for row in table for v in row if isinstance(v, _Shared)]
            strings = (_xlsx_shared_strings(zf, max(needed) + 1)
                       if needed and "xl/sharedStrings.xml" in names else [])
    except ExtractionError:
        raise
    except (zipfile.BadZipFile, KeyError, ElementTree.ParseError, OSError, RuntimeError, ValueError) as exc:
        raise ExtractionError(f"could not parse the XLSX ({type(exc).__name__})") from exc

    resolved = [(name, [[(strings[v] if v < len(strings) else "") if isinstance(v, _Shared) else v for v in row]
                         for row in table]) for name, table in tables]
    return _sheet_preview("XLSX", resolved, len(sheets), rows, max_chars)


def _sheet_preview(fmt: str, tables: list[tuple[str, list[list[object]]]], sheet_count: int,
                   rows: int, max_chars: int) -> Extracted:
    """Render each sheet's header plus ``rows`` rows as CSV; ``tables`` may hold one
    extra row per sheet, which only signals that more rows follow."""
    shown = ", ".join(repr(one_line(name, 40)) for name, _ in tables)
    details = [f"sheets: {sheet_count} ({shown}{', ...' if sheet_count > len(tables) else ''})"]
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    for name, table in tables:
        out.write(f"[sheet {one_line(name, 40)!r}]\n")
        if not table:
            out.write("(empty)\n")
        for row in table[: rows + 1]:
            writer.writerow([one_line(v, MAX_CSV_CELL_CHARS) for v in row])
        if len(table) > rows + 1:
            out.write("[... more rows]\n")
    text = out.getvalue().rstrip()
    return Extracted(text[:max_chars], details, len(text) > max_chars,
                     f"{fmt} header and first {rows} rows of each sheet")


# --------------------------------------------------------- legacy Office (OLE2)
def legacy_office(path: Path, max_chars: int, rows: int = CSV_PREVIEW_ROWS) -> Extracted:
    """Text of a Word 6-2003 .doc or PowerPoint 95-2003 .ppt, a sheet preview of an Excel
    97-2003 .xls, and the summary information (title, author, ...) of any legacy Office file."""
    try:
        import olefile
    except ImportError as exc:
        raise MissingDependency("reading legacy Office files needs olefile (install the 'documents' extra)") from exc
    try:
        with olefile.OleFileIO(str(path)) as ole:
            # olefile logs (rather than raises) malformed properties; OSError (incl.
            # OleFileError) means a property stream can't be read at all.
            try:
                meta, unreadable = ole.get_metadata(), []
            except OSError:
                meta, unreadable = None, ["summary information: unreadable (text decoded as cp1252)"]
            codec = _ole_codec(getattr(meta, "codepage", None))
            details = unreadable + _ole_summary(meta, codec)
            if ole.exists("WordDocument"):
                text, cut = _doc_text(ole, max_chars, codec)
                return Extracted(text, details, cut, "DOC text")
            if ole.exists("PowerPoint Document"):
                text, slide_details, cut = _ppt_text(ole, max_chars, codec)
                return Extracted(text, details + slide_details, cut, "PPT slide text")
            workbook = next((name for name in ("Workbook", "Book") if ole.exists(name)), None)
            if workbook and _xls_encrypted(ole.openstream(workbook).read()):
                raise ExtractionError("the workbook is password-protected")
    except ExtractionError:
        raise
    except Exception as exc:  # olefile raises OSError and others on malformed files
        raise ExtractionError(f"could not parse the legacy Office file ({type(exc).__name__})") from exc
    if workbook:
        result = _xls_preview(path, max_chars, rows)
        return result._replace(details=details + result.details)
    return Extracted("", details, False, "legacy Office file", empty="its text is not extracted")


# PowerPoint 97-2003 record types ([MS-PPT] 2.13.24).
_PPT_DOCUMENT, _PPT_SLIDE, _PPT_SLIDE_ATOM, _PPT_NOTES = 0x03E8, 0x03EE, 0x03EF, 0x03F0
_PPT_SLIDE_PERSIST, _PPT_MAIN_MASTER, _PPT_TEXT_CHARS, _PPT_TEXT_BYTES = 0x03F3, 0x03F8, 0x0FA0, 0x0FA8
_PPT_SLIDE_LIST, _PPT_USER_EDIT, _PPT_PERSIST_DIRECTORY = 0x0FF0, 0x0FF5, 0x1772
_PPT_ENCRYPTED_TOKEN = 0xF3D1C4DF  # CurrentUserAtom.headerToken of an encrypted file
MAX_PPT_DEPTH = 32  # record nesting followed at most
MAX_PPT_EDITS = 1_000  # saves followed back through the edit chain at most
MAX_PPT_PERSIST_ENTRIES = 1_000_000  # persist directory entries read at most, over all saves


def _ppt_records(data: bytes, start: int, end: int):
    """(type, instance, body start, body end, is container) of each record in data[start:end]."""
    pos = start
    while pos + 8 <= end:
        ver_instance, rec_type, length = struct.unpack_from("<HHI", data, pos)
        yield rec_type, ver_instance >> 4, pos + 8, min(pos + 8 + length, end), ver_instance & 0xF == 0xF
        pos += 8 + length


def _ppt_atoms(data: bytes, start: int, end: int, skip: frozenset[int] = frozenset(), depth: int = 0) -> list[str]:
    """The text atoms inside data[start:end], in file order, descending into containers."""
    texts = []
    for rec_type, _, body, stop, container in _ppt_records(data, start, end):
        if container and depth < MAX_PPT_DEPTH and rec_type not in skip:
            texts += _ppt_atoms(data, body, stop, skip, depth + 1)
        elif rec_type == _PPT_TEXT_CHARS:  # UTF-16
            texts.append(data[body:stop].decode("utf-16-le", errors="replace"))
        elif rec_type == _PPT_TEXT_BYTES:  # the low bytes of UTF-16 code units, i.e. Latin-1
            texts.append(data[body:stop].decode("latin-1"))
    return texts


def _ppt_record_at(data: bytes, offset: int, rec_type: int) -> tuple[int, int]:
    """(body start, body end) of the record of type ``rec_type`` at ``offset``."""
    if offset + 8 > len(data):
        raise ExtractionError("the presentation structure is damaged")
    _, found, length = struct.unpack_from("<HHI", data, offset)
    if found != rec_type:
        raise ExtractionError("the presentation structure is damaged")
    return offset + 8, min(offset + 8 + length, len(data))


def _ppt_text(ole, max_chars: int, codec: str) -> tuple[str, list[str], bool]:
    """Slide text of a .ppt, slide by slide, with each slide's speaker notes.

    PowerPoint 97-2003: the Current User stream points to the last save (UserEditAtom),
    whose chain of persist directories maps object ids to stream offsets; the document's
    slide list gives the slides in order. A slide's text is in its shapes and, for
    placeholders, in the slide list after its SlidePersistAtom. Masters are skipped.
    If that structure is damaged, all text atoms outside masters are returned instead.
    PowerPoint 95 keeps plain text in a Text_Content stream, in the document's code page.
    """
    current = ole.openstream("Current User").read() if ole.exists("Current User") else b""
    if ole.exists("EncryptedSummary") or (len(current) >= 16 and
                                          struct.unpack_from("<I", current, 12)[0] == _PPT_ENCRYPTED_TOKEN):
        raise ExtractionError("the presentation is password-protected")
    data = ole.openstream("PowerPoint Document").read()
    try:
        slides = _ppt_slides(data, struct.unpack_from("<I", current, 16)[0])
    except (ExtractionError, struct.error):
        slides = None
    if slides is None and ole.exists("Text_Content"):  # PowerPoint 95
        raw = ole.openstream("Text_Content").read(max_chars * 4)
        text = _clean_ppt_text(raw.decode(codec, errors="replace"))
        return text[:max_chars], [], len(text) > max_chars
    if slides is None:
        texts = [_clean_ppt_text(t) for t in _ppt_atoms(data, 0, len(data), frozenset({_PPT_MAIN_MASTER}))]
        text = "\n".join(t for t in texts if t)
        return text[:max_chars], ["slides: unknown (damaged file; text in file order)"], len(text) > max_chars

    parts = []
    for number, (texts, notes) in enumerate(slides, start=1):
        parts.append(f"[slide {number}]")
        parts += [t for t in map(_clean_ppt_text, texts) if t]
        notes = [t for t in map(_clean_ppt_text, notes) if t]
        if notes:
            parts += ["[notes]", *notes]
    text = "\n".join(parts)
    return text[:max_chars], [f"slides: {len(slides)}"], len(text) > max_chars


def _ppt_slides(data: bytes, current_edit: int) -> list[tuple[list[str], list[str]]]:
    """(text atoms, notes text atoms) of each slide, in presentation order."""
    offsets: dict[int, int] = {}  # persist object id -> stream offset; the newest save wins
    document_ref, edit, seen, directories, entries = None, current_edit, set(), set(), 0
    while edit and edit not in seen and len(seen) < MAX_PPT_EDITS:
        seen.add(edit)
        body, _ = _ppt_record_at(data, edit, _PPT_USER_EDIT)
        previous_edit, directory, doc_ref = struct.unpack_from("<III", data, body + 8)
        document_ref = document_ref if document_ref is not None else doc_ref
        edit = previous_edit
        if directory in directories:  # walked already: it adds nothing (setdefault), only CPU
            continue
        directories.add(directory)
        pos, end = _ppt_record_at(data, directory, _PPT_PERSIST_DIRECTORY)
        while pos + 4 <= end:
            entry = struct.unpack_from("<I", data, pos)[0]
            first, count = entry & 0xFFFFF, entry >> 20
            entries += count
            if entries > MAX_PPT_PERSIST_ENTRIES:
                raise ExtractionError("the presentation's persist directories are implausibly large")
            for k in range(count):
                offsets.setdefault(first + k, struct.unpack_from("<I", data, pos + 4 + 4 * k)[0])
            pos += 4 + 4 * count
    if document_ref not in offsets:
        raise ExtractionError("the presentation has no document record")

    lists: dict[int, list[tuple[int, int, list[str]]]] = {}  # instance -> (persist ref, id, texts)
    for rec_type, instance, body, stop, _ in _ppt_records(data, *_ppt_record_at(data, offsets[document_ref],
                                                                                 _PPT_DOCUMENT)):
        if rec_type != _PPT_SLIDE_LIST:
            continue
        entries = lists.setdefault(instance, [])  # 0: slides, 1: masters, 2: notes
        for child, _, child_body, child_stop, _ in _ppt_records(data, body, stop):
            if child == _PPT_SLIDE_PERSIST:
                persist_ref, _, _, item_id = struct.unpack_from("<IIiI", data, child_body)
                entries.append((persist_ref, item_id, []))
            elif entries and child in (_PPT_TEXT_CHARS, _PPT_TEXT_BYTES):
                entries[-1][2].extend(_ppt_atoms(data, child_body - 8, child_stop))

    notes_by_id = {}
    for persist_ref, notes_id, _ in lists.get(2, []):
        if persist_ref in offsets:
            with contextlib.suppress(ExtractionError):
                notes_by_id[notes_id] = _ppt_atoms(data, *_ppt_record_at(data, offsets[persist_ref], _PPT_NOTES))
    slides = []
    for persist_ref, _, outline in lists.get(0, []):
        texts, notes = list(outline), []
        if persist_ref in offsets:
            body, stop = _ppt_record_at(data, offsets[persist_ref], _PPT_SLIDE)
            texts += _ppt_atoms(data, body, stop)
            for rec_type, _, atom, atom_stop, _ in _ppt_records(data, body, stop):
                if rec_type == _PPT_SLIDE_ATOM and atom_stop - atom >= 20:
                    notes = notes_by_id.get(struct.unpack_from("<I", data, atom + 16)[0], [])
        slides.append((texts, notes))
    return slides


def _clean_ppt_text(text: str) -> str:
    """Paragraph (\\r) and line (\\v) breaks become newlines; other control characters go,
    and so does "*" alone, the placeholder of a slide-number field."""
    if text.strip() == "*":
        return ""
    return _tidy_lines(text.replace("\r", "\n").replace("\x0b", "\n"))


def _tidy_lines(text: str) -> str:
    """Drop control characters (except tab and newline), trailing spaces of each line,
    and runs of empty lines beyond one."""
    text = "".join(ch for ch in text if ch >= " " or ch in "\t\n")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _ole_codec(codepage: object) -> str:
    """Python codec for a Windows code page from the summary information (default cp1252)."""
    names = {1200: "utf-16-le", 65001: "utf-8", 10000: "mac_roman"}
    if isinstance(codepage, int):
        name = names.get(codepage) or names.get(codepage & 0xFFFF) or f"cp{codepage & 0xFFFF}"
        try:
            return codecs.lookup(name).name
        except LookupError:
            pass
    return "cp1252"


def _ole_summary(meta, codec: str) -> list[str]:
    """Title, subject, author, ... from the OLE SummaryInformation stream."""
    if meta is None:
        return []
    lines = []
    for attr in ("title", "subject", "author", "keywords", "company", "creating_application",
                 "create_time", "last_saved_time"):
        value = getattr(meta, attr, None)
        if isinstance(value, bytes):
            value = value.decode(codec, errors="replace")
        if isinstance(value, datetime):
            if value.year < 1980:  # unset: FILETIME 0 is 1601-01-01
                continue
            value = value.isoformat(sep=" ", timespec="seconds")
        if value not in (None, "") and one_line(value):
            lines.append(f"{attr.replace('_', ' ')}: {one_line(value)}")
    return lines


def _doc_text(ole, max_chars: int, codec: str = "cp1252") -> tuple[str, bool]:
    """The main text of a Word document, read through its piece table ([MS-DOC]).

    Word 97-2003: the FIB at the start of the WordDocument stream locates the CLX in the
    table stream; its piece table maps character positions to runs of 8-bit (cp1252) or
    UTF-16 text. Word 6/95: the text is 8-bit in the document's code page (``codec``),
    either in one run or, for fast-saved files, through a piece table in the WordDocument
    stream itself. Only the main document is read (not headers, footnotes or comments).
    """
    word = ole.openstream("WordDocument").read()
    if word[:2] == b"\xdc\xa5":  # wIdent 0xA5DC: Word 6/95
        return _doc6_text(word, max_chars, codec)

    def u16(at: int) -> int:
        return struct.unpack_from("<H", word, at)[0]

    def u32(at: int) -> int:
        return struct.unpack_from("<I", word, at)[0]

    try:
        if u16(0) != 0xA5EC:
            raise ExtractionError("not a Word 6-2003 document")
        flags = u16(0x0A)
        if flags & 0x0100:
            raise ExtractionError("the document is password-protected")
        pos = 32 + 2 + u16(32) * 2  # FibBase, csw, fibRgW
        cslw = u16(pos)
        ccp_text = u32(pos + 2 + 3 * 4)  # fibRgLw.ccpText: characters of the main document
        pos += 2 + cslw * 4
        if u16(pos) < 34:  # cbRgFcLcb: too short to hold fcClx/lcbClx (pair 33)
            raise ExtractionError("the document has no piece table")
        fc_clx, lcb_clx = struct.unpack_from("<II", word, pos + 2 + 33 * 8)
        table = ole.openstream("1Table" if flags & 0x0200 else "0Table").read()
        clx = table[fc_clx:fc_clx + lcb_clx]
    except struct.error as exc:
        raise ExtractionError("the document structure is damaged") from exc
    return _pieces_text(word, clx, ccp_text, max_chars, word97=True)


def _doc6_text(word: bytes, max_chars: int, codec: str) -> tuple[str, bool]:
    """Word 6/95: FIB fields fComplex (0x0A bit 2), fcMin (0x18), ccpText (0x34) and
    fcClx/lcbClx (0x160/0x164); there is no separate table stream."""
    try:
        flags, fc_min = struct.unpack_from("<H", word, 0x0A)[0], struct.unpack_from("<I", word, 0x18)[0]
        if flags & 0x0100:
            raise ExtractionError("the document is password-protected")
        ccp_text = struct.unpack_from("<I", word, 0x34)[0]
        if not flags & 0x0004:  # not fast-saved: the text is one run from fcMin
            raw = word[fc_min:fc_min + min(ccp_text, max_chars * 2)]
            text = _clean_doc_text(raw.decode(codec, errors="replace"))
            return text[:max_chars], ccp_text > max_chars * 2 or len(text) > max_chars
        fc_clx, lcb_clx = struct.unpack_from("<II", word, 0x160)
    except struct.error as exc:
        raise ExtractionError("the document structure is damaged") from exc
    return _pieces_text(word, word[fc_clx:fc_clx + lcb_clx], ccp_text, max_chars, word97=False, codec=codec)


def _pieces_text(word: bytes, clx: bytes, ccp_text: int, max_chars: int, *, word97: bool,
                 codec: str = "cp1252") -> tuple[str, bool]:
    """The first ``ccp_text`` characters, assembled from the piece table in ``clx``."""
    try:
        i = 0
        while i < len(clx) and clx[i] == 0x01:  # Prc entries (formatting) come first
            i += 3 + struct.unpack_from("<h", clx, i + 1)[0]
        if i >= len(clx) or clx[i] != 0x02:
            raise ExtractionError("the document has no piece table")
        lcb = struct.unpack_from("<I", clx, i + 1)[0]
        plc = clx[i + 5:i + 5 + lcb]
        n = (len(plc) - 4) // 12
        cps = struct.unpack_from(f"<{n + 1}I", plc, 0)
        fcs = [struct.unpack_from("<I", plc, (n + 1) * 4 + k * 8 + 2)[0] for k in range(n)]
    except struct.error as exc:
        raise ExtractionError("the document structure is damaged") from exc

    parts, total, budget = [], 0, max_chars * 2  # field codes are dropped afterwards
    for k in range(n):
        start, end = cps[k], min(cps[k + 1], ccp_text)
        if start >= end:
            continue
        fc = fcs[k]
        if not word97:  # Word 6/95: always 8-bit, in the document's code page
            parts.append(word[fc:fc + end - start].decode(codec, errors="replace"))
        elif fc & 0x40000000:  # "compressed": one byte per character, cp1252 by definition
            offset = (fc & 0x3FFFFFFF) // 2
            parts.append(word[offset:offset + end - start].decode("cp1252", errors="replace"))
        else:
            offset = fc & 0x3FFFFFFF
            parts.append(word[offset:offset + 2 * (end - start)].decode("utf-16-le", errors="replace"))
        total += len(parts[-1])
        if total >= budget:
            break
    text = _clean_doc_text("".join(parts))
    return text[:max_chars], total >= budget or len(text) > max_chars


_DOC_CONTROL = str.maketrans({"\r": "\n", "\x0b": "\n", "\x0c": "\n", "\x0e": "\n", "\x07": "\t",
                              "\x1e": "-", "\x1f": None, "\xa0": " "})


def _clean_doc_text(raw: str) -> str:
    """Word's control characters to plain text: paragraph/line/page breaks become newlines,
    table cell marks tabs (a row's end a newline; an empty last cell can look the same), and field instructions (e.g. HYPERLINK "...") are dropped while
    the field's displayed result is kept."""
    out, fields = [], []  # fields: per open field, True once past its separator
    for ch in raw:
        if ch == "\x13":
            fields.append(False)
        elif ch == "\x14":
            if fields:
                fields[-1] = True
        elif ch == "\x15":
            if fields:
                fields.pop()
        elif all(fields):
            out.append(ch)
    # A table row ends with its last cell mark followed by the row mark: one row per line.
    return _tidy_lines("".join(out).replace("\x07\x07", "\n").translate(_DOC_CONTROL))


_XLS_EOF, _XLS_FILEPASS = 0x000A, 0x002F  # BIFF record types


def _xls_encrypted(stream: bytes) -> bool:
    """Whether the workbook globals of a BIFF stream hold a FILEPASS record (encryption)."""
    pos = 0
    while pos + 4 <= len(stream):
        rec_type, length = struct.unpack_from("<HH", stream, pos)
        if rec_type == _XLS_FILEPASS:
            return True
        if rec_type == _XLS_EOF:
            return False
        pos += 4 + length
    return False


def _xls_preview(path: Path, max_chars: int, rows: int) -> Extracted:
    try:
        import xlrd
    except ImportError as exc:
        raise MissingDependency("reading .xls files needs xlrd (install the 'documents' extra)") from exc
    try:
        # xlrd prints warnings to stdout by default; send them nowhere.
        book = xlrd.open_workbook(str(path), on_demand=True, logfile=io.StringIO())
        try:
            tables = []
            for index in range(min(book.nsheets, MAX_XLSX_SHEETS)):
                sheet = book.sheet_by_index(index)
                table = []
                for r in range(sheet.nrows):
                    row = [_xls_cell(cell, book.datemode, xlrd) for cell in sheet.row(r)[:MAX_XLSX_COLUMNS]]
                    while row and row[-1] == "":
                        row.pop()
                    if row:
                        table.append(row)
                        if len(table) == rows + 2:
                            break
                tables.append((sheet.name, table))
                book.unload_sheet(index)
            sheet_count = book.nsheets
        finally:
            book.release_resources()
    except Exception as exc:  # xlrd.XLRDError, struct errors on damaged files
        raise ExtractionError(f"could not parse the XLS ({type(exc).__name__})") from exc
    return _sheet_preview("XLS", tables, sheet_count, rows, max_chars)


def _xls_cell(cell, datemode: int, xlrd) -> object:
    if cell.ctype == xlrd.XL_CELL_DATE:
        try:
            moment = xlrd.xldate.xldate_as_datetime(cell.value, datemode)
            return _format_excel_moment(moment, cell.value)  # rounding may overflow too
        except (ValueError, OverflowError, xlrd.xldate.XLDateError):
            return cell.value
    if cell.ctype == xlrd.XL_CELL_BOOLEAN:
        return "TRUE" if cell.value else "FALSE"
    if cell.ctype == xlrd.XL_CELL_ERROR:
        return xlrd.error_text_from_code.get(cell.value, "#ERROR")
    if cell.ctype == xlrd.XL_CELL_NUMBER and cell.value == int(cell.value):
        return int(cell.value)  # 189.0 -> 189, like Excel shows it
    return "" if cell.ctype in (xlrd.XL_CELL_EMPTY, xlrd.XL_CELL_BLANK) else cell.value


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
    if (position := _gps_decimal(tags)) is not None:
        lines.insert(1, f"gps: {position[0]:.5f}, {position[1]:.5f} (decimal degrees)")
    return lines


def _gps_decimal(tags: dict[str, object]) -> tuple[float, float] | None:
    """(latitude, longitude) from EXIF GPS degrees/minutes/seconds, or None."""
    try:
        coords = []
        for axis, negative in (("Latitude", "S"), ("Longitude", "W")):
            degrees, minutes, seconds = (float(v) for v in tags[f"GPS{axis}"])
            value = degrees + minutes / 60 + seconds / 3600
            ref = str(tags.get(f"GPS{axis}Ref", "")).strip().upper()
            coords.append(-value if ref == negative else value)
    except (KeyError, TypeError, ValueError, ZeroDivisionError):  # missing or malformed
        return None
    if not (-90 <= coords[0] <= 90 and -180 <= coords[1] <= 180):
        return None
    return coords[0], coords[1]
