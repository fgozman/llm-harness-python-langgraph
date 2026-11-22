"""Legacy Office files (OLE2): Word 6-2003 .doc text, PowerPoint 95-2003 .ppt slide text,
Excel 97-2003 .xls previews, summary information, and the fallbacks.

Real files made by LibreOffice are in tests/fixtures/office (see its README). Formats
LibreOffice can't write (Word 6/95, .ppt) and broken files are built here with a minimal
OLE2 / Compound File writer.
"""

from __future__ import annotations

import os
import shutil
import struct
import sys
import uuid
from pathlib import Path

import pytest

from file_organizer import extract
from file_organizer.workspace import Workspace

FIXTURES = Path(__file__).parent / "fixtures" / "office"

_END, _FREE, _FATSECT, _NONE = 0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFD, 0xFFFFFFFF
_SUMMARY_FMTID = uuid.UUID("F29F85E0-4FF9-1068-AB91-08002B27B3D9").bytes_le


def make_ole(streams: dict[str, bytes]) -> bytes:
    """A version-3 Compound File holding ``streams`` at the root.

    Streams are padded to 4 KiB so they live in regular sectors (no mini stream needed),
    and everything fits in one FAT sector.
    """
    names = list(streams)
    datas = [d.ljust(-(-max(len(d), 4096) // 512) * 512, b"\0") for d in streams.values()]
    dir_sectors = -(-(len(names) + 1) // 4)
    fat = [_FATSECT] + [s + 1 if s < dir_sectors else _END for s in range(1, dir_sectors + 1)]
    starts = []
    for data in datas:
        starts.append(len(fat))
        count = len(data) // 512
        fat += [len(fat) + k + 1 if k < count - 1 else _END for k in range(count)]
    assert len(fat) <= 128, "test files must fit one FAT sector"

    def entry(name: str, kind: int, start: int, size: int, child: int = _NONE, right: int = _NONE) -> bytes:
        encoded = (name + "\0").encode("utf-16-le")
        return (encoded.ljust(64, b"\0") + struct.pack("<HBBIII", len(encoded), kind, 1, _NONE, right, child)
                + b"\0" * 36 + struct.pack("<IQ", start, size))

    directory = entry("Root Entry", 5, _END, 0, child=1 if names else _NONE)
    for i, (name, start, data) in enumerate(zip(names, starts, datas), start=1):
        directory += entry(name, 2, start, len(data), right=i + 1 if i < len(names) else _NONE)
    directory = directory.ljust(dir_sectors * 512, b"\0")
    header = (b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\0" * 16 + struct.pack("<HHHHH", 0x3E, 3, 0xFFFE, 9, 6)
              + b"\0" * 6 + struct.pack("<9I", 0, 1, 1, 0, 4096, _END, 0, _END, 0)
              + struct.pack("<109I", 0, *[_FREE] * 108))
    fat_sector = struct.pack("<128I", *fat, *[_FREE] * (128 - len(fat)))
    return header + fat_sector + directory + b"".join(datas)


def summary_info(codepage: int, title: bytes) -> bytes:
    """A SummaryInformation property set with a code page and a title."""
    title_value = struct.pack("<II", 0x1E, len(title) + 1) + title + b"\0"
    title_value = title_value.ljust(-(-len(title_value) // 4) * 4, b"\0")
    values = [struct.pack("<Ih2x", 2, codepage), title_value]  # VT_I2, VT_LPSTR
    offset, table = 8 + 8 * len(values), b""
    for prop_id, value in zip((1, 2), values):
        table += struct.pack("<II", prop_id, offset)
        offset += len(value)
    section = struct.pack("<II", offset, len(values)) + table + b"".join(values)
    return struct.pack("<HHI", 0xFFFE, 0, 0x00020006) + b"\0" * 16 + struct.pack("<I", 1) \
        + _SUMMARY_FMTID + struct.pack("<I", 48) + section


def word6_stream(text: bytes, *, fast_saved: bool = False, encrypted: bool = False) -> bytes:
    """A Word 6/95 WordDocument stream. Fast-saved files store their text in pieces, here
    deliberately out of order so only a correct piece-table walk restores it."""
    fib = bytearray(0x600)
    struct.pack_into("<HH", fib, 0, 0xA5DC, 0x65)  # wIdent, nFib
    struct.pack_into("<H", fib, 0x0A, (0x0004 if fast_saved else 0) | (0x0100 if encrypted else 0))
    struct.pack_into("<I", fib, 0x18, 0x600)  # fcMin
    struct.pack_into("<I", fib, 0x34, len(text))  # ccpText
    if not fast_saved:
        return bytes(fib) + text
    half = len(text) // 2
    first, second = text[:half], text[half:]
    body = second + first  # stored in reverse order
    pcds = struct.pack("<HIH", 0, 0x600 + len(second), 0) + struct.pack("<HIH", 0, 0x600, 0)
    plc = struct.pack("<3I", 0, len(first), len(text)) + pcds
    clx = b"\x01" + struct.pack("<h", 2) + b"\0\0" + b"\x02" + struct.pack("<I", len(plc)) + plc  # Prc + Pcdt
    struct.pack_into("<II", fib, 0x160, 0x600 + len(body), len(clx))  # fcClx, lcbClx
    return bytes(fib) + body + clx


def ppt_record(rec_type: int, body: bytes = b"", *children: bytes, instance: int = 0) -> bytes:
    """A PowerPoint record; with ``children`` it is a container (recVer 0xF)."""
    container = bool(children)
    body = b"".join(children) if container else body
    return struct.pack("<HHI", (instance << 4) | (0xF if container else 0), rec_type, len(body)) + body


def ppt_text(text: str) -> bytes:
    """TextBytesAtom for Latin-1 text, TextCharsAtom (UTF-16) otherwise."""
    try:
        return ppt_record(0x0FA8, text.encode("latin-1"))
    except UnicodeEncodeError:
        return ppt_record(0x0FA0, text.encode("utf-16-le"))


def make_ppt(slides: list[dict], *, damaged_chain: bool = False, encrypted: bool = False) -> bytes:
    """A PowerPoint 97-2003 file. Each slide: {"shapes": [...], "outline": [...], "notes": [...]}.

    Laid out to catch shortcuts: slides are stored in reverse order (the slide list gives
    the order), the master holds placeholder text, and an earlier save left a stale copy
    of slide 1 that only the newest persist directory replaces.
    """
    stream, offsets = b"", {}

    def add(persist_id: int, record: bytes) -> None:
        nonlocal stream
        offsets[persist_id] = len(stream)
        stream += record

    def slide_record(index: int, shapes: list[str]) -> bytes:
        slide_atom = struct.pack("<I8sII4x", 0, b"", 1, 100 + index if slides[index].get("notes") else 0)
        drawing = ppt_record(0xF002, b"", ppt_record(0xF003, b"", *[
            ppt_record(0xF004, b"", ppt_record(0xF00D, b"", ppt_text(s))) for s in shapes]))
        return ppt_record(0x03EE, b"", ppt_record(0x03EF, slide_atom), ppt_record(0x040C, b"", drawing))

    def persist_dir(ids: list[int]) -> bytes:
        return ppt_record(0x1772, b"".join(struct.pack("<II", (1 << 20) | i, offsets[i]) for i in ids))

    def user_edit(previous: int, directory: int) -> bytes:
        return ppt_record(0x0FF5, struct.pack("<IHBBIIII", 0, 0, 0, 3, previous, directory, 2, 0))

    add(1, ppt_record(0x03F8, b"", ppt_text("Click to edit Master title style")))  # main master
    add(10, slide_record(0, ["STALE TEXT FROM AN EARLIER SAVE"]))
    for index in reversed(range(len(slides))):
        if slides[index].get("notes"):
            add(50 + index, ppt_record(0x03F0, b"", *[ppt_text(t) for t in slides[index]["notes"]]))
        if index:
            add(10 + index, slide_record(index, slides[index].get("shapes", [])))
    slide_list = [b"".join([ppt_record(0x03F3, struct.pack("<IIiII", 10 + i, 0, 0, 256 + i, 0)),
                            *[ppt_text(t) for t in slide.get("outline", [])]]) for i, slide in enumerate(slides)]
    notes_list = [ppt_record(0x03F3, struct.pack("<IIiII", 50 + i, 0, 0, 100 + i, 0))
                  for i, slide in enumerate(slides) if slide.get("notes")]
    add(2, ppt_record(0x03E8, b"", ppt_record(0x03E9, b"\0" * 40),
                      ppt_record(0x0FF0, b"", *slide_list, instance=0),
                      *([ppt_record(0x0FF0, b"", *notes_list, instance=2)] if notes_list else [])))
    ids = sorted(offsets)
    first_dir = len(stream)
    stream += persist_dir(ids)
    first_edit = len(stream)
    stream += user_edit(0, first_dir)
    add(10, slide_record(0, slides[0].get("shapes", [])))  # the newest version of slide 1
    second_dir = len(stream)
    stream += persist_dir([10])
    current_edit = len(stream)
    stream += user_edit(first_edit, second_dir)
    token = 0xF3D1C4DF if encrypted else 0xE391C05F
    current_user = ppt_record(0x0FF6, struct.pack("<III", 20, token, 7 if damaged_chain else current_edit)
                              + b"\0" * 12)
    return make_ole({"Current User": current_user, "PowerPoint Document": stream,
                     "\x05SummaryInformation": summary_info(1252, b"Q2 Review")})


@pytest.fixture
def ws(tmp_path: Path) -> Workspace:
    return Workspace(tmp_path)


def put(ws: Workspace, name: str, data: bytes) -> str:
    (ws.root / name).write_bytes(data)
    return name


def copy_fixture(ws: Workspace, name: str, as_name: str | None = None) -> str:
    shutil.copy(FIXTURES / name, ws.root / (as_name or name))
    return as_name or name


def content(out: str) -> str:
    return out.split("\n[content", 1)[1].split("\n", 1)[1]


class TestWord97:
    def test_text_of_a_real_document(self, ws: Workspace):
        out = ws.read_file(copy_fixture(ws, "word97_contract.doc"))
        assert "mime type: application/msword" in out and "title: Employment contract" in out
        assert content(out) == (
            "Arbeitsvertrag – Müller GmbH\n"
            "Gehalt: 4.200 € brutto. Mitarbeiterin: Zoë Ångström.\n"
            "Portal: Personalportal.\n"  # a hyperlink field: its text is kept, its code dropped
            "Urlaub\t30 Tage\n"  # table rows
            "Probezeit\t6 Monate\n"
            "Ende des Vertrags."
        )

    def test_long_documents_use_the_default_view_and_paging(self, tmp_path: Path):
        ws = Workspace(tmp_path, max_read_chars=30)
        name = copy_fixture(ws, "word97_contract.doc")
        assert "truncated" in ws.read_file(name)
        assert content(ws.read_file(name, offset=0, limit=14)) == "Arbeitsvertrag"

    def test_password_protected(self, ws: Workspace):
        fib = struct.pack("<HHHHH", 0xA5EC, 0xC1, 0, 0, 0)[:10] + struct.pack("<H", 0x0100)
        out = ws.read_file(put(ws, "secret.doc", make_ole({"WordDocument": fib})))
        assert "password-protected" in out and "mime type: application/msword" in out

    def test_damaged(self, ws: Workspace):
        out = ws.read_file(put(ws, "broken.doc", make_ole({"WordDocument": struct.pack("<H", 0xA5EC)})))
        assert "[content: not available - the document has no piece table" in out


class TestWord6:
    def test_simple_file_in_its_code_page(self, ws: Workspace):
        text = "Договор аренды\rАрендная плата: 50 000 рублей".encode("cp1251")
        out = ws.read_file(put(ws, "lease.doc", make_ole({
            "WordDocument": word6_stream(text),
            "\x05SummaryInformation": summary_info(1251, "Договор".encode("cp1251")),
        })))
        assert "title: Договор" in out
        assert content(out) == "Договор аренды\nАрендная плата: 50 000 рублей"

    def test_fast_saved_file_is_read_through_its_piece_table(self, ws: Workspace):
        text = b"Invoice 2024-031\rTotal due: 120 EUR\rThank you"
        out = ws.read_file(put(ws, "invoice.doc", make_ole({"WordDocument": word6_stream(text, fast_saved=True)})))
        assert content(out) == "Invoice 2024-031\nTotal due: 120 EUR\nThank you"

    def test_password_protected(self, ws: Workspace):
        stream = word6_stream(b"secret", encrypted=True)
        assert "password-protected" in ws.read_file(put(ws, "s.doc", make_ole({"WordDocument": stream})))


class TestExcel97:
    def test_preview_of_a_real_workbook(self, ws: Workspace):
        out = ws.read_file(copy_fixture(ws, "excel97_budget.xls"))
        assert "mime type: application/vnd.ms-excel" in out and "sheets: 2 ('Expenses', 'Notes')" in out
        assert content(out).splitlines() == [
            "[sheet 'Expenses']", "date,item,amount",
            "2024-01-03,Flight BER-LIS,189", "2024-01-04,Hotel Lisbon,420.5",
            "2024-01-05,Taxi 0,10", "2024-01-06,Taxi 1,11", "2024-01-07,Taxi 2,12", "[... more rows]",
            "[sheet 'Notes']", "approved by,Anna Berg", "paid,yes",
        ]

    def test_type_comes_from_the_content_not_the_name(self, ws: Workspace):
        out = ws.read_file(copy_fixture(ws, "excel97_budget.xls", "report.doc"))
        assert "mime type: application/msword" in out  # the name's type, as metadata
        assert "XLS header and first 5 rows" in out and "Flight BER-LIS" in out

    def test_missing_xlrd(self, ws: Workspace, monkeypatch):
        monkeypatch.setitem(sys.modules, "xlrd", None)
        out = ws.read_file(copy_fixture(ws, "excel97_budget.xls"))
        assert "needs xlrd" in out and "Flight" not in out

    def test_password_protected(self, ws: Workspace):
        bof = struct.pack("<HH", 0x0809, 16) + b"\0" * 16
        filepass = struct.pack("<HH", 0x002F, 6) + b"\0" * 6
        out = ws.read_file(put(ws, "secret.xls", make_ole({"Workbook": bof + filepass})))
        assert "the workbook is password-protected" in out


@pytest.mark.parametrize(("stream", "encrypted"), [
    (struct.pack("<HH", 0x0809, 0) + struct.pack("<HH", 0x002F, 0), True),
    (struct.pack("<HH", 0x0809, 0) + struct.pack("<HH", 0x000A, 0) + struct.pack("<HH", 0x002F, 0), False),
    (b"", False),
])
def test_xls_encryption_is_read_from_the_globals(stream: bytes, encrypted: bool):
    assert extract._xls_encrypted(stream) is encrypted


PPT_SLIDES = [
    {"outline": ["Quarterly Review\rQ2 2024"], "shapes": ["Confidential"], "notes": ["Mention Lisbon", "*"]},
    {"outline": ["Results"], "shapes": ["Revenue up 12 %\x0bTwo new clients", "Umsatz: 4.200 €"]},
    {"shapes": ["Next steps"], "notes": ["Hire two engineers"]},
]


class TestPowerPoint:
    def test_slides_in_order_with_their_notes(self, ws: Workspace):
        out = ws.read_file(put(ws, "q2.ppt", make_ppt(PPT_SLIDES)))
        assert "mime type: application/vnd.ms-powerpoint" in out and "title: Q2 Review" in out
        assert "slides: 3" in out
        assert content(out) == (
            "[slide 1]\nQuarterly Review\nQ2 2024\nConfidential\n[notes]\nMention Lisbon\n"
            "[slide 2]\nResults\nRevenue up 12 %\nTwo new clients\nUmsatz: 4.200 €\n"
            "[slide 3]\nNext steps\n[notes]\nHire two engineers"
        )  # no master text, no stale slide, no slide-number placeholder ("*")

    def test_damaged_edit_chain_falls_back_to_all_text_outside_masters(self, ws: Workspace):
        out = ws.read_file(put(ws, "q2.ppt", make_ppt(PPT_SLIDES, damaged_chain=True)))
        assert "slides: unknown" in out and "Revenue up 12 %" in out and "Hire two engineers" in out
        assert "Master title" not in out

    def test_password_protected(self, ws: Workspace):
        out = ws.read_file(put(ws, "q2.ppt", make_ppt(PPT_SLIDES, encrypted=True)))
        assert "password-protected" in out and "Revenue" not in out

    def test_powerpoint_95_text_content(self, ws: Workspace):
        out = ws.read_file(put(ws, "old.ppt", make_ole({
            "PowerPoint Document": b"\0" * 64,
            "Text_Content": "Отчёт\r\rВыручка выросла".encode("cp1251"),
            "\x05SummaryInformation": summary_info(1251, b"Report"),
        })))
        assert content(out) == "Отчёт\n\nВыручка выросла"

    def test_empty_presentation(self, ws: Workspace):
        out = ws.read_file(put(ws, "blank.ppt", make_ole({"PowerPoint Document": b"\0" * 64})))
        assert "[content: none - PPT slide text: it has no extractable text" in out


class TestOtherOle:
    def test_other_ole_files_get_summary_information_only(self, ws: Workspace):
        out = ws.read_file(put(ws, "mail.msg", make_ole({
            "__substg1.0_0037001F": "Subject".encode("utf-16-le"),
            "\x05SummaryInformation": summary_info(1252, b"Weekly report"),
        })))
        assert "title: Weekly report" in out and "legacy Office file: its text is not extracted" in out

    def test_corrupt_container(self, ws: Workspace):
        out = ws.read_file(put(ws, "bad.doc", b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\0" * 600))
        assert "mime type: application/msword" in out and "could not parse the legacy Office file" in out

    def test_missing_olefile(self, ws: Workspace, monkeypatch):
        monkeypatch.setitem(sys.modules, "olefile", None)
        out = ws.read_file(copy_fixture(ws, "word97_contract.doc"))
        assert "needs olefile" in out and "Arbeitsvertrag" not in out

    def test_over_the_byte_cap_is_not_parsed(self, tmp_path: Path):
        ws = Workspace(tmp_path, max_parse_bytes=1_000)
        assert "not parsed" in ws.read_file(copy_fixture(ws, "word97_contract.doc"))

    def test_unset_dates_are_not_shown(self, ws: Workspace):
        # LibreOffice leaves creation/save times at FILETIME 0 (1601-01-01).
        assert "1601" not in ws.read_file(copy_fixture(ws, "word97_contract.doc"))


@pytest.mark.parametrize(("raw", "text"), [
    ("a\rb\x0bc\x0cd", "a\nb\nc\nd"),  # paragraph, line and page breaks
    ('x \x13 HYPERLINK "http://e.com" \x14link\x15 y', "x link y"),  # field result kept
    ("\x13 PAGE \x15end", "end"),  # field without a result
    ("\x13 IF \x13 DATE \x14d\x15 \x14outer\x15", "outer"),  # nested fields
    ("c1\x07c2\x07\x07c3\x07c4\x07\x07", "c1\tc2\nc3\tc4"),  # table cells and rows
    ("a\r\r\r\r\rb", "a\n\nb"),  # runs of empty paragraphs
    ("soft\x1fhy\x1ephen\x01\x08", "softhy-phen"),
])
def test_clean_doc_text(raw: str, text: str):
    assert extract._clean_doc_text(raw) == text


def test_ppt_record_walk_stops_at_its_bounds():
    # A length running past the parent's end and a never-ending nesting stay bounded.
    data = ppt_record(0x0FA8, b"ok") + struct.pack("<HHI", 0, 0x0FA8, 10_000) + b"cut"
    assert extract._ppt_atoms(data, 0, len(data)) == ["ok", "cut"]
    nested = ppt_text("deep")
    for _ in range(100):
        nested = ppt_record(0x0FF0, b"", nested)
    assert extract._ppt_atoms(nested, 0, len(nested)) == []  # deeper than MAX_PPT_DEPTH


def test_ppt_edit_chain_walks_each_persist_directory_once(monkeypatch):
    # Many saves pointing at one big persist directory must not re-walk it each time.
    directory = ppt_record(0x1772, struct.pack("<I", (3 << 20) | 1) + struct.pack("<III", 0, 0, 0))
    edits, data = [], directory
    for _ in range(2):  # UserEditAtom: previous edit, persist directory, document ref
        previous = edits[-1] if edits else 0
        edits.append(len(data))
        data += ppt_record(0x0FF5, struct.pack("<8xIII", previous, 0, 99))
    monkeypatch.setattr(extract, "MAX_PPT_PERSIST_ENTRIES", 5)  # 3 entries, read once
    with pytest.raises(extract.ExtractionError, match="no document record"):
        extract._ppt_slides(data, edits[-1])
    monkeypatch.setattr(extract, "MAX_PPT_PERSIST_ENTRIES", 2)
    with pytest.raises(extract.ExtractionError, match="implausibly large"):
        extract._ppt_slides(data, edits[-1])


@pytest.mark.parametrize(("codepage", "codec"), [
    (1251, "cp1251"), (-535, "utf-8"), (65001, "utf-8"), (1200, "utf-16-le"), (None, "cp1252"), (9999, "cp1252"),
])
def test_ole_codec(codepage, codec: str):
    assert extract._ole_codec(codepage) == codec


def test_fixtures_are_present():
    assert sorted(os.listdir(FIXTURES)) == ["README.md", "excel97_budget.xls", "word97_contract.doc"]
