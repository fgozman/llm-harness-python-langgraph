"""Keep the demo data, its expectations and its helper script consistent."""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

import pytest

from file_organizer.workspace import Workspace

DEMO = Path(__file__).resolve().parent.parent / "demo"


@pytest.fixture
def demo(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("demo_script", DEMO / "demo.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "INBOX", tmp_path / "inbox")  # never touch the real demo/inbox
    return module


def sample_files() -> set[str]:
    sample = DEMO / "sample"
    return {
        p.relative_to(sample).as_posix()
        for p in sample.rglob("*")
        if p.is_file() and not p.name.startswith(".")
    }


def test_expected_covers_exactly_the_sample_files(demo):
    assert set(demo.load_expected()) == sample_files()


def test_sample_exercises_edge_cases():
    ws = Workspace(DEMO / "sample")
    listing = ws.list_files()
    assert "finance/" in listing  # pre-existing category
    assert not any(name.startswith(".") for name in listing)  # hidden file skipped
    photo = ws.read_file("vacation_photo.png")
    assert "mime type: image/png" in photo and "image metadata" in photo  # no pixel dump
    assert "truncated" in ws.read_file("tax_return_2023.txt")
    assert "first 5 rows" in ws.read_file("bank_statement_march.csv")
    assert "Downloads/" in listing  # a non-category directory, nested one level deeper too
    assert "Downloads/old stuff/" in ws.list_files("Downloads")


@pytest.mark.parametrize(("path", "label", "snippet"), [
    ("electricity_bill_feb_2024.pdf", "PDF text", "Total due 104.98 EUR"),
    ("cv_leo_gozman.docx", "DOCX text", "Senior Backend Engineer"),
    ("policy_documents.doc", "DOC text", "Comprehensive cover"),  # long: found by the middle sample
    ("budget_2024.xlsx", "XLSX header", "Rent,1100"),
    ("onboarding.ppt", "PPT slide text", "[slide 2]\nYour first week"),
    ("IMG_4410.jpg", "image metadata", "gps: 38.69143, -9.20997"),
    ("Downloads/scan0003.pdf", "PDF text", "Certificate of Vaccination"),
    ("Downloads/attachment (1).docx", "DOCX text", "rental deposit"),
    ("Downloads/team_offsite.xls", "XLS header", "2024-06-13,11:00,Roadmap H2"),
    ("Downloads/old stuff/Document1.doc", "DOC text", "Transcript of Records"),
    ("Downloads/old stuff/presentation.ppt", "PPT slide text", "[notes]\nExam question"),
    ("Downloads/old stuff/export.xlsx", "XLSX header", "started ramipril"),
])
def test_sample_has_one_example_per_document_format(path: str, label: str, snippet: str):
    out = Workspace(DEMO / "sample").read_file(path)
    assert f"[content: {label}" in out and snippet in out


def test_reset_then_check_reports_unorganized_inbox(demo, capsys):
    assert demo.reset() == 0
    assert sample_files() <= {
        p.relative_to(demo.INBOX).as_posix() for p in demo.INBOX.rglob("*") if p.is_file()
    }
    assert demo.check() == 1  # nothing organized yet


def test_check_passes_for_ideal_layout(demo):
    demo.reset()
    for name, allowed in demo.load_expected().items():
        target = allowed[0]
        src = demo.INBOX / name
        if target and src.parent.name != target:
            (demo.INBOX / target).mkdir(exist_ok=True)
            shutil.move(src, demo.INBOX / target / src.name)
    assert demo.check() == 1  # the emptied Downloads/ tree is reported as leftover
    shutil.rmtree(demo.INBOX / "Downloads")
    assert demo.check() == 0
