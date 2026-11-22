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
    assert demo.check() == 0
