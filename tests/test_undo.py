"""Undoing the changes of a session (--undo), from the decision log in the store."""

from functools import partial
from pathlib import Path

import pytest

from file_organizer import cli
from file_organizer.cli import main, run, undo
from file_organizer.workspace import Workspace

from .conftest import ScriptedChatModel, ai, call
from .test_resume import answers


@pytest.fixture
def state_db(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("state") / "organizer.sqlite"


def organize(folder: Path, state_db: Path, *replies: str) -> int:
    """A session: create finance/2024 (two directories), move two files, reject one."""
    model = ScriptedChatModel(responses=[
        ai(call("create_directory", path="finance/2024"), call("create_directory", path="junk")),
        ai(call("move_file", source="invoice.txt", destination="finance/2024"),
           call("move_file", source="flight.txt", destination="finance")),
        ai(content="Done."),
    ])
    return run(folder, model, input_fn=answers(*replies), output_fn=lambda *_: None, state_db=state_db)


def listing(folder: Path) -> list[str]:
    return sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*"))


def test_undo_restores_the_folder_after_showing_the_plan(folder: Path, state_db: Path):
    before = listing(folder)
    assert organize(folder, state_db, "y", "n", "all") == 0
    assert (folder / "finance" / "2024" / "invoice.txt").exists()

    output: list[str] = []
    fake = answers("y")
    assert undo(folder, input_fn=fake, output_fn=output.append, state_db=state_db) == 0
    assert listing(folder) == before
    assert output[output.index("\nUndo plan (4 steps, newest change first):") + 1:][:4] == [
        "    1. Move 'finance/flight.txt' back to 'flight.txt'",
        "    2. Move 'finance/2024/invoice.txt' back to 'invoice.txt'",
        "    3. Remove directory 'finance/2024' (if empty)",
        "    4. Remove directory 'finance' (if empty)",
    ]
    # the log is used up: a second undo has nothing to do
    output.clear()
    assert undo(folder, input_fn=answers(), output_fn=output.append, state_db=state_db) == 0
    assert any("Nothing to undo" in line for line in output)


def test_declining_changes_nothing(folder: Path, state_db: Path):
    organize(folder, state_db, "y", "n", "all")
    after = listing(folder)
    assert undo(folder, input_fn=answers("n"), output_fn=lambda *_: None, state_db=state_db) == 0
    assert listing(folder) == after


def test_conflicts_are_skipped_and_can_be_retried(folder: Path, state_db: Path):
    organize(folder, state_db, "y", "n", "all")
    (folder / "invoice.txt").write_text("a new file with the old name")
    (folder / "finance" / "2024" / "notes.txt").write_text("added later")

    output: list[str] = []
    assert undo(folder, input_fn=answers("y"), output_fn=output.append, state_db=state_db) == 1
    assert (folder / "flight.txt").exists()  # the other steps went ahead
    assert (folder / "invoice.txt").read_text() == "a new file with the old name"  # not overwritten
    assert (folder / "finance" / "2024" / "notes.txt").exists()
    assert any("refusing to overwrite" in line for line in output)
    assert any("not empty" in line for line in output)

    # fix the conflicts, retry: only the remaining steps are planned
    (folder / "invoice.txt").rename(folder / "elsewhere.txt")
    (folder / "finance" / "2024" / "notes.txt").unlink()
    output.clear()
    assert undo(folder, input_fn=answers("y"), output_fn=output.append, state_db=state_db) == 0
    assert (folder / "invoice.txt").read_text().startswith("Invoice")
    assert not (folder / "finance").exists()
    assert "Undo plan (3 steps" in "\n".join(output)


def test_undo_discards_an_interrupted_session(folder: Path, state_db: Path):
    assert organize(folder, state_db, "y", "n") == 130  # Ctrl-C in the 2nd batch
    assert (folder / "finance" / "2024").is_dir()
    output: list[str] = []
    assert undo(folder, input_fn=answers("y"), output_fn=output.append, state_db=state_db) == 0
    assert any("interrupted session" in line for line in output)
    assert not (folder / "finance").exists()
    # nothing left to resume: the next run starts without asking
    model = ScriptedChatModel(responses=[ai(content="Fresh start.")])
    assert run(folder, model, input_fn=answers(), output_fn=lambda *_: None, state_db=state_db) == 0


def test_main_undo_needs_no_model(folder: Path, state_db: Path, monkeypatch):
    organize(folder, state_db, "y", "n", "all")
    monkeypatch.setattr(cli, "undo", partial(cli.undo, input_fn=answers("y")))

    def factory(name):  # pragma: no cover - must not be reached
        raise AssertionError("no model for --undo")

    assert main([str(folder), "--undo", "--state-db", str(state_db)], model_factory=factory) == 0
    assert (folder / "invoice.txt").exists()


@pytest.mark.parametrize("error", [KeyboardInterrupt, PermissionError])
def test_undo_stopped_half_way_can_be_retried(folder: Path, state_db: Path, monkeypatch, error):
    """Ctrl-C or an unexpected OSError after the first move: the move already undone
    must not be retried (its record is gone), so a second --undo finishes cleanly."""
    before = listing(folder)
    organize(folder, state_db, "y", "n", "all")
    real_undo_move = Workspace.undo_move
    calls = []

    def flaky_undo_move(self, moved_to, original):
        calls.append(moved_to)
        if len(calls) == 2:
            raise error
        return real_undo_move(self, moved_to, original)

    monkeypatch.setattr(Workspace, "undo_move", flaky_undo_move)
    if error is KeyboardInterrupt:
        assert undo(folder, input_fn=answers("y"), output_fn=lambda *_: None, state_db=state_db) == 130
    else:
        with pytest.raises(PermissionError):
            undo(folder, input_fn=answers("y"), output_fn=lambda *_: None, state_db=state_db)
    assert (folder / "flight.txt").exists()  # the first step was done
    monkeypatch.setattr(Workspace, "undo_move", real_undo_move)

    output: list[str] = []
    assert undo(folder, input_fn=answers("y"), output_fn=output.append, state_db=state_db) == 0
    assert "Undo plan (3 steps" in "\n".join(output)
    assert listing(folder) == before
