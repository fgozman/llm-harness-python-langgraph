"""Saving state in SQLite and resuming an interrupted session in a later run."""

from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage

from file_organizer.cli import main, run

from .conftest import ScriptedChatModel, ai, call


def answers(*replies: str):
    """Fake input(): returns the replies in order, then simulates Ctrl-C."""
    queue = list(replies)

    def fake_input(prompt: str = "") -> str:
        if not queue:
            raise KeyboardInterrupt
        return queue.pop(0)

    return fake_input


@pytest.fixture
def state_db(tmp_path: Path) -> Path:
    return tmp_path / "state" / "organizer.sqlite"


def organize(folder, model, state_db, *replies, output: list[str] | None = None):
    output = [] if output is None else output
    return run(folder, model, input_fn=answers(*replies), output_fn=output.append,
               state_db=state_db)


def human_messages(model: ScriptedChatModel) -> list[str]:
    return [m.content for m in model.received[0] if isinstance(m, HumanMessage)]


def test_resumes_after_a_crash_during_the_model_call(folder: Path, state_db: Path):
    crashing = ScriptedChatModel(responses=[ai(call("list_files"))])  # 2nd call raises
    output: list[str] = []
    assert organize(folder, crashing, state_db, output=output) == 1
    assert any("run the same command again" in line for line in output)

    model = ScriptedChatModel(responses=[ai(content="Resumed and done.")])
    assert organize(folder, model, state_db, "y") == 0  # "y" = resume

    sent = model.received[0]
    assert len(human_messages(model)) == 1  # the original request, not sent twice
    assert "invoice.txt" in sent[-1].content  # the list_files result from the first run


def test_resumes_a_pending_approval_without_reusing_old_answers(folder: Path, state_db: Path):
    batch = ai(
        call("create_directory", path="x"),
        call("create_directory", path="y"),
        call("create_directory", path="z"),
    )
    # Run 1: approve x, then Ctrl-C while being asked about y.
    assert organize(folder, ScriptedChatModel(responses=[batch]), state_db, "y") == 130
    assert not (folder / "x").exists()  # nothing runs until the whole batch is decided

    # Run 2: resume; asked about y again (not x), reject it, approve z.
    output: list[str] = []
    model = ScriptedChatModel(responses=[ai(content="done")])
    assert organize(folder, model, state_db, "y", "n", "y", output=output) == 0

    questions = [line for line in output if line.startswith("\n? ")]
    assert questions == ["\n? Create directory 'y'", "\n? Create directory 'z'"]
    assert (folder / "x").is_dir()  # approved in run 1
    assert not (folder / "y").exists()  # rejected after the restart
    assert (folder / "z").is_dir()


def test_declining_to_resume_starts_over(folder: Path, state_db: Path):
    organize(folder, ScriptedChatModel(responses=[ai(call("list_files"))]), state_db)

    model = ScriptedChatModel(responses=[ai(content="Fresh start.")])
    assert organize(folder, model, state_db, "n") == 0  # "n" = don't resume

    assert len(model.received[0]) == 2  # just the system prompt + a new request


def test_finished_session_leaves_nothing_to_resume(folder: Path, state_db: Path):
    assert organize(folder, ScriptedChatModel(responses=[ai(content="done")]), state_db) == 0

    # No "resume?" question: the only scripted reply is never needed.
    model = ScriptedChatModel(responses=[ai(content="again")])
    output: list[str] = []
    assert organize(folder, model, state_db, output=output) == 0
    assert "again" in "\n".join(output)


def test_each_folder_has_its_own_saved_session(folder: Path, tmp_path: Path, state_db: Path):
    other = tmp_path / "other"
    other.mkdir()
    organize(folder, ScriptedChatModel(responses=[ai(call("list_files"))]), state_db)  # crashes

    # The interrupted session belongs to `folder`, so `other` starts without asking.
    assert organize(other, ScriptedChatModel(responses=[ai(content="ok")]), state_db) == 0


def test_main_uses_the_state_db_option(folder: Path, tmp_path: Path, monkeypatch):
    db = tmp_path / "custom.sqlite"
    monkeypatch.setenv("FILE_ORGANIZER_STATE_DB", str(tmp_path / "from-env.sqlite"))
    main([str(folder), "--state-db", str(db)],
         model_factory=lambda name: ScriptedChatModel(responses=[ai(content="ok")]))
    assert db.exists()
    assert not (tmp_path / "from-env.sqlite").exists()
