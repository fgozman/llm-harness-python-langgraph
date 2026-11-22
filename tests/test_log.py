"""Tests for the background (queue-based) file logging."""

import io
import json
import logging
import re
from logging.handlers import QueueHandler
from pathlib import Path

from langgraph.types import Command

from file_organizer.cli import main
from file_organizer.graph import build_graph
from file_organizer.log import configure_logging, logging_config
from file_organizer.workspace import Workspace, WorkspaceError

from .conftest import ScriptedChatModel, ai, call, memory_sqlite_saver


def test_records_reach_the_file_after_stop(tmp_path: Path):
    log_file = tmp_path / "app.log"
    stop = configure_logging(log_file, "DEBUG")
    logging.getLogger("file_organizer.test").debug("hello from the app")
    logging.getLogger("some_library").info("library chatter")  # below WARNING: dropped
    logging.getLogger("some_library").warning("library warning")
    stop()

    text = log_file.read_text()
    assert "hello from the app" in text
    assert "library warning" in text
    assert "library chatter" not in text


def test_stop_detaches_the_handler_and_closes_the_file(tmp_path: Path):
    root = logging.getLogger()
    stop = configure_logging(tmp_path / "app.log")
    assert any(isinstance(h, QueueHandler) for h in root.handlers)
    stop()
    assert not any(isinstance(h, QueueHandler) for h in root.handlers)


def test_line_format_is_iso_utc_with_level_logger_and_location(tmp_path: Path):
    log_file = tmp_path / "app.log"
    stop = configure_logging(log_file)
    logging.getLogger("file_organizer.demo").info("formatted %s", "ok")
    stop()

    line = log_file.read_text().strip()
    assert re.fullmatch(
        r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z INFO     "
        r"file_organizer\.demo:test_line_format_is_iso_utc_with_level_logger_and_location:\d+ "
        r"- formatted ok",
        line,
    ), line


def test_file_rotates_and_existing_loggers_survive(tmp_path: Path):
    config = logging_config(tmp_path / "app.log", "INFO")
    assert config["handlers"]["file"]["class"] == "logging.handlers.RotatingFileHandler"
    assert config["handlers"]["file"]["maxBytes"] > 0
    assert config["disable_existing_loggers"] is False


def test_creates_missing_log_directory(tmp_path: Path):
    log_file = tmp_path / "nested" / "dir" / "app.log"
    stop = configure_logging(log_file)
    logging.getLogger("file_organizer").warning("x")
    stop()
    assert log_file.exists()


def test_main_logs_the_session_and_changes(folder: Path, tmp_path: Path, monkeypatch):
    log_file = tmp_path / "session.log"
    model = ScriptedChatModel(
        responses=[ai(call("create_directory", path="finance")), ai(content="All done.")]
    )
    monkeypatch.setattr("sys.stdin", io.StringIO("y\n"))  # approve the change

    code = main([str(folder), "--log-file", str(log_file)], model_factory=lambda name: model)

    assert code == 0
    text = log_file.read_text()
    for expected in ["Session", "started", "Model requested create_directory",
                     "User approved", "Created directory", "Model finished (9 chars)"]:
        assert expected in text, expected


def test_model_text_and_tool_arguments_are_only_logged_at_debug(folder: Path, tmp_path: Path,
                                                                monkeypatch):
    def run_at(level: str) -> str:
        log_file = tmp_path / f"{level}.log"
        model = ScriptedChatModel(responses=[
            ai(call("read_file", path="bloodwork.txt")), ai(content="Your blood test was fine."),
        ])
        monkeypatch.setattr("sys.stdin", io.StringIO(""))
        main([str(folder), "--log-file", str(log_file), "--log-level", level],
             model_factory=lambda name: model)
        return log_file.read_text()

    info = run_at("INFO")
    assert "Model requested read_file" in info and "bloodwork.txt" not in info
    assert "blood test" not in info
    debug = run_at("DEBUG")
    assert "read_file({'path': 'bloodwork.txt'})" in debug and "blood test" in debug


def test_main_logs_model_failures_with_traceback(folder: Path, tmp_path: Path):
    log_file = tmp_path / "session.log"

    def factory(name):
        raise ValueError("no such provider")

    assert main([str(folder), "--log-file", str(log_file)], model_factory=factory) == 2
    text = log_file.read_text()
    assert "Could not create model" in text and "Traceback" in text


def test_each_decision_is_logged_once_despite_review_replays(workspace: Workspace, caplog):
    caplog.set_level(logging.INFO, logger="file_organizer")
    model = ScriptedChatModel(
        responses=[
            ai(call("create_directory", path="a"), call("create_directory", path="b")),
            ai(content="ok"),
        ]
    )
    graph = build_graph(model, workspace, checkpointer=memory_sqlite_saver())
    config = {"configurable": {"thread_id": "t"}}
    # ("user", "go") is shorthand that add_messages converts into a HumanMessage.
    graph.invoke({"messages": [("user", "go")]}, config)
    graph.invoke(Command(resume=True), config)  # review re-runs on each resume
    graph.invoke(Command(resume=False), config)

    decisions = [r.getMessage() for r in caplog.records if r.getMessage().startswith("User ")]
    assert decisions == [
        "User approved: Create directory 'a' (feedback: None)",
        "User rejected: Create directory 'b' (feedback: None)",
    ]


def test_blocked_paths_are_logged(workspace: Workspace, caplog):
    try:
        workspace.read_file("../secret.txt")
    except WorkspaceError:
        pass
    assert any("Blocked path" in r.getMessage() for r in caplog.records)


def read_json_lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_json_format_writes_one_object_per_line(tmp_path: Path):
    log_file = tmp_path / "app.log"
    stop = configure_logging(log_file, fmt="json")
    logging.getLogger("file_organizer.demo").info("moved %s", "a.txt", extra={"file": "a.txt"})
    stop()

    [entry] = read_json_lines(log_file)
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z", entry.pop("timestamp"))
    assert entry == {
        "level": "INFO",
        "logger": "file_organizer.demo",
        "function": "test_json_format_writes_one_object_per_line",
        "line": entry["line"],
        "message": "moved a.txt",
        "file": "a.txt",  # from extra=
    }


def test_tracebacks_get_their_own_json_field(tmp_path: Path):
    log_file = tmp_path / "app.log"
    stop = configure_logging(log_file, fmt="json")
    try:
        raise ValueError("boom")
    except ValueError:
        logging.getLogger("file_organizer.demo").exception("it failed")
    stop()

    [entry] = read_json_lines(log_file)  # the multi-line traceback stays on one line
    assert entry["message"] == "it failed"
    assert "Traceback" in entry["exception"] and "ValueError: boom" in entry["exception"]


def test_text_format_still_appends_tracebacks(tmp_path: Path):
    log_file = tmp_path / "app.log"
    stop = configure_logging(log_file)
    try:
        raise ValueError("boom")
    except ValueError:
        logging.getLogger("file_organizer.demo").exception("it failed")
    stop()

    text = log_file.read_text()
    assert "- it failed\nTraceback" in text and "ValueError: boom" in text


def test_main_tags_every_session_record_with_run_and_thread_ids(folder: Path, tmp_path: Path, monkeypatch):
    log_file = tmp_path / "session.log"
    model = ScriptedChatModel(
        responses=[ai(call("list_files")), ai(call("create_directory", path="x")), ai(content="ok")]
    )
    monkeypatch.setattr("sys.stdin", io.StringIO("y\n"))

    main([str(folder), "--log-file", str(log_file), "--log-format", "JSON"],
         model_factory=lambda name: model)

    entries = read_json_lines(log_file)
    session = [e for e in entries if e["function"] != "main"]  # main logs before the session
    ids = {e.get("run_id") for e in session}
    assert len(session) > 3 and len(ids) == 1 and None not in ids
    assert {e.get("thread_id") for e in session} == {f"organize:{folder.resolve()}"}
    assert not any("trace_id" in e for e in entries)  # tracing is off
    assert any(e["logger"] == "file_organizer.workspace" for e in session)  # tool code too


def test_every_llm_call_and_its_tokens_are_logged(folder: Path, tmp_path: Path, monkeypatch):
    log_file = tmp_path / "session.jsonl"
    first = ai(call("list_files"))
    first.usage_metadata = {"input_tokens": 100, "output_tokens": 10, "total_tokens": 110}
    second = ai(content="Done.")
    second.usage_metadata = {"input_tokens": 200, "output_tokens": 20, "total_tokens": 220}
    model = ScriptedChatModel(responses=[first, second])

    code = main([str(folder), "--log-file", str(log_file), "--log-format", "json"],
                model_factory=lambda name: model)
    assert code == 0

    entries = [json.loads(line) for line in log_file.read_text().splitlines()]
    starts = [e for e in entries if e["message"].startswith("LLM call") and "sending" in e["message"]]
    done = [e for e in entries if e["message"].startswith("LLM call") and " done in " in e["message"]]
    assert [e["llm_call"] for e in starts] == [1, 2]
    assert [(e["llm_call"], e["input_tokens"], e["output_tokens"], e["total_tokens"]) for e in done] == [
        (1, 100, 10, 110), (2, 200, 20, 220),
    ]
    assert done[-1]["session_total_tokens"] == 330
    assert all(e["duration_s"] >= 0 for e in done)
    (total,) = [e for e in entries if e["message"].startswith("Session token usage")]
    assert total["session_input_tokens"] == 300 and total["session_output_tokens"] == 30
    assert total["session_llm_calls"] == 2
