"""Tests for the agent graph: tool loop + human approval of every filesystem change."""

from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.types import Command

from file_organizer.graph import build_graph
from file_organizer.persistence import sqlite_checkpointer
from file_organizer.workspace import Workspace

from .conftest import ScriptedChatModel, ai, call, memory_sqlite_saver

# Every invoke in a test uses the same thread_id, so resumes continue the run that the
# checkpointer saved. Each test builds a new graph with its own in-memory SQLite database,
# so tests can't see each other's threads.
CONFIG = {"configurable": {"thread_id": "test"}}


def make(workspace: Workspace, *responses):
    model = ScriptedChatModel(responses=list(responses))
    return build_graph(model, workspace, checkpointer=memory_sqlite_saver()), model


def start(graph, text: str = "Organize this folder."):
    # invoke() runs until END or the first interrupt and returns the final state values.
    return graph.invoke({"messages": [HumanMessage(text)]}, CONFIG)


def interrupts(result) -> list:
    # A run paused by interrupt() returns its state plus an "__interrupt__" key holding
    # Interrupt objects; .value is the payload the review node passed to interrupt().
    return [i.value for i in result.get("__interrupt__", [])]


def tool_messages(result) -> list[ToolMessage]:
    return [m for m in result["messages"] if isinstance(m, ToolMessage)]


def test_binds_the_tools_and_sends_system_prompt(workspace: Workspace):
    graph, model = make(workspace, ai(content="Nothing to do."))
    start(graph)
    assert {t.name for t in model.bound_tools} == {
        "list_files", "read_file", "create_directory", "move_file",
    }
    first_prompt = model.received[0]
    assert isinstance(first_prompt[0], SystemMessage)
    assert "content" in first_prompt[0].content.lower()


def test_finishes_when_model_stops_calling_tools(workspace: Workspace):
    graph, _ = make(workspace, ai(content="All done."))
    result = start(graph)
    assert result["messages"][-1].content == "All done."
    assert interrupts(result) == []


def test_read_only_tools_run_without_approval(workspace: Workspace):
    graph, model = make(
        workspace,
        ai(call("list_files"), call("read_file", path="invoice.txt")),
        ai(content="Looked around."),
    )
    result = start(graph)
    assert interrupts(result) == []
    contents = [m.content for m in tool_messages(result)]
    assert "invoice.txt" in contents[0]
    assert "Invoice #42" in contents[1]
    # the tool results were fed back to the model
    assert isinstance(model.received[1][-1], ToolMessage)


def test_create_directory_waits_for_approval(workspace: Workspace, folder: Path):
    graph, _ = make(workspace, ai(call("create_directory", path="finance")), ai(content="done"))
    result = start(graph)

    [request] = interrupts(result)
    assert request["tool"] == "create_directory"
    assert request["args"] == {"path": "finance"}
    assert "finance" in request["description"]
    assert not (folder / "finance").exists(), "must not touch the disk before approval"

    # Resume the paused thread. This dict is a plain value (its keys are not interrupt
    # ids), so it becomes the return value of the single pending interrupt() call.
    result = graph.invoke(Command(resume={"approved": True}), CONFIG)
    assert (folder / "finance").is_dir()
    assert "Created directory" in tool_messages(result)[-1].content
    assert result["messages"][-1].content == "done"


def test_rejected_change_is_not_applied_and_model_is_told_why(workspace: Workspace, folder: Path):
    graph, model = make(
        workspace,
        ai(call("move_file", source="invoice.txt", destination="flight.txt")),
        ai(content="ok"),
    )
    start(graph)
    result = graph.invoke(
        Command(resume={"approved": False, "feedback": "invoices belong in finance"}), CONFIG
    )

    assert (folder / "invoice.txt").exists()
    msg = tool_messages(result)[-1]
    assert "rejected" in msg.content.lower()
    assert "invoices belong in finance" in msg.content
    assert msg in model.received[-1]


def test_boolean_resume_values_are_accepted(workspace: Workspace, folder: Path):
    graph, _ = make(workspace, ai(call("create_directory", path="travel")), ai(content="ok"))
    start(graph)
    graph.invoke(Command(resume=True), CONFIG)
    assert (folder / "travel").is_dir()


def test_each_change_in_a_batch_is_approved_separately(workspace: Workspace, folder: Path):
    graph, _ = make(
        workspace,
        ai(
            call("create_directory", path="finance"),
            call("read_file", path="flight.txt"),
            call("create_directory", path="junk"),
            call("create_directory", path="health"),
        ),
        ai(content="ok"),
    )
    # Each Command(resume=...) answers the pending interrupt. The review node then
    # re-runs, replays the earlier answers, and pauses at the next change.
    first = interrupts(start(graph))
    second = interrupts(graph.invoke(Command(resume=True), CONFIG))
    third = interrupts(graph.invoke(Command(resume=False), CONFIG))
    result = graph.invoke(Command(resume=True), CONFIG)

    assert [r[0]["args"]["path"] for r in (first, second, third)] == ["finance", "junk", "health"]
    assert (folder / "finance").is_dir()
    assert not (folder / "junk").exists()
    assert (folder / "health").is_dir()
    # one result per tool call, in the original order
    assert [m.content.split()[0][:5] for m in tool_messages(result)] == [
        "Creat", "<file", "Rejec", "Creat",
    ]


def test_failing_approved_change_reports_error_to_model(workspace: Workspace):
    graph, _ = make(
        workspace, ai(call("move_file", source="ghost.txt", destination="x")), ai(content="ok")
    )
    start(graph)
    result = graph.invoke(Command(resume=True), CONFIG)
    assert tool_messages(result)[-1].content.startswith("Error:")
    assert tool_messages(result)[-1].status == "error"


def test_unknown_tool_reports_error_to_model(workspace: Workspace):
    graph, _ = make(workspace, ai(call("delete_everything")), ai(content="ok"))
    result = start(graph)
    assert interrupts(result) == []
    assert "unknown tool" in tool_messages(result)[-1].content.lower()


def test_full_organizing_session(workspace: Workspace, folder: Path):
    graph, _ = make(
        workspace,
        ai(call("list_files")),
        ai(
            call("read_file", path="bloodwork.txt"),
            call("read_file", path="flight.txt"),
            call("read_file", path="invoice.txt"),
        ),
        ai(call("create_directory", path="finance"), call("create_directory", path="travel")),
        ai(
            call("move_file", source="invoice.txt", destination="finance"),
            call("move_file", source="flight.txt", destination="travel"),
        ),
        ai(content="Organized 2 files."),
    )
    result = start(graph)
    approvals = 0
    while interrupts(result):
        approvals += 1
        result = graph.invoke(Command(resume=True), CONFIG)

    assert approvals == 4
    assert (folder / "finance" / "invoice.txt").exists()
    assert (folder / "travel" / "flight.txt").exists()
    assert (folder / "bloodwork.txt").exists()
    assert result["messages"][-1].content == "Organized 2 files."


@pytest.mark.parametrize("bad", [False, "yes", 1, {"feedback": "x"}, {"approved": "true"}])
def test_ambiguous_resume_values_count_as_rejection(workspace: Workspace, folder: Path, bad):
    graph, _ = make(workspace, ai(call("create_directory", path="finance")), ai(content="ok"))
    start(graph)
    graph.invoke(Command(resume=bad), CONFIG)
    assert not (folder / "finance").exists()


def test_approved_move_to_a_missing_directory_does_not_rename(workspace: Workspace, folder: Path):
    # Regression: an approved move to a non-existent "directory" used to rename the file.
    graph, model = make(
        workspace, ai(call("move_file", source="invoice.txt", destination="ghost")), ai(content="ok")
    )
    start(graph)
    result = graph.invoke(Command(resume=True), CONFIG)

    assert (folder / "invoice.txt").exists()
    assert not (folder / "ghost").exists()
    error = tool_messages(result)[-1]
    assert error.status == "error" and "create it first" in error.content
    assert model.received[-1][-1] == error  # the model is told to create the directory


def test_a_checkpointer_is_required(workspace: Workspace):
    with pytest.raises(TypeError, match="checkpointer"):
        build_graph(ScriptedChatModel(responses=[]), workspace)


def test_paused_run_survives_reopening_the_sqlite_file(workspace: Workspace, folder: Path, tmp_path: Path):
    db = tmp_path / "state" / "graph.sqlite"  # parent folder is created on demand
    with sqlite_checkpointer(db) as saver:
        graph = build_graph(ScriptedChatModel(responses=[ai(call("create_directory", path="finance"))]),
                            workspace, checkpointer=saver)
        start(graph)  # pauses for approval

    with sqlite_checkpointer(db) as saver:  # e.g. a new process
        # A brand-new graph object: everything it knows about the paused run comes from
        # the checkpoint stored under CONFIG's thread_id.
        graph = build_graph(ScriptedChatModel(responses=[ai(content="done")]), workspace, checkpointer=saver)
        [pending] = graph.get_state(CONFIG).interrupts
        # Resume map {interrupt id: answer}: targets exactly this pending question.
        assert pending.value["description"] == "Create directory 'finance'"
        result = graph.invoke(Command(resume={pending.id: True}), CONFIG)

    assert (folder / "finance").is_dir()
    assert result["messages"][-1].content == "done"


def test_describe_escapes_control_characters():
    from file_organizer.approval import describe

    text = describe("move_file", {"source": "a.txt\n? Create directory 'x'\x1b[2K", "destination": "d"})
    assert "\n" not in text and "\x1b" not in text


def test_every_question_carries_the_whole_batch(workspace: Workspace):
    graph, _ = make(
        workspace,
        ai(call("create_directory", path="finance"), call("list_files"),
           call("move_file", source="invoice.txt", destination="finance")),
        ai(content="ok"),
    )
    first = interrupts(start(graph))[0]
    second = interrupts(graph.invoke(Command(resume=True), CONFIG))[0]
    plan = ["Create directory 'finance'", "Move 'invoice.txt' -> 'finance'"]
    assert [item["description"] for item in first["batch"]] == plan  # read-only calls left out
    assert (first["index"], second["index"]) == (0, 1)
    assert second["batch"] == first["batch"]


def test_rename_on_clash_is_announced_in_the_question(folder: Path):
    (folder / "finance").mkdir()
    (folder / "finance" / "invoice.txt").write_text("older invoice")
    graph, _ = make(
        Workspace(folder, rename_on_clash=True),
        ai(call("move_file", source="invoice.txt", destination="finance")),
        ai(content="ok"),
    )
    question = interrupts(start(graph))[0]
    assert question["description"] == (
        "Move 'invoice.txt' -> 'finance' (name taken there; will be saved as 'finance/invoice (2).txt')")
    result = graph.invoke(Command(resume=True), CONFIG)
    assert (folder / "finance" / "invoice (2).txt").read_text().startswith("Invoice #42")
    assert (folder / "finance" / "invoice.txt").read_text() == "older invoice"
    assert "renamed" in tool_messages(result)[-1].content


@pytest.mark.parametrize("first_answer, expected_note", [
    (True, " (name taken there; will be saved as 'finance/report (2).txt')"),
    (False, ""),  # the first move was rejected: the name stays free
])
def test_rename_on_clash_counts_earlier_moves_of_the_batch(folder: Path, first_answer, expected_note):
    for sub in ("a", "b", "finance"):
        (folder / sub).mkdir()
    (folder / "a" / "report.txt").write_text("report a")
    (folder / "b" / "report.txt").write_text("report b")
    graph, _ = make(
        Workspace(folder, rename_on_clash=True),
        ai(call("move_file", source="a/report.txt", destination="finance"),
           call("move_file", source="b/report.txt", destination="finance")),
        ai(content="ok"),
    )
    first = interrupts(start(graph))[0]
    emptied = " (then removes the emptied directory '{}')"
    assert first["description"] == "Move 'a/report.txt' -> 'finance'" + emptied.format("a")
    second = interrupts(graph.invoke(Command(resume=first_answer), CONFIG))[0]
    assert second["description"] == "Move 'b/report.txt' -> 'finance'" + expected_note + emptied.format("b")
    graph.invoke(Command(resume=True), CONFIG)
    if first_answer:
        assert (folder / "finance" / "report (2).txt").read_text() == "report b"
    else:
        assert (folder / "finance" / "report.txt").read_text() == "report b"


def _downloads(folder: Path) -> None:
    (folder / "docs").mkdir()
    (folder / "Downloads" / "old stuff").mkdir(parents=True)
    (folder / "Downloads" / "old stuff" / "a.txt").write_text("a")
    (folder / "Downloads" / "b.txt").write_text("b")


def _empty_downloads_batch():
    return ai(call("move_file", source="Downloads/old stuff/a.txt", destination="docs"),
              call("move_file", source="Downloads/b.txt", destination="docs"))


def test_directories_a_move_empties_are_shown_and_removed_only_when_approved(folder: Path):
    _downloads(folder)
    graph, _ = make(Workspace(folder), _empty_downloads_batch(), ai(content="ok"))
    first = interrupts(start(graph))[0]
    old = " (then removes the emptied directory 'Downloads/old stuff')"
    both = " (then removes the emptied directory 'Downloads')"
    # The plan assumes all are approved; each question counts the earlier answers.
    assert [i["description"] for i in first["batch"]] == [
        "Move 'Downloads/old stuff/a.txt' -> 'docs'" + old, "Move 'Downloads/b.txt' -> 'docs'" + both]
    assert first["description"] == first["batch"][0]["description"]
    second = interrupts(graph.invoke(Command(resume=True), CONFIG))[0]
    assert second["description"] == first["batch"][1]["description"]
    result = graph.invoke(Command(resume=True), CONFIG)
    assert not (folder / "Downloads").exists()
    moved = tool_messages(result)
    assert "Removed the now-empty directory 'Downloads/old stuff'." in moved[0].content
    assert moved[0].artifact["removed"] == ["Downloads/old stuff"]
    assert moved[1].artifact["removed"] == ["Downloads"]


def test_moving_into_an_ancestor_removes_only_the_emptied_subdirectory(folder: Path):
    (folder / "work" / "old").mkdir(parents=True)
    (folder / "work" / "old" / "a.txt").write_text("a")
    graph, _ = make(Workspace(folder), ai(call("move_file", source="work/old/a.txt", destination="work")),
                    ai(content="ok"))
    question = interrupts(start(graph))[0]
    assert question["description"] == (
        "Move 'work/old/a.txt' -> 'work' (then removes the emptied directory 'work/old')")
    result = graph.invoke(Command(resume=True), CONFIG)
    assert (folder / "work" / "a.txt").exists() and not (folder / "work" / "old").exists()
    assert "cleanup stopped" not in tool_messages(result)[0].content


def test_rejected_move_keeps_its_directory_and_its_parents(folder: Path):
    _downloads(folder)
    graph, _ = make(Workspace(folder), _empty_downloads_batch(), ai(content="ok"))
    start(graph)
    second = interrupts(graph.invoke(Command(resume=False), CONFIG))[0]
    # a.txt stays, so moving b.txt no longer empties Downloads/.
    assert second["description"] == "Move 'Downloads/b.txt' -> 'docs'"
    graph.invoke(Command(resume=True), CONFIG)
    assert (folder / "Downloads" / "old stuff" / "a.txt").exists()


def test_failed_directory_removal_is_reported_to_the_model(folder: Path, monkeypatch):
    _downloads(folder)
    graph, _ = make(Workspace(folder), _empty_downloads_batch(), ai(content="ok"))
    start(graph)
    graph.invoke(Command(resume=True), CONFIG)

    def no_permission(self):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(Path, "rmdir", no_permission)
    result = graph.invoke(Command(resume=True), CONFIG)
    moved = tool_messages(result)
    assert "could not remove 'Downloads/old stuff': Permission denied" in moved[0].content
    assert "removed" not in moved[0].artifact
    assert (folder / "docs" / "b.txt").exists()  # the moves themselves still happened
    assert "kept 'Downloads': it is not an empty directory" in moved[1].content  # old stuff/ remains
