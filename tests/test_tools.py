"""Tests for the LangChain tool wrappers exposed to the model."""

from pathlib import Path

from file_organizer.tools import MUTATING_TOOLS, make_tools
from file_organizer.workspace import Workspace


def by_name(workspace: Workspace):
    return {t.name: t for t in make_tools(workspace)}


def test_exposes_exactly_the_four_tools(workspace: Workspace):
    assert set(by_name(workspace)) == {"list_files", "read_file", "create_directory", "move_file"}


def test_only_filesystem_changes_are_marked_mutating():
    assert MUTATING_TOOLS == {"create_directory", "move_file"}


def test_every_tool_has_a_description(workspace: Workspace):
    assert all(t.description for t in make_tools(workspace))


def test_list_files_defaults_to_root(workspace: Workspace):
    out = by_name(workspace)["list_files"].invoke({})
    assert out.splitlines() == ["bloodwork.txt", "flight.txt", "invoice.txt"]


def test_list_files_reports_empty_directory(workspace: Workspace, folder: Path):
    (folder / "empty").mkdir()
    assert "empty" in by_name(workspace)["list_files"].invoke({"directory": "empty"}).lower()


def test_read_file(workspace: Workspace):
    assert "Invoice #42" in by_name(workspace)["read_file"].invoke({"path": "invoice.txt"})


def test_create_directory_and_move_file(workspace: Workspace, folder: Path):
    tools = by_name(workspace)
    tools["create_directory"].invoke({"path": "finance"})
    tools["move_file"].invoke({"source": "invoice.txt", "destination": "finance"})
    assert (folder / "finance" / "invoice.txt").exists()


def test_workspace_errors_are_returned_to_the_model_not_raised(workspace: Workspace):
    out = by_name(workspace)["read_file"].invoke({"path": "../../etc/passwd"})
    assert out.startswith("Error:")
    assert "outside" in out


def test_malformed_arguments_are_returned_to_the_model_not_raised(workspace: Workspace):
    tools = {t.name: t for t in make_tools(workspace)}
    msg = tools["move_file"].invoke(
        {"type": "tool_call", "name": "move_file", "args": {"source": "x"}, "id": "1"}
    )
    assert msg.status == "error" and "invalid arguments" in msg.content
