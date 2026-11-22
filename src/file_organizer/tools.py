"""LangChain tools the model uses to inspect and reorganize the folder.

Tools are a LangChain concept (LangGraph has no tool type of its own): each one bundles
a name, a description and a JSON argument schema (sent to the model via bind_tools) with
the Python function to run. The graph's tools node looks them up by name and invokes them.
"""

from __future__ import annotations

import secrets
from collections.abc import Callable
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool

from .workspace import Change, Workspace

#: Tools that change the filesystem and therefore require human approval.
MUTATING_TOOLS = frozenset({"create_directory", "move_file"})


def _format_tool_error(error: Exception) -> str:
    """Report a failed tool call (e.g. a WorkspaceError) back to the model."""
    return f"Error: {error}"


def _format_validation_error(error: Exception) -> str:
    """Report malformed arguments (missing/wrong-typed fields) back to the model."""
    return f"Error: invalid arguments: {error}"


def wrap_file_content(path: str, content: str) -> str:
    """Delimit untrusted file content with tags the file can't forge.

    A fixed ``</file>`` could simply be written inside a file to "end" the content
    early and smuggle instructions after it. The tag name carries a random id instead
    (``<file-3f9a...>`` ... ``</file-3f9a...>``), new for every read, so only the real
    closing tag ends the content; the system prompt tells the model so.
    """
    while True:
        marker = f"file-{secrets.token_hex(8)}"
        if marker not in content and marker not in path:  # practically always true
            return f"<{marker} path={path!r}>\n{content}\n</{marker}>"


def _to_tool(fn: Callable[..., Any], *, description: str | None = None,
             with_artifact: bool = False) -> BaseTool:
    """Wrap a function as a tool whose failures become error results, not crashes.

    from_function takes the tool name from the function, the description from its
    docstring, and the argument schema from its type hints. WorkspaceError is a
    ToolException, so it is caught by handle_tool_error; the resulting "Error: ..."
    tool result (status="error") lets the model see what went wrong and recover.

    ``with_artifact``: ``fn`` returns (content, artifact). The model sees only the
    content; the artifact (here: how to undo the change) ends up on the ToolMessage's
    ``.artifact`` when the tool is invoked with a whole tool call.
    """
    return StructuredTool.from_function(
        fn,
        description=description,
        response_format="content_and_artifact" if with_artifact else "content",
        handle_tool_error=_format_tool_error,
        handle_validation_error=_format_validation_error,
    )


def _with_undo(change: Change) -> tuple[str, dict[str, Any] | None]:
    return change.message, change.undo


def make_tools(workspace: Workspace) -> list[BaseTool]:
    """Build the four tools, all bound to one sandboxed workspace."""

    def list_files(directory: str = ".") -> str:
        """List the files and sub-directories of a directory (relative to the folder being
        organized; defaults to its root). Directories end with '/'."""
        entries = workspace.list_files(directory)
        return "\n".join(entries) if entries else f"(directory '{directory}' is empty)"

    def read_file(path: str, offset: int | None = None, limit: int | None = None) -> str:
        """Read a file (relative path). Returns its metadata (name, extension, MIME type,
        size, modification time) first, then its content: text of text/PDF/DOC/DOCX
        files, slide text and notes of PPT files, the header and first 5 rows of a CSV or
        of each XLS/XLSX sheet, EXIF tags of an image; other binary files get metadata
        only. Long text shows only samples of its beginning, middle and end; to read more,
        pass offset (character position to start at) and optionally limit (characters to
        return, capped per call)."""
        # Delimit the content so the model can tell untrusted file data from instructions.
        return wrap_file_content(path, workspace.read_file(path, offset, limit))

    def create_directory(path: str) -> tuple[str, dict[str, Any] | None]:
        """Create a directory (relative path, parents included). Requires human approval."""
        return _with_undo(workspace.create_directory_change(path))

    def move_file(source: str, destination: str) -> tuple[str, dict[str, Any] | None]:
        return _with_undo(workspace.move_file_change(source, destination))

    if workspace.rename_on_clash:
        move_description = (
            "Move a file into an existing directory (relative path). If a file with the same "
            "name is already there, the moved file gets a free name like 'name (2).ext'; "
            "nothing is ever overwritten. Requires human approval.")
    else:
        move_description = (
            "Move a file into an existing directory (relative path), keeping its name. Files "
            "are never renamed or overwritten. Requires human approval.")

    return [
        _to_tool(list_files),
        _to_tool(read_file),
        _to_tool(create_directory, with_artifact=True),
        _to_tool(move_file, description=move_description, with_artifact=True),
    ]
