"""LangChain tools the model uses to inspect and reorganize the folder.

Tools are a LangChain concept (LangGraph has no tool type of its own): each one bundles
a name, a description and a JSON argument schema (sent to the model via bind_tools) with
the Python function to run. The graph's tools node looks them up by name and invokes them.
"""

from __future__ import annotations

from collections.abc import Callable

from langchain_core.tools import BaseTool, StructuredTool

from .workspace import Workspace

#: Tools that change the filesystem and therefore require human approval.
MUTATING_TOOLS = frozenset({"create_directory", "move_file"})


def _format_tool_error(error: Exception) -> str:
    """Report a failed tool call (e.g. a WorkspaceError) back to the model."""
    return f"Error: {error}"


def _format_validation_error(error: Exception) -> str:
    """Report malformed arguments (missing/wrong-typed fields) back to the model."""
    return f"Error: invalid arguments: {error}"


def _to_tool(fn: Callable[..., str]) -> BaseTool:
    """Wrap a function as a tool whose failures become error results, not crashes.

    from_function takes the tool name from the function, the description from its
    docstring, and the argument schema from its type hints. WorkspaceError is a
    ToolException, so it is caught by handle_tool_error; the resulting "Error: ..."
    tool result (status="error") lets the model see what went wrong and recover.
    """
    return StructuredTool.from_function(
        fn,
        handle_tool_error=_format_tool_error,
        handle_validation_error=_format_validation_error,
    )


def make_tools(workspace: Workspace) -> list[BaseTool]:
    """Build the four tools, all bound to one sandboxed workspace."""

    def list_files(directory: str = ".") -> str:
        """List the files and sub-directories of a directory (relative to the folder being
        organized; defaults to its root). Directories end with '/'."""
        entries = workspace.list_files(directory)
        return "\n".join(entries) if entries else f"(directory '{directory}' is empty)"

    def read_file(path: str) -> str:
        """Read the text content of a file (relative path). Long files are truncated;
        binary files are reported as such."""
        # Delimit the content so the model can tell untrusted file data from instructions.
        return f"<file path={path!r}>\n{workspace.read_file(path)}\n</file>"

    def create_directory(path: str) -> str:
        """Create a directory (relative path, parents included). Requires human approval."""
        return workspace.create_directory(path)

    def move_file(source: str, destination: str) -> str:
        """Move a file into an existing directory (relative path), keeping its name. Files
        are never renamed or overwritten. Requires human approval."""
        return workspace.move_file(source, destination)

    return [
        _to_tool(list_files),
        _to_tool(read_file),
        _to_tool(create_directory),
        _to_tool(move_file),
    ]
