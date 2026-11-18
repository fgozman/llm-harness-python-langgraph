"""Tool-calling agent that sorts a folder's files into category directories, with
human approval of every filesystem change."""

import logging

from .graph import build_graph
from .memory import ContextPolicy, DecisionLog
from .persistence import sqlite_checkpointer, sqlite_store
from .tools import MUTATING_TOOLS, make_tools
from .workspace import Workspace, WorkspaceError

# Library best practice: the package never configures logging itself. NullHandler keeps
# its log records silent unless the application (e.g. our CLI) sets up handlers.
logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "MUTATING_TOOLS",
    "ContextPolicy",
    "DecisionLog",
    "Workspace",
    "WorkspaceError",
    "build_graph",
    "make_tools",
    "sqlite_checkpointer",
    "sqlite_store",
]
