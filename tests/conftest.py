from __future__ import annotations

import itertools
import logging
import sqlite3
from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from file_organizer.persistence import PruningSqliteSaver
from file_organizer.workspace import Workspace

_ids = itertools.count()


def call(name: str, **args: Any) -> dict[str, Any]:
    """Build a tool call as it would appear on an AIMessage.

    The unique "id" matters: the graph keys approval decisions by it, and each
    ToolMessage answers its call through tool_call_id.
    """
    return {"name": name, "args": args, "id": f"call_{next(_ids)}", "type": "tool_call"}


def memory_sqlite_saver() -> PruningSqliteSaver:
    """The same SQLite checkpointer as in production, on a private in-memory database.

    Tests exercise the real storage and serialization without writing files.
    (langgraph.checkpoint.memory.InMemorySaver would also work for most tests.)
    """
    return PruningSqliteSaver(sqlite3.connect(":memory:", check_same_thread=False))


def ai(*tool_calls: dict[str, Any], content: str = "") -> AIMessage:
    return AIMessage(content=content, tool_calls=list(tool_calls))


class ScriptedChatModel(BaseChatModel):
    """Chat model that replays scripted responses and records what it was sent.

    Subclassing BaseChatModel makes it a drop-in replacement for a real LLM in the
    graph: LangChain's invoke() calls _generate() below.
    """

    responses: list[AIMessage]
    received: list[list[BaseMessage]] = []
    bound_tools: list[Any] = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):  # noqa: ANN001
        # build_graph calls this to advertise the tools; a real model would return a
        # wrapped runnable. Here we just record the tools so tests can check them.
        self.bound_tools = list(tools)
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):  # noqa: ANN001
        # The one method a BaseChatModel must implement: messages in, ChatResult out.
        self.received.append(list(messages))
        if not self.responses:
            raise AssertionError("ScriptedChatModel ran out of responses")
        return ChatResult(generations=[ChatGeneration(message=self.responses.pop(0))])


@pytest.fixture(autouse=True)
def _isolated_cwd(tmp_path_factory, monkeypatch):
    """Keep the developer's real .env (and tracing settings) out of the tests."""
    monkeypatch.chdir(tmp_path_factory.mktemp("cwd"))
    monkeypatch.delenv("FILE_ORGANIZER_PHOENIX", raising=False)


@pytest.fixture(autouse=True)
def _restore_logging():
    """configure_logging() (called by main) changes global logging state; undo it."""
    root, app = logging.getLogger(), logging.getLogger("file_organizer")
    saved = list(root.handlers), root.level, app.level
    yield
    root.handlers[:] = saved[0]
    root.setLevel(saved[1])
    app.setLevel(saved[2])


@pytest.fixture
def folder(tmp_path: Path) -> Path:
    (tmp_path / "invoice.txt").write_text("Invoice #42: total due 120 EUR")
    (tmp_path / "flight.txt").write_text("Boarding pass LH123 to Lisbon")
    (tmp_path / "bloodwork.txt").write_text("Lab results: cholesterol normal")
    return tmp_path


@pytest.fixture
def workspace(folder: Path) -> Workspace:
    return Workspace(folder)
