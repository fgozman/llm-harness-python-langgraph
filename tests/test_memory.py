"""Context management: summarization, the trimming safety net, the decision store, and
retention (checkpoint pruning, store TTL)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command

from file_organizer.cli import default_summary_model, run
from file_organizer.graph import build_graph
from file_organizer.memory import (
    ContextPolicy,
    DecisionLog,
    decisions_namespace,
    fit_context,
    is_summary,
    summary_gain,
)
from file_organizer.persistence import sqlite_checkpointer, sqlite_store
from file_organizer.workspace import Workspace

from .conftest import ScriptedChatModel, ai, call, memory_sqlite_saver

CONFIG = {"configurable": {"thread_id": "t"}}
# Tiny limits so a handful of small files triggers summarization.
SMALL = ContextPolicy(summarize_at_tokens=1500, max_input_tokens=3000)


@pytest.fixture
def big_folder(tmp_path: Path) -> Path:
    for i in range(6):
        (tmp_path / f"f{i}.txt").write_text(f"document {i} " * 200)  # ~500 tokens each
    return tmp_path


def notes(text: str = "NOTES: f0-f3 are documents", tokens: int = 0) -> AIMessage:
    msg = AIMessage(text)
    if tokens:
        msg.usage_metadata = {"input_tokens": tokens, "output_tokens": 10, "total_tokens": tokens + 10}
    return msg


def cheap_model(n: int = 10, tokens: int = 0) -> ScriptedChatModel:
    return ScriptedChatModel(responses=[notes(tokens=tokens) for _ in range(n)])


def reading_session(n_files: int = 6) -> list[AIMessage]:
    return [ai(call("read_file", path=f"f{i}.txt")) for i in range(n_files)] + [ai(content="done")]


def assert_valid_history(messages) -> None:
    """Every ToolMessage answers a tool call of an earlier AIMessage still in the list."""
    open_calls: set[str] = set()
    for m in messages:
        if isinstance(m, AIMessage):
            open_calls |= {tc["id"] for tc in m.tool_calls}
        elif isinstance(m, ToolMessage):
            assert m.tool_call_id in open_calls, "orphaned tool result"


# ---- summarization ---------------------------------------------------------------------


def test_no_summary_below_the_threshold(big_folder: Path):
    cheap = cheap_model()
    graph = build_graph(ScriptedChatModel(responses=reading_session(2)), Workspace(big_folder),
                        checkpointer=memory_sqlite_saver(), summary_model=cheap)  # default policy
    result = graph.invoke({"messages": [HumanMessage("go")]}, CONFIG)
    assert cheap.received == []
    assert not any(is_summary(m) for m in result["messages"])


def test_old_messages_are_replaced_by_a_summary_from_the_cheap_model(big_folder: Path):
    main, cheap = ScriptedChatModel(responses=reading_session()), cheap_model(tokens=100)
    graph = build_graph(main, Workspace(big_folder), checkpointer=memory_sqlite_saver(),
                        summary_model=cheap, policy=SMALL)
    result = graph.invoke({"messages": [HumanMessage("go")]}, CONFIG)

    assert cheap.received, "the cheap model wrote the summary"
    assert "file-organizing agent" in cheap.received[0][0].content  # our summary prompt
    messages = result["messages"]
    assert is_summary(messages[0]) and "NOTES" in messages[0].content
    assert len(messages) < 1 + 2 * 6 + 1  # history shrank in the state, not just in requests
    assert_valid_history(messages)
    # the main model continued from the summary, never from deleted messages
    last_request = main.received[-1]
    assert isinstance(last_request[0], SystemMessage) and is_summary(last_request[1])
    assert_valid_history(last_request)
    # usage of both models survives the deletion of the messages that carried it
    assert result["usage"]["calls"] == 7
    assert result["summary_usage"]["calls"] == len(cheap.received)
    assert result["summary_usage"]["input_tokens"] == 100 * len(cheap.received)


def test_summary_model_defaults_to_the_main_model(big_folder: Path):
    responses = reading_session(3)
    main = ScriptedChatModel(responses=[*responses[:-1], *[notes() for _ in range(5)], responses[-1]])
    graph = build_graph(main, Workspace(big_folder), checkpointer=memory_sqlite_saver(),
                        policy=ContextPolicy(summarize_at_tokens=1000))
    graph.invoke({"messages": [HumanMessage("go")]}, CONFIG)
    assert any("file-organizing agent" in str(r[0].content) for r in main.received)


@pytest.mark.parametrize(("main_model", "cheap"), [
    ("anthropic:claude-sonnet-4-5", "anthropic:claude-haiku-5-5"),
    ("openai:gpt-4.1", "openai:gpt-4.1-mini"),
    ("ollama:llama3", "ollama:llama3"),
])
def test_default_summary_model_is_a_cheap_one_of_the_same_provider(main_model, cheap):
    assert default_summary_model(main_model) == cheap


# ---- trimming safety net ---------------------------------------------------------------


def _conversation(n_turns: int) -> list:
    msgs = [SystemMessage("rules"), HumanMessage("organize")]
    for i in range(n_turns):
        tc = call("read_file", path=f"f{i}.txt")
        msgs += [AIMessage("", tool_calls=[tc]), ToolMessage("x " * 400, tool_call_id=tc["id"])]
    return msgs


def test_fit_context_leaves_small_requests_alone():
    msgs = _conversation(2)
    assert fit_context(msgs, 10_000) is msgs


def test_fit_context_first_shortens_old_tool_results():
    msgs = _conversation(10)  # ~200 tokens per tool result
    reduced = fit_context(msgs, 1_500)
    assert len(reduced) == len(msgs)  # nothing dropped
    assert reduced[-1] is msgs[-1]  # the newest results are sent in full
    for old in reduced[3:-1:2]:
        assert old.content.startswith("x x x") and "characters omitted from this request" in old.content
    assert all("omitted" not in m.content for m in msgs)  # the originals are untouched
    assert count_tokens_approximately(reduced) <= 1_500 < count_tokens_approximately(msgs)


def test_fit_context_then_drops_the_oldest_turns_and_says_so():
    msgs = _conversation(10)
    trimmed = fit_context(msgs, 600)
    assert "messages were left out of this request" in trimmed[0].content  # the model is told
    assert trimmed[1] is msgs[1]  # the task
    assert isinstance(trimmed[2], AIMessage)  # never starts with an orphaned tool result
    assert trimmed[-1] is msgs[-1]
    assert count_tokens_approximately(trimmed) <= 600
    assert_valid_history(trimmed)


def test_fit_context_sends_the_newest_turn_even_if_it_alone_is_too_big():
    msgs = _conversation(3)
    trimmed = fit_context(msgs, 50)
    assert trimmed[1:] == [msgs[1], *msgs[-2:]]
    assert "4 earlier messages were left out" in trimmed[0].content


def test_agent_trims_the_request_but_not_the_state(big_folder: Path):
    main = ScriptedChatModel(responses=[ai(*(call("read_file", path=f"f{i}.txt") for i in range(6))),
                                        ai(call("list_files")), ai(content="done")])
    graph = build_graph(main, Workspace(big_folder), checkpointer=memory_sqlite_saver(),
                        summary_model=cheap_model(), policy=SMALL)
    result = graph.invoke({"messages": [HumanMessage("go")]}, CONFIG)
    shortened = [m for m in main.received[-1] if "omitted from this request" in str(m.content)]
    assert len(shortened) == 6  # the request got previews of the old file contents...
    assert not any("omitted" in str(m.content) for m in result["messages"])  # ...the state the originals
    assert_valid_history(main.received[-1])


# ---- consistent limits, no repeated or pointless summaries -----------------------------


def test_policy_defaults_and_validation():
    policy = ContextPolicy(summarize_at_tokens=1000, max_input_tokens=2000)
    assert (policy.keep_tokens, policy.min_summary_gain_tokens) == (500, 250)
    with pytest.raises(ValueError, match="keep_tokens"):
        ContextPolicy(summarize_at_tokens=1000, keep_tokens=1000)
    with pytest.raises(ValueError, match="at least twice"):
        ContextPolicy(summarize_at_tokens=1000, max_input_tokens=1999)
    with pytest.raises(ValueError, match="positive"):
        ContextPolicy(summarize_at_tokens=0)


def test_cli_rejects_inconsistent_limits(folder: Path, capsys):
    from file_organizer.cli import main

    with pytest.raises(SystemExit) as exit_info:
        main([str(folder), "--summarize-at", "1000", "--max-context-tokens", "1500"],
             model_factory=lambda name: ScriptedChatModel(responses=[]))
    assert exit_info.value.code == 2
    assert "at least twice" in capsys.readouterr().err


def test_summary_gain_ignores_a_batch_that_fills_the_kept_window():
    msgs = _conversation(1)[1:]  # task + one small turn
    batch = [call("read_file", path=f"f{i}.txt") for i in range(5)]
    msgs += [AIMessage("", tool_calls=batch), *(ToolMessage("y " * 400, tool_call_id=c["id"]) for c in batch)]
    # The kept window (300 tokens) ends inside the batch, so the cut moves back to the
    # batch's AI message: only the task and the first turn would be summarized.
    assert summary_gain(msgs, 300, count_tokens_approximately) == count_tokens_approximately(msgs[:3])


class EchoSummarizer(BaseChatModel):
    """Fake cheap model whose 'summary' keeps a marker of every file content it was shown."""

    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "echo"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):  # noqa: ANN001
        self.calls += 1
        markers = sorted(set(re.findall(r"body-of-f\d+", str(messages[-1].content))))
        return ChatResult(generations=[ChatGeneration(message=AIMessage("NOTES read: " + ", ".join(markers)))])


def test_small_limits_never_lose_what_the_agent_has_read(tmp_path: Path):
    """Regression: summarize-at 1000 / trim at 2000 made the agent forget a batch of
    file reads (summarized away nothing, trimmed away everything) and re-read it."""
    for i in range(14):
        (tmp_path / f"f{i}.txt").write_text(f"body-of-f{i}. " * 170)  # ~550 tokens each
    responses = [
        ai(call("list_files")),
        ai(*(call("read_file", path=f"f{i}.txt") for i in range(12))),  # one ~7k-token batch
        ai(call("list_files", directory=".")),
        ai(call("read_file", path="f12.txt")),
        ai(call("read_file", path="f13.txt")),
        ai(content="done"),
    ]
    main, echo = ScriptedChatModel(responses=list(responses)), EchoSummarizer()
    graph = build_graph(main, Workspace(tmp_path), checkpointer=memory_sqlite_saver(),
                        summary_model=echo, policy=SMALL)
    result = graph.invoke({"messages": [HumanMessage("go")]}, CONFIG)
    assert result["messages"][-1].content == "done"

    # Every request shows something of every file read so far: its full content, a
    # preview, or the summary's notes (file names alone don't count: list_files has them).
    read_so_far: set[str] = set()
    for response, request in zip(responses, main.received):
        text = "\n".join(str(m.content) for m in request)
        shown = set(re.findall(r"body-of-(f\d+)", text))
        missing = {name for name in read_so_far if name.removesuffix(".txt") not in shown}
        assert not missing, f"the model was shown nothing of {sorted(missing)}"
        read_so_far |= {tc["args"]["path"] for tc in response.tool_calls if tc["name"] == "read_file"}
    # One summary, once newer reads pushed the batch out of the kept window; none while
    # the batch alone filled it (that would have condensed almost nothing, repeatedly).
    assert echo.calls == 1
    assert is_summary(result["messages"][0]) and "body-of-f0" in result["messages"][0].content


# ---- finalized decisions in the store --------------------------------------------------


def test_finalized_decisions_move_from_state_to_store(workspace: Workspace, folder: Path):
    store = InMemoryStore()
    model = ScriptedChatModel(responses=[
        ai(call("create_directory", path="finance"), call("move_file", source="invoice.txt", destination="nope")),
        ai(call("create_directory", path="junk")),
        ai(content="ok"),
    ])
    graph = build_graph(model, workspace, checkpointer=memory_sqlite_saver(), store=store)
    graph.invoke({"messages": [HumanMessage("go")]}, CONFIG)
    graph.invoke(Command(resume=True), CONFIG)
    result = graph.invoke(Command(resume=True), CONFIG)  # approved, but "nope" doesn't exist
    assert result["decisions"] == {}  # cleared once final

    records = DecisionLog(store, "t").records()
    assert [(r["description"], r["outcome"]) for r in records] == [
        ("Create directory 'finance'", "applied"),
        ("Move 'invoice.txt' -> 'nope'", "failed"),
    ]
    result = graph.invoke(Command(resume={"approved": False, "feedback": "no junk"}), CONFIG)
    assert DecisionLog(store, "t").records()[-1]["outcome"] == "rejected"

    # every later LLM call sees the decision log in its system prompt
    system = model.received[-1][0].content
    assert "- APPLIED: Create directory 'finance'" in system
    assert "- APPROVED BUT FAILED: Move 'invoice.txt' -> 'nope'" in system
    assert "- REJECTED: Create directory 'junk' (feedback: no junk)" in system


def test_decisions_survive_summarization(workspace: Workspace):
    store = InMemoryStore()
    model = ScriptedChatModel(responses=[
        ai(call("create_directory", path="junk")),
        *[ai(call("read_file", path=name)) for name in ("invoice.txt", "flight.txt", "bloodwork.txt")],
        ai(content="ok"),
    ])
    graph = build_graph(model, workspace, checkpointer=memory_sqlite_saver(), store=store,
                        summary_model=cheap_model(), policy=ContextPolicy(summarize_at_tokens=50))
    graph.invoke({"messages": [HumanMessage("go")]}, CONFIG)
    result = graph.invoke(Command(resume={"approved": False, "feedback": "never"}), CONFIG)

    assert not any("junk" in str(m.content) for m in result["messages"][1:])  # summarized away
    assert "- REJECTED: Create directory 'junk' (feedback: never)" in model.received[-1][0].content


def test_decision_log_is_per_thread_and_clearable():
    store = InMemoryStore()
    a, b = DecisionLog(store, "organize:/tmp/a.b"), DecisionLog(store, "organize:/tmp/c")
    record = {"tool": "create_directory", "args": {}, "description": "d", "approved": True,
              "feedback": None, "outcome": "applied", "result": "", "decided_at": "2026"}
    a.record("1", record)
    a.record("1", record)  # upsert, not a duplicate
    b.record("2", record)
    assert len(a.records()) == 1 and len(b.records()) == 1
    assert "." not in "".join(decisions_namespace("organize:/tmp/a.b"))  # valid store labels
    assert a.clear() == 1 and a.records() == [] and len(b.records()) == 1


# ---- retention: keep_latest checkpoints, store TTL -------------------------------------


def test_keep_latest_prune_preserves_a_half_answered_batch(workspace: Workspace, folder: Path):
    saver = memory_sqlite_saver()
    batch = ai(*(call("create_directory", path=p) for p in ("x", "y", "z")))
    graph = build_graph(ScriptedChatModel(responses=[batch, ai(content="ok")]), workspace, checkpointer=saver)
    graph.invoke({"messages": [HumanMessage("go")]}, CONFIG)
    graph.invoke(Command(resume=True), CONFIG)  # x approved, now asking about y
    assert len(list(saver.list(CONFIG))) > 1

    saver.prune(["t"], strategy="keep_latest")
    assert len(list(saver.list(CONFIG))) == 1
    [pending] = graph.get_state(CONFIG).interrupts
    assert pending.value["args"] == {"path": "y"}  # x's answer survived the pruning

    graph.invoke(Command(resume=False), CONFIG)
    result = graph.invoke(Command(resume=True), CONFIG)
    assert (folder / "x").is_dir() and not (folder / "y").exists() and (folder / "z").is_dir()
    assert result["messages"][-1].content == "ok"


def test_prune_delete_removes_the_thread_and_rejects_unknown_strategies(workspace: Workspace):
    saver = memory_sqlite_saver()
    graph = build_graph(ScriptedChatModel(responses=[ai(call("create_directory", path="x"))]),
                        workspace, checkpointer=saver)
    graph.invoke({"messages": [HumanMessage("go")]}, CONFIG)
    saver.prune(["t"], strategy="delete")
    assert list(saver.list(CONFIG)) == []
    with pytest.raises(ValueError, match="strategy"):
        saver.prune(["t"], strategy="oldest")


def test_store_items_get_a_ttl_and_expired_ones_are_swept(tmp_path: Path):
    db = tmp_path / "state.sqlite"
    with sqlite_store(db) as store:
        assert store.supports_ttl
        store.put(("ns",), "old", {"v": 1})
        store.put(("ns",), "new", {"v": 2})
        row = store.conn.execute("SELECT ttl_minutes FROM store WHERE key = 'new'").fetchone()
        assert row[0] == 7 * 24 * 60
        store.conn.execute("UPDATE store SET expires_at = '2000-01-01 00:00:00' WHERE key = 'old'")
    with sqlite_store(db) as store:  # next run: expired items are swept on open
        assert store.get(("ns",), "old") is None
        assert store.get(("ns",), "new").value == {"v": 2}


def test_checkpointer_and_store_share_one_file(tmp_path: Path):
    db = tmp_path / "state.sqlite"
    with sqlite_checkpointer(db) as saver, sqlite_store(db) as store:
        store.put(("ns",), "k", {"v": 1})
        assert list(saver.list(CONFIG)) == []
    tables = {r[0] for r in __import__("sqlite3").connect(db).execute("SELECT name FROM sqlite_master")}
    assert {"checkpoints", "writes", "store"} <= tables


# ---- the CLI wiring --------------------------------------------------------------------


def _answers(*replies: str):
    queue = list(replies)

    def fake_input(prompt: str = "") -> str:
        if not queue:
            raise KeyboardInterrupt
        return queue.pop(0)

    return fake_input


def test_cli_prunes_while_paused_and_clears_decisions_on_a_fresh_start(folder: Path, tmp_path: Path):
    db = tmp_path / "state.sqlite"
    thread_id = f"organize:{Workspace(folder).root}"
    model = ScriptedChatModel(responses=[
        ai(call("list_files")), ai(call("create_directory", path="a"), call("create_directory", path="b")),
    ])
    assert run(folder, model, input_fn=_answers("n"), output_fn=lambda *_: None, state_db=db) == 130
    with sqlite_checkpointer(db) as saver, sqlite_store(db) as store:
        assert len(list(saver.list({"configurable": {"thread_id": thread_id}}))) == 1  # keep_latest
        # Ctrl-C came before the batch was applied: nothing is final yet
        assert DecisionLog(store, thread_id).records() == []

    # resume, finish the batch, finish the session: the decisions stay in the store
    model = ScriptedChatModel(responses=[ai(content="done")])
    assert run(folder, model, input_fn=_answers("y", "y"), output_fn=lambda *_: None, state_db=db) == 0
    with sqlite_checkpointer(db) as saver, sqlite_store(db) as store:
        assert list(saver.list({"configurable": {"thread_id": thread_id}})) == []
        assert [r["outcome"] for r in DecisionLog(store, thread_id).records()] == ["rejected", "applied"]

    # the next fresh session starts without the old decisions
    model = ScriptedChatModel(responses=[ai(content="again")])
    assert run(folder, model, input_fn=_answers(), output_fn=lambda *_: None, state_db=db) == 0
    assert "most recent decisions" not in model.received[0][0].content


def test_cli_reports_summaries_and_summarizer_tokens(big_folder: Path, tmp_path: Path):
    output: list[str] = []
    code = run(big_folder, ScriptedChatModel(responses=reading_session()), summary_model=cheap_model(tokens=100),
               policy=SMALL, input_fn=_answers(), output_fn=output.append, state_db=tmp_path / "s.sqlite")
    assert code == 0
    calls = [line for line in output if line.startswith("  [summarizer call #")]
    assert calls and calls[0] == "  [summarizer call #1: 100 in + 10 out tokens; summarizer total 110]"
    effects = [line for line in output if line.startswith("  [history ~")]
    assert len(effects) == len(calls)
    match = re.fullmatch(r"  \[history ~([\d,]+) -> ~([\d,]+) tokens: (\d+) older messages summarized, "
                         r"(\d+) kept\]", effects[0])
    assert match, effects[0]
    before, after = (int(g.replace(",", "")) for g in match.groups()[:2])
    assert before >= SMALL.summarize_at_tokens > after
    n = len(calls)
    assert f"  plus summarizer: {100 * n:,} in + {10 * n:,} out = {110 * n:,} tokens in {n} LLM call" \
        f"{'' if n == 1 else 's'}" in output
    assert any(line.startswith("  all models: ") for line in output)


def test_each_summarizer_call_is_logged_with_its_tokens(big_folder: Path, caplog):
    caplog.set_level("INFO", logger="file_organizer")
    graph = build_graph(ScriptedChatModel(responses=reading_session()), Workspace(big_folder),
                        checkpointer=memory_sqlite_saver(), summary_model=cheap_model(tokens=100), policy=SMALL)
    graph.invoke({"messages": [HumanMessage("go")]}, CONFIG)
    records = [r for r in caplog.records if r.getMessage().startswith("Summarizer call #")]
    assert [r.summary_call for r in records] == list(range(1, len(records) + 1))
    assert all(r.summary_input_tokens == 100 and r.summary_total_tokens == 110 for r in records)
    assert records[-1].session_summary_total_tokens == 110 * len(records)
    assert "tokens input=100 output=10 total=110" in records[0].getMessage()
