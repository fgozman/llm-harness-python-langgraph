"""LangGraph agent: model <-> tools loop with a human approval gate and managed memory.

    START -> summarize -> agent --(no tool calls)--> END
                ^           |
                |           +--(tool calls)--> review --> tools --+
                +-------------------------------------------------+

* summarize: once the history is too long, a cheap model condenses older messages into
          a summary and they are deleted from the state (see memory.py). Usually a no-op.
* agent:  asks the LLM what to do next (trimmed to a token budget as a safety net, with
          the human's finalized decisions from the store added to the system prompt).
* review: pauses with interrupt() once for every filesystem-changing tool call and stores
          the human's answers. It changes nothing on disk, because LangGraph re-runs the
          whole node on every resume (already-answered interrupts replay their answers).
* tools:  runs read-only calls and approved changes. Rejected changes get a
          "Rejected ..." result instead, including the human's feedback. Each decision
          is then final and moves from the state into the store.

LangGraph in a nutshell (for readers new to the framework):

* State:  a typed dict shared by all nodes (``AgentState`` below). Each key is a
          "channel"; LangGraph owns the current value and hands it to every node.
* Node:   a plain function ``state -> partial update``. It returns only the keys it
          wants to change; LangGraph merges them into the state (see "reducers").
          A node may also take a ``runtime`` argument (``langgraph.runtime.Runtime``)
          to reach the store and run info such as the thread id.
* Edge:   says which node runs next. Fixed edges always go to the same node;
          conditional edges call a routing function that picks the next node.
* Graph:  ``StateGraph`` is the builder; ``compile()`` turns it into a runnable
          object with ``invoke()`` / ``stream()`` / ``get_state()``.
* Steps:  execution happens in "super-steps": run the scheduled node(s), merge their
          updates into the state, save a checkpoint, follow the edges, repeat until END.
* Checkpointer + thread_id: persist the state after each step, per conversation
          ("thread"). This is short-term memory, and what makes interrupt/resume work.
* Store:  a key-value store shared by all threads (long-term memory). Items live
          under a namespace tuple and are independent of checkpoints.

LangGraph also ships ready-made pieces for this pattern (``langgraph.prebuilt.ToolNode``,
``tools_condition``, and ``langchain.agents.create_agent`` with ``HumanInTheLoopMiddleware``
and ``SummarizationMiddleware``). They are written by hand here so every mechanism is
visible, and so the human review asks about one change at a time.
"""

from __future__ import annotations

import logging
import time
from typing import Annotated, Any, TypedDict

# LangChain (langchain_core): the building blocks: chat models, messages, tools.
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage

# LangGraph: the orchestration layer that wires those blocks into a stateful graph.
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,  # interface of all checkpointers
)
from langgraph.graph import (  # START/END are virtual entry/exit nodes
    END,
    START,
    StateGraph,
)
from langgraph.graph.message import add_messages  # reducer that appends chat messages
from langgraph.runtime import Runtime  # injected into nodes that ask for it
from langgraph.store.base import BaseStore  # interface of all stores
from langgraph.store.memory import InMemoryStore
from langgraph.types import interrupt  # pause the graph and wait for human input

from .approval import Decision, _safe, describe, parse_decision
from .memory import (
    ContextPolicy,
    DecisionLog,
    SummaryStats,
    decision_line,
    fit_context,
    make_summarize_node,
    now_iso,
    token_counter_for,
)
from .prompts import system_prompt
from .tools import MUTATING_TOOLS, make_tools
from .usage import TokenUsage, add_usage
from .workspace import Workspace

log = logging.getLogger(__name__)


class AgentState(TypedDict):
    """The graph's state schema: every node receives it and returns a partial update.

    LangGraph reads these type hints when building the graph. ``Annotated[type, fn]``
    attaches a *reducer* ``fn(old_value, update) -> new_value`` that decides how a
    node's update is merged; a key without a reducer is simply overwritten.
    """

    # add_messages is a reducer: nodes return only NEW messages and they get appended
    # (a message whose id already exists replaces the old one, and RemoveMessage
    # deletes, which is how summarization shrinks the history).
    messages: Annotated[list[AnyMessage], add_messages]
    # No reducer: the review node overwrites this with tool_call_id -> decision for the
    # current batch only. It must live in the state (saved in the checkpoint) because
    # the tools node runs in a later step, maybe after a restart. Once applied, the
    # decisions are final: the tools node writes them to the store and clears this.
    decisions: dict[str, Decision]
    # Running token totals of the main model and of the summarizer. Kept here rather
    # than recomputed from the messages, because summarization deletes messages.
    usage: Annotated[dict[str, int], add_usage]
    summary_usage: Annotated[dict[str, int], add_usage]
    # No reducer: overwritten by each summary, so progress output can show its effect.
    last_summary: SummaryStats


def _decision_log(runtime: Runtime) -> DecisionLog:
    """The decision log of the thread this node is running in."""
    return DecisionLog(runtime.store, runtime.execution_info.thread_id)


def build_graph(
    model: BaseChatModel,
    workspace: Workspace,
    *,
    checkpointer: BaseCheckpointSaver,
    store: BaseStore | None = None,
    summary_model: BaseChatModel | None = None,
    policy: ContextPolicy | None = None,
):
    """Build the agent graph.

    ``checkpointer`` stores its state, normally from ``sqlite_checkpointer()``. It's
    required: interrupt() can only pause a run whose state is saved somewhere.
    ``store`` keeps the finalized decisions (normally ``sqlite_store()``; defaults to an
    ``InMemoryStore``, lost on exit). ``summary_model`` writes summaries of old messages;
    pass a cheap one (defaults to ``model``). ``policy`` sets the context-size limits.
    """
    policy = policy or ContextPolicy()
    tools = {t.name: t for t in make_tools(workspace)}
    # LangChain, not LangGraph: bind_tools sends the tools' JSON schemas with every
    # request, so the model can answer with structured "tool calls" instead of text.
    llm = model.bind_tools(list(tools.values()))  # tell the model which tools exist

    # ---- Nodes ---------------------------------------------------------------------
    # A node is any function (or Runnable) taking the current state and returning a
    # dict with the keys to update. LangGraph calls it; you never call it yourself.
    # Defining them inside build_graph lets them close over `llm` and `tools`.

    def tool_calls(state: AgentState) -> list[dict[str, Any]]:
        """Tool calls on the model's latest message: [{"name", "args", "id"}, ...]."""
        return state["messages"][-1].tool_calls

    # Summarization and trimming must agree on what "too long" means: one counter.
    token_counter = token_counter_for(model)
    # Node: SummarizationMiddleware's before_model hook, wrapped as a node (memory.py).
    summarize = make_summarize_node(model, summary_model, policy, token_counter)

    def agent(state: AgentState, runtime: Runtime) -> dict[str, Any]:
        """Node: ask the LLM for its next move given the conversation so far."""
        # Totals so far come from the state, so they survive a pause/restart.
        session = TokenUsage.from_dict(state.get("usage"))
        call_no = session.calls + 1
        # The system prompt is built on every call rather than stored in the state, so
        # it is never duplicated in the history (or summarized away), and it carries
        # the up-to-date decision log from the store.
        # (Reading doesn't refresh the decisions' TTL: that would be a write per call.)
        decisions = _decision_log(runtime).records(policy.max_decisions_in_prompt, refresh_ttl=False)
        system = SystemMessage(system_prompt([decision_line(d) for d in decisions]))
        # Safety net: the state keeps everything, only this request is trimmed.
        request = fit_context([system, *state["messages"]], policy.max_input_tokens, token_counter)
        log.info("LLM call #%d: sending %d messages", call_no, len(request),
                 extra={"llm_call": call_no})
        started = time.perf_counter()
        try:
            response = llm.invoke(request)
        except Exception:
            log.warning("LLM call #%d failed after %.2fs", call_no, time.perf_counter() - started,
                        extra={"llm_call": call_no})
            raise
        elapsed = time.perf_counter() - started
        this_call = TokenUsage.of([response])
        session.merge(this_call)
        if this_call.reported:
            log.info(
                "LLM call #%d done in %.2fs: tokens input=%d output=%d total=%d "
                "(session: input=%d output=%d total=%d)",
                call_no, elapsed, this_call.input_tokens, this_call.output_tokens,
                this_call.total_tokens, session.input_tokens, session.output_tokens,
                session.total_tokens,
                extra={"llm_call": call_no, "duration_s": round(elapsed, 3),
                       **this_call.as_extra(), **session.as_extra("session_")},
            )
        else:
            log.info("LLM call #%d done in %.2fs: token usage not reported", call_no, elapsed,
                     extra={"llm_call": call_no, "duration_s": round(elapsed, 3)})
        # INFO keeps only names and sizes; arguments and text (which can quote what the
        # files say) are DEBUG. Approved changes are still logged with their paths by the
        # workspace (the audit trail).
        for tc in response.tool_calls:
            log.info("Model requested %s", tc["name"], extra={"tool": tc["name"]})
            log.debug("Model requested %s(%s)", tc["name"], tc["args"])
        if not response.tool_calls:
            log.info("Model finished (%d chars)", len(response.text))
            log.debug("Model finished: %s", response.text)
        # Partial update: only the NEW message (appended by add_messages) and this
        # call's tokens (added to the totals by add_usage); "decisions" is untouched.
        return {"messages": [response], "usage": this_call.as_dict()}

    def route_after_agent(state: AgentState) -> str:
        """Routing function for a conditional edge: returns the NAME of the next node.

        It runs after the agent node's update has been merged, so the last message is
        the model's reply. Returning END stops the graph.
        """
        return "review" if tool_calls(state) else END

    def review(state: AgentState) -> dict[str, Any]:
        """Node: human-in-the-loop gate. Asks about each change, applies nothing.

        How interrupt() works, step by step:
        1. The first interrupt() call raises a special exception. LangGraph catches it,
           saves a checkpoint, and returns control to the caller with the payload.
        2. The caller resumes with ``Command(resume=answer)`` and the same thread_id.
        3. LangGraph re-runs this node FROM THE TOP. Interrupts already answered now
           return their saved answers (matched by order), and the next unanswered
           interrupt() pauses again. Hence: no side effects in this node.
        4. Once every interrupt() has an answer, the node returns normally.
        """
        decisions = {}
        changes = [tc for tc in tool_calls(state) if tc["name"] in MUTATING_TOOLS]
        # The whole batch goes with every question, so the caller can show the full
        # plan before the first answer (and offer "approve all remaining"). The plan
        # assumes every change is approved, which is what "approve all" does.
        batch = []
        claimed: set[str] = set()  # destinations of earlier approved moves of this batch
        gone: set[str] = set()  # their sources, and the directories they leave empty
        for tc in changes:
            planned, emptied = _move_effects(tc, claimed, gone)
            batch.append({"tool_call_id": tc["id"], "description": _describe(tc, planned, emptied)})
            _claim(tc, planned, emptied, claimed, gone)
        claimed, gone = set(), set()
        for index, tc in enumerate(changes):
            planned, emptied = _move_effects(tc, claimed, gone)
            # Pauses the run and shows this dict to the caller, who later resumes
            # with Command(resume=answer); interrupt() then returns that answer.
            # The payload must be serializable: it is stored in the checkpoint.
            answer = interrupt(
                {
                    "tool_call_id": tc["id"],
                    "tool": tc["name"],
                    "args": tc["args"],
                    "description": _describe(tc, planned, emptied),
                    "index": index,
                    "batch": batch,
                }
            )
            decision = parse_decision(answer)
            if decision["approved"]:
                _claim(tc, planned, emptied, claimed, gone)
                if emptied:
                    decision["remove_dirs"] = emptied
            decisions[tc["id"]] = decision
        return {"decisions": decisions}

    def _move_effects(tc: dict[str, Any], claimed: set[str],
                      gone: set[str]) -> tuple[str | None, list[str]]:
        """For a move_file call: where the file will land and which directories it will
        leave empty, after the earlier approved moves of the batch (``claimed``, ``gone``).
        (None, []) if it is not a move or would fail. Read-only, so safe in review."""
        planned = _planned_move(tc, claimed)
        if planned is None:
            return None, []
        # The move's own destination counts too: moving into an ancestor keeps it non-empty.
        return planned, workspace.emptied_directories(tc["args"]["source"], gone, [*claimed, planned])

    def _claim(tc: dict[str, Any], planned: str | None, emptied: list[str],
               claimed: set[str], gone: set[str]) -> None:
        """Count an approved move's effects for the batch's later moves."""
        if planned is not None:
            claimed.add(planned)
            gone.update([tc["args"]["source"], *emptied])

    def _describe(tc: dict[str, Any], planned: str | None, emptied: list[str]) -> str:
        return describe(tc["name"], tc["args"]) + _clash_note(tc, planned) + _removal_note(emptied)

    def _removal_note(emptied: list[str]) -> str:
        """Tell the human which directories the move will remove because it empties them."""
        if not emptied:
            return ""
        return " (then removes the emptied director{} {})".format(
            "y" if len(emptied) == 1 else "ies", ", ".join(f"'{_safe(d)}'" for d in emptied))

    def _planned_move(tc: dict[str, Any], claimed: set[str]) -> str | None:
        """Where an approved move_file call will put the file (relative path), counting
        the destinations claimed by earlier moves of the batch; None if it is not a move
        or would fail. (A read-only check, so it is safe in the review node.)"""
        args = tc["args"] if isinstance(tc["args"], dict) else {}
        if tc["name"] != "move_file":
            return None
        source, destination = args.get("source"), args.get("destination")
        if not isinstance(source, str) or not isinstance(destination, str):
            return None
        return workspace.planned_destination(source, destination, claimed)

    def _clash_note(tc: dict[str, Any], planned: str | None) -> str:
        """With rename-on-clash, tell the human up front if the file will get a new name."""
        if planned is None or not workspace.rename_on_clash:
            return ""
        if planned.rsplit("/", 1)[-1] == tc["args"]["source"].rsplit("/", 1)[-1]:
            return ""
        return f" (name taken there; will be saved as '{_safe(planned)}')"

    def _remove_emptied(result: ToolMessage, paths: list[str]) -> None:
        """After an applied move, remove the emptied directories the human approved with
        it; report the outcome in the result and record the removals for undo."""
        removed, problem = workspace.remove_emptied_directories(paths)
        if removed:
            result.content += " Removed the now-empty director{} {}.".format(
                "y" if len(removed) == 1 else "ies", ", ".join(f"'{d}'" for d in removed))
            result.artifact = {**result.artifact, "removed": removed}
        if problem:
            log.warning("Directory cleanup after a move stopped: %s", problem)
            result.content += f" Directory cleanup stopped: {problem}."

    def run_tools(state: AgentState, runtime: Runtime) -> dict[str, Any]:
        """Node: execute the tool calls (what langgraph.prebuilt.ToolNode does, plus the
        approval check), return the results as messages for the model to read, and move
        the now-final decisions to the store."""
        decision_log = _decision_log(runtime)
        # Every tool call must be answered by a ToolMessage with the same tool_call_id;
        # model providers reject a conversation where a tool call has no answer.
        results = []
        for tc in tool_calls(state):
            name, call_id = tc["name"], tc["id"]
            decision = state["decisions"].get(call_id, parse_decision(None))
            if name in MUTATING_TOOLS:  # logged here, not in review, which re-runs on resume
                verdict = "approved" if decision["approved"] else "rejected"
                log.info("User %s: %s (feedback: %s)", verdict, describe(name, tc["args"]), decision["feedback"])

            if name not in tools:
                text = f"Error: unknown tool '{name}'. Available: {', '.join(tools)}."
                results.append(ToolMessage(text, tool_call_id=call_id, name=name, status="error"))
                continue
            if name in MUTATING_TOOLS and not decision["approved"]:
                text = "Rejected by the user; the change was NOT applied."
                if decision["feedback"]:
                    text += f" User feedback: {decision['feedback']}"
                result = ToolMessage(text, tool_call_id=call_id, name=name)
            else:
                # Invoking with the whole tool call returns a ready-made ToolMessage.
                result = tools[name].invoke(tc)
                if result.status != "error" and decision.get("remove_dirs"):
                    _remove_emptied(result, decision["remove_dirs"])
            results.append(result)

            if name in MUTATING_TOOLS:
                outcome = ("rejected" if not decision["approved"]
                           else "failed" if result.status == "error" else "applied")
                decision_log.record(call_id, {
                    "tool": name,
                    "args": tc["args"],
                    "description": describe(name, tc["args"]),
                    "approved": decision["approved"],
                    "feedback": decision["feedback"],
                    "outcome": outcome,
                    "result": str(result.content)[:500],
                    "decided_at": now_iso(),
                    # How to reverse it (for --undo); None if nothing changed on disk.
                    "undo": result.artifact if outcome == "applied" else None,
                })

        for msg in results:
            if msg.status == "error":
                log.warning("Tool %s failed: %s", msg.name, msg.content)
            else:
                log.debug("Tool %s returned %d chars", msg.name, len(msg.content))
        # Messages are appended by add_messages; the next edge goes back (through
        # summarize) to the agent, which reads these results. The pending decisions are
        # cleared: from now on the store is their single source of truth.
        return {"messages": results, "decisions": {}}

    # ---- Wiring --------------------------------------------------------------------
    # StateGraph is a builder: it only records nodes and edges. The schema passed here
    # defines the state keys and their reducers.
    builder = StateGraph(AgentState)
    # add_node(name, function): the name is what edges, routers and stream output use.
    builder.add_node("summarize", summarize)
    builder.add_node("agent", agent)
    builder.add_node("review", review)
    builder.add_node("tools", run_tools)
    # START is the virtual entry point: the graph input is merged into the state first,
    # then "summarize" runs (a no-op until the history is long).
    builder.add_edge(START, "summarize")
    builder.add_edge("summarize", "agent")
    # Conditional edge: after "agent", call route_after_agent to choose the next node.
    # The list declares every possible destination (used for validation and for
    # drawing the graph, e.g. graph.get_graph().draw_mermaid()).
    builder.add_conditional_edges("agent", route_after_agent, ["review", END])
    # Plain edges: always go from one node to the next.
    builder.add_edge("review", "tools")
    # This edge closes the loop. Going through "summarize" means the history is
    # checked right after tool results (the usual cause of growth) are added.
    builder.add_edge("tools", "summarize")
    # compile() validates the graph (e.g. every edge targets a known node) and returns a
    # CompiledStateGraph, a LangChain Runnable with invoke()/stream()/get_state().
    # The checkpointer saves state after each step; interrupt() needs it to pause/resume.
    # The store is handed to nodes as runtime.store.
    return builder.compile(checkpointer=checkpointer, store=store if store is not None else InMemoryStore())
