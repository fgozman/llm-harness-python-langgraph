"""Interactive command line: run the agent and ask the human to approve each change."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

# Command is the input you send to a graph to control it rather than to add data,
# here: Command(resume=...) answers a pending interrupt() and continues the run.
from langgraph.types import Command

from .approval import Decision
from .graph import build_graph
from .log import (
    DEFAULT_LOG_FILE,
    DEFAULT_LOG_FORMAT,
    DEFAULT_LOG_LEVEL,
    LOG_FORMATS,
    LOG_LEVELS,
    configure_logging,
    session_id,
)
from .memory import ContextPolicy, DecisionLog
from .persistence import DEFAULT_STATE_DB, sqlite_checkpointer, sqlite_store
from .prompts import START_MESSAGE
from .tools import MUTATING_TOOLS
from .tracing import (
    DEFAULT_PHOENIX_ENDPOINT,
    DEFAULT_PHOENIX_PROJECT,
    TracingUnavailable,
    configure_tracing,
    trace_session,
)
from .usage import TokenUsage
from .workspace import Workspace, WorkspaceError

log = logging.getLogger(__name__)

DEFAULT_MODEL = "anthropic:claude-sonnet-4-5"
# Cheap models for summarizing old messages, by provider of the main model. Other
# providers summarize with the main model unless --summary-model is given.
CHEAP_MODELS = {"anthropic": "anthropic:claude-haiku-5-5", "openai": "openai:gpt-4.1-mini"}
RESUME_HINT = "Progress is saved; run the same command again to resume."
# Max graph steps (super-steps, i.e. node executions here) per run. LangGraph raises
# GraphRecursionError when it is exceeded. Its default is ~10,000, so lower it to fail
# fast if the model gets stuck in a loop. Each tool round trip takes 4 steps
# (summarize, agent, review, tools).
RECURSION_LIMIT = 300

InputFn = Callable[[str], str]
OutputFn = Callable[[str], Any]


def ask_approval(request: dict[str, Any], input_fn: InputFn, output_fn: OutputFn) -> Decision:
    """Ask the human about one proposed change.

    "y"/"yes" approves, "n"/"no" rejects, and any other text rejects and is passed to
    the model as feedback. An empty answer asks again.
    """
    output_fn(f"\n? {request['description']}")
    answer = ""
    while not answer:
        answer = input_fn("  approve? [y]es / [n]o / or type feedback to reject: ").strip()
    if answer.lower() in {"y", "yes"}:
        return {"approved": True, "feedback": None}
    if answer.lower() in {"n", "no"}:
        return {"approved": False, "feedback": None}
    return {"approved": False, "feedback": answer}


def default_summary_model(model: str) -> str:
    """The cheap summarizer for a main model id: 'anthropic:...' -> Claude Haiku, etc."""
    provider = model.split(":", 1)[0] if ":" in model else ""
    return CHEAP_MODELS.get(provider, model)


@dataclass
class SessionUsage:
    """Token totals of a session: the main model and the (cheap) summarizer."""

    agent: TokenUsage = field(default_factory=TokenUsage)
    summarizer: TokenUsage = field(default_factory=TokenUsage)

    @classmethod
    def from_state(cls, values: dict[str, Any]) -> SessionUsage:
        return cls(TokenUsage.from_dict(values.get("usage")),
                   TokenUsage.from_dict(values.get("summary_usage")))


def _report(node: str, update: dict[str, Any] | None, output_fn: OutputFn, usage: SessionUsage) -> None:
    """Print progress for what one graph node just produced.

    ``update`` is exactly the dict the node returned (with stream_mode="updates"),
    e.g. {"messages": [AIMessage(...)]} from the agent node, not the full state.
    ``usage`` holds the session's token totals; this step's usage is added to it here.
    """
    if not update:  # e.g. the summarize node when there was nothing to do
        return
    if node == "summarize":
        this_call = TokenUsage.from_dict(update.get("summary_usage"))
        usage.summarizer.merge(this_call)
        tokens = (f"{this_call.input_tokens:,} in + {this_call.output_tokens:,} out tokens; "
                  f"summarizer total {usage.summarizer.total_tokens:,}"
                  if this_call.reported else "token usage not reported")
        output_fn(f"  [summarizer call #{usage.summarizer.calls}: {tokens}]")
        if stats := update.get("last_summary"):
            output_fn(f"  [history ~{stats['tokens_before']:,} -> ~{stats['tokens_after']:,} tokens: "
                      f"{stats['summarized_messages']} older messages summarized, "
                      f"{stats['kept_messages']} kept]")
        return
    for msg in update.get("messages", []):
        if node == "agent":
            usage.agent.add(msg)
            if meta := msg.usage_metadata:
                output_fn(
                    f"  [llm call #{usage.agent.calls}: {meta.get('input_tokens', 0):,} in + "
                    f"{meta.get('output_tokens', 0):,} out tokens; "
                    f"session total {usage.agent.total_tokens:,}]"
                )
            if not msg.tool_calls and msg.text:  # no tool calls: the model's final summary
                output_fn(f"\n{msg.text}")
            for tc in msg.tool_calls:  # changes are shown later, in the approval prompt
                if tc["name"] not in MUTATING_TOOLS:
                    args = ", ".join(f"{k}={v!r}" for k, v in tc["args"].items())
                    output_fn(f"  · {tc['name']}({args})")
        elif node == "tools" and msg.status == "error":
            output_fn(f"  ! {msg.content}")


def _report_usage(usage: SessionUsage, output_fn: OutputFn) -> None:
    """Print and log the session's token totals."""
    output_fn(f"\nTokens used: {usage.agent.summary()}")
    extra = usage.agent.as_extra("session_")
    message = usage.agent.summary()
    if usage.summarizer.calls:
        total = TokenUsage()
        total.merge(usage.agent)
        total.merge(usage.summarizer)
        output_fn(f"  plus summarizer: {usage.summarizer.summary()}")
        output_fn(f"  all models: {total.summary()}")
        message = f"main model: {message}; summarizer: {usage.summarizer.summary()}; all: {total.summary()}"
        extra |= usage.summarizer.as_extra("summary_") | total.as_extra("all_")
    log.info("Session token usage: %s", message, extra=extra)


def confirm(question: str, input_fn: InputFn) -> bool:
    """Yes/no question where an empty answer means yes."""
    while True:
        answer = input_fn(f"{question} [Y/n] ").strip().lower()
        if answer in {"", "y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False


def run(
    folder: str | Path,
    model: BaseChatModel,
    *,
    summary_model: BaseChatModel | None = None,
    policy: ContextPolicy | None = None,
    input_fn: InputFn = input,
    output_fn: OutputFn = print,
    state_db: str | Path = DEFAULT_STATE_DB,
) -> int:
    """Organize ``folder`` interactively. Returns a process exit code.

    The graph state is saved in the SQLite file ``state_db`` after every step. If a run
    stops early (Ctrl-C, crash, model/API error), running it again on the same folder
    offers to resume it where it stopped. Finalized decisions are kept in the same
    file's store; ``summary_model`` (cheap) summarizes old messages per ``policy``.
    """
    workspace = Workspace(folder)
    # A "thread" is one persisted conversation/run. Its id is chosen by us and selects
    # which saved state the checkpointer loads and writes. One saved session per
    # folder: the same folder always maps to the same thread.
    thread_id = f"organize:{workspace.root}"
    # The RunnableConfig passed to every graph call. "configurable.thread_id" is
    # required as soon as the graph has a checkpointer.
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": RECURSION_LIMIT}

    output_fn(f"Organizing {workspace.root}")
    # Tag every log record of this run with a fresh id (a "session_id" field in JSON logs).
    session_token = session_id.set(str(uuid.uuid4()))
    usage = SessionUsage()  # token totals of this session (including resumed earlier runs)
    log.info("Run %s started on %s (state: %s)", session_id.get(), workspace.root, state_db)
    try:
        # LangGraph's SQLite checkpointer (graph state per thread_id) and store (finalized
        # decisions, with a TTL), both in the state_db file.
        # trace_session: if tracing to Phoenix is on, this whole session (the start and
        # every resume after an approval) is one trace, under one root span.
        with (
            trace_session(thread_id, input=f"Organize {workspace.root}",
                          metadata={"run_id": session_id.get(),
                                    "folder": str(workspace.root)}) as trace,
            sqlite_checkpointer(state_db) as checkpointer,
            sqlite_store(state_db) as store,
        ):
            graph = build_graph(model, workspace, checkpointer=checkpointer, store=store,
                                summary_model=summary_model, policy=policy)

            # Input for a fresh run: a partial state update, merged by the reducers
            # (add_messages appends this first user message to the empty history).
            graph_input: Any = {"messages": [HumanMessage(START_MESSAGE)]}
            pending = None  # an approval question (Interrupt) waiting for an answer
            # get_state() loads the latest checkpoint of this thread as a StateSnapshot:
            #   .values     the state dict (messages, decisions, usage, summary_usage)
            #   .next       nodes that would run next; empty if the run finished or never
            #               started, non-empty if it stopped midway (pause or crash)
            #   .interrupts pending interrupt() requests, each with .id and .value
            saved = graph.get_state(config)
            if saved.next and confirm("Resume the interrupted session for this folder?", input_fn):
                log.info("Resuming the interrupted session (stopped before: %s)", saved.next)
                graph_input = None  # None = continue from the last saved step
                run_name = "resume after restart"
                pending = saved.interrupts[0] if saved.interrupts else None
                usage = SessionUsage.from_state(saved.values)
            else:  # start over with a clean state
                checkpointer.prune([thread_id], strategy="delete")
                DecisionLog(store, thread_id).clear()
                run_name = "organize"

            while True:
                if pending is not None:
                    # Answer exactly this question (by interrupt id), so an old answer can
                    # never be applied to a different question after a restart.
                    decision = ask_approval(pending.value, input_fn, output_fn)
                    trace.decision(pending.value, decision)
                    answer = "approved" if decision["approved"] else "rejected"
                    run_name = f"resume: {answer} {pending.value['description']}"[:120]
                    # A dict keyed by interrupt ids is a "resume map"; a plain value
                    # (Command(resume=decision)) would answer whichever one is pending.
                    graph_input = Command(resume={pending.id: decision})
                    pending = None
                # Runs the graph until it finishes or pauses; yields {node_name: output}.
                # (invoke() would do the same but only return the final state; other
                # stream modes: "values" = full state after each step, "messages" =
                # LLM tokens as they are generated.)
                # durability="sync" writes each checkpoint before the next step starts,
                # so nothing is lost if the process dies (default "async" writes in the
                # background; "exit" only when the run ends).
                # run_name: the name of this graph run's span in tracing tools (Phoenix).
                run_config = {**config, "run_name": run_name}
                for step in graph.stream(graph_input, run_config, stream_mode="updates",
                                         durability="sync"):
                    for node, update in step.items():
                        if node == "__interrupt__":  # paused by interrupt() in the review node
                            # update is a tuple of Interrupt objects; the review node
                            # asks one question at a time, so there is only one.
                            # The stream then ends: the run is paused, not finished.
                            pending = update[0]
                        else:
                            _report(node, update, output_fn, usage)
                if pending is not None:
                    # Paused for a human answer. Resuming needs only the newest checkpoint
                    # (and its writes), so drop the older ones to keep the file small.
                    checkpointer.prune([thread_id], strategy="keep_latest")
                else:  # finished without pausing: the agent is done
                    messages = graph.get_state(config).values.get("messages", [])
                    trace.finish(messages[-1].text if messages else "")
                    checkpointer.prune([thread_id], strategy="delete")  # nothing left to resume
                    _report_usage(usage, output_fn)
                    log.info("Session finished")
                    return 0
    except (KeyboardInterrupt, EOFError):
        output_fn("\nAborted. No further changes were made.")
        _report_usage(usage, output_fn)
        output_fn(RESUME_HINT)
        log.warning("Session aborted by the user")
        return 130
    except Exception as exc:  # model/API failures: short message here, traceback in the log
        log.exception("Session failed")
        output_fn(f"error: {type(exc).__name__}: {exc}")
        output_fn("Changes approved so far have been applied; nothing else was modified.")
        _report_usage(usage, output_fn)
        output_fn(RESUME_HINT)
        return 1
    finally:
        session_id.reset(session_token)


def _env_flag(name: str) -> bool:
    """True for an environment variable set to 1/true/yes/on (case-insensitive)."""
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive number, got {value}")
    return number


def main(
    argv: Sequence[str] | None = None,
    *,
    model_factory: Callable[[str], BaseChatModel] = init_chat_model,
) -> int:
    # API keys and FILE_ORGANIZER_MODEL may live in ./.env; real env vars take precedence.
    load_dotenv(Path.cwd() / ".env", override=False)

    parser = argparse.ArgumentParser(
        prog="file-organizer",
        description="Sort a folder's files into category sub-directories based on their "
        "content. Every filesystem change must be approved by you.",
    )
    parser.add_argument("folder", help="folder to organize")
    parser.add_argument(
        "--model",
        default=os.environ.get("FILE_ORGANIZER_MODEL", DEFAULT_MODEL),
        help="chat model as 'provider:model' (default: $FILE_ORGANIZER_MODEL or %(default)s)",
    )
    parser.add_argument(
        "--summary-model",
        default=os.environ.get("FILE_ORGANIZER_SUMMARY_MODEL"),
        help="cheap chat model that summarizes old messages (default: "
        "$FILE_ORGANIZER_SUMMARY_MODEL, else a cheap model of the main model's provider: "
        + ", ".join(CHEAP_MODELS.values()) + ")",
    )
    parser.add_argument(
        "--summarize-at",
        type=_positive_int,
        # A string default goes through type= too, so a bad env value is reported nicely.
        default=os.environ.get("FILE_ORGANIZER_SUMMARIZE_AT", str(ContextPolicy.summarize_at_tokens)),
        metavar="TOKENS",
        help="summarize older messages once the history reaches about this many tokens "
        "(default: $FILE_ORGANIZER_SUMMARIZE_AT or %(default)s)",
    )
    parser.add_argument(
        "--max-context-tokens",
        type=_positive_int,
        default=os.environ.get("FILE_ORGANIZER_MAX_CONTEXT_TOKENS", str(ContextPolicy.max_input_tokens)),
        metavar="TOKENS",
        help="hard cap per LLM request; older content is shortened or left out beyond it. "
        "Must be at least twice --summarize-at "
        "(default: $FILE_ORGANIZER_MAX_CONTEXT_TOKENS or %(default)s)",
    )
    parser.add_argument(
        "--log-file",
        default=os.environ.get("FILE_ORGANIZER_LOG_FILE", DEFAULT_LOG_FILE),
        help="where to write the log (default: $FILE_ORGANIZER_LOG_FILE or %(default)s)",
    )
    parser.add_argument(
        "--log-level",
        default=os.environ.get("FILE_ORGANIZER_LOG_LEVEL", DEFAULT_LOG_LEVEL),
        choices=LOG_LEVELS,
        type=str.upper,
        help="log detail (default: $FILE_ORGANIZER_LOG_LEVEL or %(default)s)",
    )
    parser.add_argument(
        "--log-format",
        default=os.environ.get("FILE_ORGANIZER_LOG_FORMAT", DEFAULT_LOG_FORMAT),
        choices=LOG_FORMATS,
        type=str.lower,
        help="'text' for reading, 'json' (one object per line) for log tools like ELK "
        "(default: $FILE_ORGANIZER_LOG_FORMAT or %(default)s)",
    )
    parser.add_argument(
        "--state-db",
        default=os.environ.get("FILE_ORGANIZER_STATE_DB", DEFAULT_STATE_DB),
        help="SQLite file where progress is saved for resuming "
        "(default: $FILE_ORGANIZER_STATE_DB or %(default)s)",
    )
    parser.add_argument(
        "--phoenix",
        action=argparse.BooleanOptionalAction,
        default=_env_flag("FILE_ORGANIZER_PHOENIX"),
        help="trace the agent (LLM calls, tools, graph steps) to Arize Phoenix; needs "
        "'uv sync --extra phoenix' and a running Phoenix "
        "(docker compose -f docker/phoenix/compose.yaml up -d) "
        "(default: $FILE_ORGANIZER_PHOENIX or off)",
    )
    parser.add_argument(
        "--phoenix-endpoint",
        default=os.environ.get("PHOENIX_COLLECTOR_ENDPOINT", DEFAULT_PHOENIX_ENDPOINT),
        metavar="URL",
        help="Phoenix base URL (default: $PHOENIX_COLLECTOR_ENDPOINT or %(default)s)",
    )
    parser.add_argument(
        "--phoenix-project",
        default=os.environ.get("PHOENIX_PROJECT_NAME", DEFAULT_PHOENIX_PROJECT),
        metavar="NAME",
        help="Phoenix project for the traces (default: $PHOENIX_PROJECT_NAME or %(default)s)",
    )
    args = parser.parse_args(argv)
    try:  # check that the context limits fit together before doing anything
        policy = ContextPolicy(summarize_at_tokens=args.summarize_at, max_input_tokens=args.max_context_tokens)
    except ValueError as exc:
        parser.error(str(exc))  # exits with code 2

    stop_logging = configure_logging(args.log_file, args.log_level, args.log_format)
    stop_tracing: Callable[[], None] | None = None
    try:
        if args.phoenix:
            try:
                stop_tracing = configure_tracing(args.phoenix_endpoint, args.phoenix_project)
            except TracingUnavailable as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 2
            except Exception as exc:  # e.g. a malformed endpoint URL
                log.exception("Could not set up Phoenix tracing")
                print(f"error: could not set up Phoenix tracing: {exc}", file=sys.stderr)
                return 2
            print(f"Tracing to Phoenix: {args.phoenix_endpoint} (project {args.phoenix_project!r})")
        Workspace(args.folder)  # validate before spending anything on a model
        model_name = args.model
        model = model_factory(model_name)  # e.g. init_chat_model("anthropic:claude-...")
        model_name = args.summary_model or default_summary_model(args.model)
        summary_model = model if model_name == args.model else model_factory(model_name)
        log.info("Using model %s, summarizing with %s", args.model, model_name)
        return run(args.folder, model, summary_model=summary_model, policy=policy,
                   state_db=args.state_db)
    except WorkspaceError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # missing provider package, bad name, missing API key, ...
        log.exception("Could not create model %s", model_name)
        print(f"error: could not create model '{model_name}': {exc}", file=sys.stderr)
        return 2
    finally:
        if stop_tracing is not None:
            try:
                stop_tracing()  # send spans still buffered to Phoenix
            except Exception:  # Phoenix down: never fail the session over its traces
                log.warning("Could not flush traces to Phoenix", exc_info=True)
        stop_logging()  # flush queued log records before exiting


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
