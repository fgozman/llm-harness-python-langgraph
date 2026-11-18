"""Keeping the model's context small and its knowledge of human decisions reliable.

Three layers, cheapest first, with limits that fit together (see ``ContextPolicy``):

1. Summarization (persistent, in the graph): when the history grows past
   ``summarize_at_tokens``, a cheap model condenses the older messages into one summary
   message, and the old messages are deleted from the state (and thus from every later
   checkpoint). The newest ``keep_tokens`` (half the threshold by default) stay
   verbatim, so the history drops well below the threshold and the next summary only
   happens after real growth. LangChain's ``SummarizationMiddleware`` implements the
   hard parts (where to cut without separating a tool call from its result, retries,
   context-overflow fallback); its ``before_model`` hook runs as a node of this graph.
   A summary is skipped when it would condense too little new content, e.g. when one
   big batch of tool results fills the whole kept window.
2. Trimming (transient, at the LLM call): a safety net that only shortens what is
   *sent*, never the state. It first shortens old tool results to a preview, and only
   then drops the oldest turns, telling the model that it did. The newest turn (the
   results the model is waiting for) is always sent in full.
3. A decision log in the LangGraph store (long-term memory): every finalized approval
   or rejection is written there and shown to the model in the system prompt on every
   call. Summaries are lossy and trimming drops messages, but "the human rejected X"
   must never be forgotten.

Both layers count tokens with the same approximate counter (``token_counter_for``).
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from typing import Any, Literal, TypedDict

from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately, trim_messages
from langgraph.runtime import Runtime
from langgraph.store.base import BaseStore

from .approval import _safe
from .prompts import SUMMARY_PROMPT
from .usage import TokenUsage, collect_usage

log = logging.getLogger(__name__)

TokenCounter = Callable[[Sequence[AnyMessage]], int]

#: Characters of an old tool result kept when the safety net has to shorten it.
PREVIEW_CHARS = 300


@dataclass(frozen=True)
class ContextPolicy:
    """Context-size limits, in approximate tokens, checked for consistency.

    * ``summarize_at_tokens``: summarize once the history reaches this size.
    * ``keep_tokens``: recent history kept verbatim by a summary (default: half of
      ``summarize_at_tokens``). Must be below the threshold, or summarizing would not
      bring the history back under it and would repeat on every step.
    * ``max_input_tokens``: hard cap per LLM request (trimming safety net). Must be at
      least twice the threshold, leaving room for the growth of one step, or trimming
      would fire constantly and drop messages that no summary has captured yet.
    * ``min_summary_gain_tokens``: skip a summary that would condense less new content
      than this (default: a quarter of the threshold).
    """

    summarize_at_tokens: int = 60_000
    keep_tokens: int | None = None
    max_input_tokens: int = 150_000
    min_summary_gain_tokens: int | None = None
    max_decisions_in_prompt: int = 200  # newest decisions shown to the model

    def __post_init__(self) -> None:
        # frozen dataclass: derived defaults are filled in with object.__setattr__
        if self.keep_tokens is None:
            object.__setattr__(self, "keep_tokens", self.summarize_at_tokens // 2)
        if self.min_summary_gain_tokens is None:
            object.__setattr__(self, "min_summary_gain_tokens", self.summarize_at_tokens // 4)
        for name in ("summarize_at_tokens", "keep_tokens", "max_input_tokens", "max_decisions_in_prompt"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        if self.keep_tokens >= self.summarize_at_tokens:
            raise ValueError(
                f"keep_tokens ({self.keep_tokens}) must be below summarize_at_tokens "
                f"({self.summarize_at_tokens}), or every step would summarize again"
            )
        if self.max_input_tokens < 2 * self.summarize_at_tokens:
            raise ValueError(
                f"max_input_tokens ({self.max_input_tokens}) must be at least twice "
                f"summarize_at_tokens ({self.summarize_at_tokens}), or trimming would drop "
                "messages that were never summarized"
            )


def token_counter_for(model: BaseChatModel) -> TokenCounter:
    """Approximate token counter for ``model``'s provider (no API call).

    Claude's tokenizer averages about 3.3 characters per token (the estimate
    SummarizationMiddleware uses for Anthropic models); others about 4.
    """
    if model._llm_type.startswith("anthropic"):
        return partial(count_tokens_approximately, chars_per_token=3.3)
    return count_tokens_approximately


# ---- 1. Summarization ------------------------------------------------------------------


def model_label(model: BaseChatModel) -> str:
    """Short name of a chat model for logs, e.g. 'claude-haiku-5-5'."""
    return str(getattr(model, "model", None) or getattr(model, "model_name", None) or model._llm_type)


def is_summary(message: AnyMessage) -> bool:
    """True for the message that SummarizationMiddleware puts in place of old history."""
    return message.additional_kwargs.get("lc_source") == "summarization"


def summary_gain(messages: Sequence[AnyMessage], keep_tokens: int, token_counter: TokenCounter) -> int:
    """Tokens of NEW content (not an earlier summary) that a summary would condense.

    Mirrors where the middleware cuts: keep the newest ``keep_tokens``, and move the cut
    back to the AI message whose tool results it would otherwise split.
    """
    sizes = [token_counter([m]) for m in messages]
    cutoff, kept = len(messages), 0
    while cutoff > 0 and kept + sizes[cutoff - 1] <= keep_tokens:
        cutoff -= 1
        kept += sizes[cutoff]
    if cutoff < len(messages) and isinstance(messages[cutoff], ToolMessage):
        while cutoff > 0 and not isinstance(messages[cutoff], AIMessage):
            cutoff -= 1
    return sum(size for m, size in zip(messages[:cutoff], sizes) if not is_summary(m))


class SummaryStats(TypedDict):
    """What the latest summary did to the history (approximate tokens)."""

    summarized_messages: int
    kept_messages: int
    tokens_before: int
    tokens_after: int  # summary + kept messages (without the system prompt)


def make_summarize_node(
    model: BaseChatModel,
    summary_model: BaseChatModel | None,
    policy: ContextPolicy,
    token_counter: TokenCounter,
) -> Callable[[dict[str, Any], Runtime], dict[str, Any]]:
    """Build the graph node that summarizes old messages with ``summary_model`` (cheap).

    ``model`` is the main model, whose context window is being managed (the middleware
    reads its profile); ``summarizer=`` makes the cheap model write the summaries.
    ``summary_model=None`` lets the main model summarize too.
    """
    summarizer = SummarizationMiddleware(
        model=model,
        summarizer=summary_model,
        trigger=("tokens", policy.summarize_at_tokens),
        keep=("tokens", policy.keep_tokens),
        token_counter=token_counter,  # the same counter as the trimming safety net
        summary_prompt=SUMMARY_PROMPT,
        # The middleware pre-trims what it summarizes with start_on="human". A tool loop
        # has only one human message (the task), so that would usually leave nothing to
        # summarize. The cheap model gets everything; on a context overflow the
        # middleware retries with the middle of the history cut out.
        trim_tokens_to_summarize=None,
    )
    label = model_label(summary_model or model)

    def summarize(state: dict[str, Any], runtime: Runtime) -> dict[str, Any]:
        """Node: replace older messages with a summary once the history is too long."""
        messages = state["messages"]
        total = token_counter(messages)
        if total < policy.summarize_at_tokens:
            return {}
        gain = summary_gain(messages, policy.keep_tokens, token_counter)
        if gain < policy.min_summary_gain_tokens:
            log.info("Summary skipped: only ~%d new tokens to condense (minimum %d); history ~%d tokens",
                     gain, policy.min_summary_gain_tokens, total,
                     extra={"summary_gain_tokens": gain, "approx_tokens": total})
            return {}
        session = TokenUsage.from_dict(state.get("summary_usage"))
        started = time.perf_counter()
        # Capture the summary call's token usage: the middleware doesn't return the
        # model's reply, but a callback sees every chat model call made in this block.
        with collect_usage() as usage:
            update = summarizer.before_model(state, runtime)
        if not update:  # nothing safe to cut
            return {}
        elapsed = time.perf_counter() - started
        session.merge(usage)
        # update["messages"] = [RemoveMessage(REMOVE_ALL_MESSAGES), summary, *kept]:
        # the add_messages reducer clears the history, then appends the rest.
        kept = len(update["messages"]) - 2
        summarized = len(messages) - kept
        after = token_counter(update["messages"][1:])
        tokens = (
            f"tokens input={usage.input_tokens} output={usage.output_tokens} total={usage.total_tokens} "
            f"(summarizer session: input={session.input_tokens} output={session.output_tokens} "
            f"total={session.total_tokens})"
            if usage.reported else "token usage not reported"
        )
        log.info(
            "Summarizer call #%d (%s) done in %.2fs: summarized %d older messages, kept %d "
            "(history ~%d -> ~%d tokens); %s",
            session.calls, label, elapsed, summarized, kept, total, after, tokens,
            extra={"summary_call": session.calls, "summary_model": label, "duration_s": round(elapsed, 3),
                   "summarized_messages": summarized, "kept_messages": kept,
                   "history_tokens_before": total, "history_tokens_after": after,
                   **usage.as_extra("summary_"), **session.as_extra("session_summary_")},
        )
        stats: SummaryStats = {"summarized_messages": summarized, "kept_messages": kept,
                               "tokens_before": total, "tokens_after": after}
        return {**update, "summary_usage": usage.as_dict(), "last_summary": stats}

    return summarize


# ---- 2. Trimming safety net ------------------------------------------------------------


def _shorten(message: ToolMessage, preview_chars: int) -> ToolMessage:
    text = message.text
    note = f"\n[... {len(text) - preview_chars} more characters omitted from this request to fit the context window]"
    return message.model_copy(update={"content": text[:preview_chars] + note})


def fit_context(
    messages: list[AnyMessage],
    max_tokens: int,
    token_counter: TokenCounter = count_tokens_approximately,
    preview_chars: int = PREVIEW_CHARS,
) -> list[AnyMessage]:
    """Return ``messages`` reduced to about ``max_tokens`` for one LLM request.

    1. Old tool results (all but the newest turn's) are shortened to a preview, which
       keeps e.g. a file's path and first lines: enough to remember what it was.
    2. If that's not enough, the oldest turns are dropped. The leading system message(s)
       and the first human message (the task, or the summary that replaced it) stay, the
       kept tail starts at an AI message (no orphaned tool results), and a note in the
       system message tells the model that messages were left out.
    The newest turn is always sent in full: better a provider error than silently
    dropping the results the model is waiting for.
    """
    total = token_counter(messages)
    if total <= max_tokens:
        return messages
    last_ai = next((i for i in range(len(messages) - 1, -1, -1) if isinstance(messages[i], AIMessage)), len(messages))
    reduced = [
        _shorten(m, preview_chars) if i < last_ai and isinstance(m, ToolMessage) and len(m.text) > preview_chars else m
        for i, m in enumerate(messages)
    ]
    shortened = sum(a is not b for a, b in zip(messages, reduced))
    dropped = 0
    if token_counter(reduced) > max_tokens:
        n_head = 0
        while n_head < len(reduced) and isinstance(reduced[n_head], SystemMessage):
            n_head += 1
        if n_head < len(reduced) and isinstance(reduced[n_head], HumanMessage):
            n_head += 1
        head, rest = reduced[:n_head], reduced[n_head:]
        budget = max_tokens - token_counter(head) - 50  # ~50 tokens for the note below
        tail: list[AnyMessage] = []
        if budget > 0 and rest:
            tail = trim_messages(rest, max_tokens=budget, token_counter=token_counter, strategy="last", start_on="ai")
        if not tail and rest:
            newest = next((i for i in range(len(rest) - 1, -1, -1) if isinstance(rest[i], AIMessage)), None)
            tail = rest[newest:] if newest is not None else rest[-1:]
        dropped = len(rest) - len(tail)
        if dropped and head and isinstance(head[0], SystemMessage):
            head[0] = SystemMessage(
                f"{head[0].text}\n\n[Note: {dropped} earlier messages were left out of this request to "
                "fit the context window. Rely on the summary and the decision list above for them.]"
            )
        reduced = [*head, *tail]
    log.warning(
        "Context safety net: ~%d tokens > limit %d; shortened %d old tool results, dropped %d messages "
        "(now ~%d tokens)",
        total, max_tokens, shortened, dropped, token_counter(reduced),
        extra={"approx_tokens": total, "shortened_tool_results": shortened, "trimmed_messages": dropped},
    )
    return reduced


# ---- 3. Decision log in the store ------------------------------------------------------


class DecisionRecord(TypedDict):
    """A finalized human decision about one filesystem change, as stored."""

    tool: str
    args: dict[str, Any]
    description: str
    approved: bool
    feedback: str | None
    outcome: Literal["applied", "failed", "rejected"]
    result: str  # what the tool (or the rejection) reported to the model
    decided_at: str  # ISO-8601 UTC, also used for ordering


def decisions_namespace(thread_id: str) -> tuple[str, ...]:
    """Store namespace of one thread's decisions.

    Store namespaces are tuples of labels (like folders) and labels may not contain
    '.', which paths often do, hence a hash of the thread id.
    """
    return ("file_organizer", "decisions", hashlib.sha256(thread_id.encode()).hexdigest()[:16])


class DecisionLog:
    """Finalized decisions of one thread, kept in a LangGraph ``BaseStore``.

    Writes don't set a TTL themselves, so the store's ``default_ttl`` applies (if it
    supports TTLs at all; ``InMemoryStore`` doesn't, and keeps items until deleted).
    """

    def __init__(self, store: BaseStore, thread_id: str) -> None:
        self.store = store
        self.thread_id = thread_id
        self.namespace = decisions_namespace(thread_id)

    def record(self, tool_call_id: str, record: DecisionRecord) -> None:
        # put() is an upsert keyed by tool_call_id: re-running a batch after a crash
        # overwrites the record instead of duplicating it.
        self.store.put(self.namespace, tool_call_id, {**record, "thread_id": self.thread_id})

    def records(self, limit: int = 1000, *, refresh_ttl: bool | None = None) -> list[DecisionRecord]:
        """The newest ``limit`` decisions, oldest first.

        ``refresh_ttl=False`` reads without extending the items' TTL (which would be a
        database write); ``None`` follows the store's ``refresh_on_read`` setting.
        """
        items = self.store.search(self.namespace, limit=limit, refresh_ttl=refresh_ttl)  # default limit is 10!
        return sorted((item.value for item in items), key=lambda v: v["decided_at"])  # type: ignore[return-value]

    def clear(self) -> int:
        removed = 0
        while items := self.store.search(self.namespace, limit=100):
            for item in items:
                self.store.delete(self.namespace, item.key)
            removed += len(items)
        return removed


def decision_line(record: DecisionRecord) -> str:
    """One line of the decision log shown to the model."""
    if record["outcome"] == "applied":
        return f"- APPLIED: {record['description']}"
    if record["outcome"] == "failed":
        return f"- APPROVED BUT FAILED: {record['description']} ({_safe(record['result'], 150)})"
    line = f"- REJECTED: {record['description']}"
    return f"{line} (feedback: {_safe(record['feedback'])})" if record["feedback"] else line


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")
