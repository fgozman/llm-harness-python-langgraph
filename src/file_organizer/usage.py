"""Token accounting for LLM calls (from LangChain's ``AIMessage.usage_metadata``)."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, fields
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import LLMResult
from langchain_core.tracers.context import register_configure_hook


@dataclass
class TokenUsage:
    """Running token totals over LLM replies.

    ``calls`` counts every LLM reply; ``reported`` only those whose provider sent usage
    numbers (a few models/fakes don't), so totals can be shown as unknown instead of 0.

    The totals are kept in the graph state as a plain dict (``as_dict``/``from_dict``),
    not recomputed from the message history: summarization deletes old messages, and
    their token counts must not disappear with them.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    calls: int = 0
    reported: int = 0

    def add(self, message: BaseMessage) -> None:
        if not isinstance(message, AIMessage):
            return
        self.calls += 1
        if usage := message.usage_metadata:
            self.add_counts(usage)

    def add_counts(self, usage: Mapping[str, int]) -> None:
        """Add one reported usage dict ({"input_tokens", "output_tokens", "total_tokens"})."""
        self.reported += 1
        self.input_tokens += usage.get("input_tokens", 0)
        self.output_tokens += usage.get("output_tokens", 0)
        self.total_tokens += usage.get("total_tokens", 0)

    def merge(self, other: TokenUsage) -> None:
        for f in fields(self):
            setattr(self, f.name, getattr(self, f.name) + getattr(other, f.name))

    @classmethod
    def of(cls, messages: Iterable[BaseMessage]) -> TokenUsage:
        totals = cls()
        for message in messages:
            totals.add(message)
        return totals

    def as_dict(self) -> dict[str, int]:
        """Serializable form, stored in the graph state (and so in the checkpoint)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, int] | None) -> TokenUsage:
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (data or {}).items() if k in names})

    def summary(self) -> str:
        """e.g. '1,234 in + 56 out = 1,290 tokens in 3 LLM calls'."""
        calls = f"{self.calls} LLM call{'' if self.calls == 1 else 's'}"
        if not self.reported:
            return f"token usage not reported in {calls}"
        return (
            f"{self.input_tokens:,} in + {self.output_tokens:,} out = "
            f"{self.total_tokens:,} tokens in {calls}"
        )

    def as_extra(self, prefix: str = "") -> dict[str, int]:
        """Fields for ``log.info(..., extra=...)`` (top-level keys in JSON logs)."""
        return {
            f"{prefix}input_tokens": self.input_tokens,
            f"{prefix}output_tokens": self.output_tokens,
            f"{prefix}total_tokens": self.total_tokens,
            f"{prefix}llm_calls": self.calls,
        }


def add_usage(current: dict[str, int] | None, update: dict[str, int] | None) -> dict[str, int]:
    """LangGraph reducer for a usage channel: nodes return only the NEW counts, which are
    added to the running totals (like add_messages appends instead of overwriting)."""
    total = TokenUsage.from_dict(current)
    total.merge(TokenUsage.from_dict(update))
    return total.as_dict()


class _UsageCollector(BaseCallbackHandler):
    """Callback handler that adds up the usage of every chat model reply it sees."""

    def __init__(self) -> None:
        self.usage = TokenUsage()

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        for generations in response.generations:
            for generation in generations:
                if isinstance(message := getattr(generation, "message", None), AIMessage):
                    self.usage.add(message)


# A handler stored in this context variable is attached to every LangChain call made in
# the same context, including calls made deep inside library code with their own config
# (the same mechanism as langchain_core's get_usage_metadata_callback, which however
# skips replies without a model name and registers a new hook per use).
_collector: ContextVar[_UsageCollector | None] = ContextVar("file_organizer_usage", default=None)
register_configure_hook(_collector, inheritable=True)


@contextmanager
def collect_usage() -> Iterator[TokenUsage]:
    """Yield a TokenUsage that receives every LLM reply made inside the ``with`` block."""
    collector = _UsageCollector()
    token = _collector.set(collector)
    try:
        yield collector.usage
    finally:
        _collector.reset(token)
