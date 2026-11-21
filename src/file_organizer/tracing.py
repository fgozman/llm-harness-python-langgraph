"""Optional tracing of the LangGraph agent to Arize Phoenix (OpenTelemetry / OpenInference).

One session of the CLI is ONE trace in Phoenix, even though the graph pauses for every
human approval: each pause ends a ``graph.stream()`` call and each answer starts a new
one, which the LangChain instrumentation would report as separate traces. So the CLI
wraps the whole session in a root span (``trace_session``); every graph run (the start
and each resume) becomes a child of it, with one span per node (summarize, agent,
review, tools) and the LLM calls (prompts, responses, token counts, latency) and tool
calls inside. Each human decision is an event on the root span.

A session resumed in a later process (after Ctrl-C or a crash) is a new trace, but it
has the same Phoenix session id (the folder's thread id), so all of them show up
together in Phoenix's "Sessions" tab.

The dependencies are an optional extra (``uv sync --extra phoenix``). Tracing is off
unless asked for, and the agent never fails because of it: if the packages are missing
or Phoenix is unreachable, spans are dropped and the run continues.

Phoenix runs locally in Docker (see ``docker/phoenix/compose.yaml``):
``docker compose -f docker/phoenix/compose.yaml up -d``, then open http://localhost:6006.
"""

from __future__ import annotations

import contextlib
import json
import logging
from collections.abc import Callable, Iterator
from typing import Any

log = logging.getLogger(__name__)

DEFAULT_PHOENIX_ENDPOINT = "http://localhost:6006"  # the Phoenix UI and OTLP/HTTP collector
DEFAULT_PHOENIX_PROJECT = "file-organizer"


# OpenInference attribute names (as plain strings, so this module imports without the extra).
_SPAN_KIND = "openinference.span.kind"
_SESSION_ID = "session.id"
_METADATA = "metadata"
_INPUT, _OUTPUT = "input.value", "output.value"

# The tracer of the active configure_tracing() call; None while tracing is off. It comes
# from Phoenix's own TracerProvider: the process-wide (global) provider is left alone.
_tracer: Any = None


class TracingUnavailable(RuntimeError):
    """The optional tracing packages are not installed."""


def traces_url(endpoint: str) -> str:
    """The OTLP/HTTP traces URL for a Phoenix base URL: http://host:6006 -> .../v1/traces."""
    endpoint = endpoint.rstrip("/")
    return endpoint if endpoint.endswith("/v1/traces") else f"{endpoint}/v1/traces"


def configure_tracing(
    endpoint: str = DEFAULT_PHOENIX_ENDPOINT,
    project_name: str = DEFAULT_PHOENIX_PROJECT,
    *,
    tracer_provider: Any = None,
) -> Callable[[], None]:
    """Send LangChain/LangGraph spans to Phoenix. Returns a ``stop`` function that flushes
    the spans still buffered and removes the instrumentation; call it before exiting.

    ``tracer_provider`` replaces the Phoenix exporter (tests pass one with an in-memory
    exporter). Raises ``TracingUnavailable`` if the ``phoenix`` extra is not installed.
    """
    global _tracer
    try:
        from openinference.instrumentation.langchain import LangChainInstrumentor
        if tracer_provider is None:
            from phoenix.otel import register
    except ImportError as exc:
        raise TracingUnavailable(
            "Phoenix tracing needs the optional packages: uv sync --extra phoenix"
        ) from exc

    if tracer_provider is None:
        url = traces_url(endpoint)
        tracer_provider = register(
            endpoint=url,
            project_name=project_name,
            batch=True,  # export in a background thread, never blocking the agent
            protocol="http/protobuf",
            set_global_tracer_provider=False,  # don't change process-wide OpenTelemetry state
            verbose=False,  # register() would otherwise print a banner into the CLI
        )
        log.info("Tracing to Phoenix at %s (project %r)", url, project_name)
    instrumentor = LangChainInstrumentor()  # covers LangGraph too (it runs on LangChain)
    instrumentor.instrument(tracer_provider=tracer_provider)
    _tracer = tracer_provider.get_tracer(__name__)

    def stop() -> None:
        global _tracer
        _tracer = None
        instrumentor.uninstrument()
        try:
            tracer_provider.force_flush(timeout_millis=5_000)
        finally:
            tracer_provider.shutdown()

    return stop


class SessionTrace:
    """The root span of one CLI session. Every method is a no-op while tracing is off."""

    def __init__(self, span: Any = None) -> None:
        self._span = span

    def decision(self, request: dict[str, Any], decision: dict[str, Any]) -> None:
        """Record the human's answer to one approval question as an event on the root span."""
        if self._span is not None:
            self._span.add_event("human decision", {
                "tool": str(request.get("tool", "")),
                "description": str(request.get("description", "")),
                "approved": bool(decision.get("approved")),
                "feedback": decision.get("feedback") or "",
            })

    def finish(self, output: str) -> None:
        """Mark the session as finished, with the agent's final message as its output."""
        if self._span is not None:
            from opentelemetry.trace import Status, StatusCode

            self._span.set_attribute(_OUTPUT, output)
            self._span.set_status(Status(StatusCode.OK))


@contextlib.contextmanager
def trace_session(session_id: str, *, name: str = "file-organizer", input: str = "",
                  metadata: dict[str, Any] | None = None) -> Iterator[SessionTrace]:
    """Run the block inside one root span, so all graph runs in it form a single trace.

    The span is the "current" OpenTelemetry span while the block runs. The LangChain
    instrumentation makes every top-level run (each ``graph.stream()`` call) a child of
    the current span; without this, each resume after an approval would be its own trace.
    All spans in the block also get ``session_id`` as their Phoenix session id.

    An exception leaving the block marks the session as failed (Ctrl-C: aborted) and is
    re-raised. A no-op (yielding a no-op handle) while tracing is off.
    """
    tracer = _tracer
    if tracer is None:
        yield SessionTrace()
        return

    from openinference.instrumentation import using_attributes
    from opentelemetry.trace import Status, StatusCode

    metadata = metadata or {}
    attributes = {_SPAN_KIND: "AGENT", _SESSION_ID: session_id,
                  _METADATA: json.dumps(metadata, default=str)}
    if input:
        attributes[_INPUT] = input
    with (
        using_attributes(session_id=session_id, metadata=metadata),
        tracer.start_as_current_span(name, attributes=attributes, record_exception=False,
                                     set_status_on_exception=False) as span,
    ):
        try:
            yield SessionTrace(span)
        except (KeyboardInterrupt, EOFError):
            span.set_attribute(_OUTPUT, "Aborted by the user")
            span.add_event("aborted by the user")
            raise
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, f"{type(exc).__name__}: {exc}"))
            raise
