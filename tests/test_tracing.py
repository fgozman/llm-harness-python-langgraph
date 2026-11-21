"""Tests for the optional Arize Phoenix tracing (no Phoenix server needed)."""

import json
from pathlib import Path

import pytest

from file_organizer import cli
from file_organizer.cli import main, run
from file_organizer.tracing import (
    TracingUnavailable,
    configure_tracing,
    trace_session,
    traces_url,
)

from .conftest import ScriptedChatModel, ai, call
from .test_cli import scripted_input


@pytest.mark.parametrize("endpoint", [
    "http://localhost:6006", "http://localhost:6006/", "http://localhost:6006/v1/traces",
])
def test_traces_url(endpoint):
    assert traces_url(endpoint) == "http://localhost:6006/v1/traces"


def _model(name):
    return ScriptedChatModel(responses=[ai(content="Nothing to do.")])


@pytest.fixture
def fake_tracing(monkeypatch):
    """Replaces configure_tracing in the CLI; records its arguments and the stop() call."""
    calls: dict = {"args": None, "stopped": False}

    def fake_configure(endpoint, project):
        calls["args"] = (endpoint, project)
        return lambda: calls.update(stopped=True)

    monkeypatch.setattr(cli, "configure_tracing", fake_configure)
    return calls


def test_tracing_is_off_by_default(folder: Path, fake_tracing):
    assert main([str(folder)], model_factory=_model) == 0
    assert fake_tracing["args"] is None


def test_phoenix_flag_enables_tracing_and_flushes_at_exit(folder: Path, fake_tracing):
    assert main([str(folder), "--phoenix", "--phoenix-endpoint", "http://phoenix:6006",
                 "--phoenix-project", "demo"], model_factory=_model) == 0
    assert fake_tracing == {"args": ("http://phoenix:6006", "demo"), "stopped": True}


def test_tracing_settings_from_env(folder: Path, fake_tracing, monkeypatch):
    monkeypatch.setenv("FILE_ORGANIZER_PHOENIX", "true")
    monkeypatch.setenv("PHOENIX_COLLECTOR_ENDPOINT", "http://env:6006")
    monkeypatch.setenv("PHOENIX_PROJECT_NAME", "from-env")
    main([str(folder)], model_factory=_model)
    assert fake_tracing["args"] == ("http://env:6006", "from-env")


def test_no_phoenix_overrides_env(folder: Path, fake_tracing, monkeypatch):
    monkeypatch.setenv("FILE_ORGANIZER_PHOENIX", "1")
    main([str(folder), "--no-phoenix"], model_factory=_model)
    assert fake_tracing["args"] is None


def test_missing_packages_are_reported(folder: Path, monkeypatch, capsys):
    def unavailable(endpoint, project):
        raise TracingUnavailable("Phoenix tracing needs the optional packages")

    monkeypatch.setattr(cli, "configure_tracing", unavailable)
    assert main([str(folder), "--phoenix"], model_factory=_model) == 2
    assert "optional packages" in capsys.readouterr().err


def test_a_failing_flush_does_not_fail_the_session(folder: Path, monkeypatch):
    def broken_stop():
        raise ConnectionError("Phoenix is down")

    monkeypatch.setattr(cli, "configure_tracing", lambda endpoint, project: broken_stop)
    assert main([str(folder), "--phoenix"], model_factory=_model) == 0


@pytest.fixture
def captured_spans():
    """configure_tracing() with an in-memory exporter instead of Phoenix."""
    pytest.importorskip("openinference.instrumentation.langchain")
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    stop = configure_tracing(tracer_provider=provider)
    yield exporter
    stop()


def _session_model():
    return ScriptedChatModel(responses=[
        ai(call("read_file", path="invoice.txt")),
        ai(call("create_directory", path="finance")),
        ai(call("move_file", source="invoice.txt", destination="finance")),
        ai(content="Moved the invoice to finance."),
    ])


def test_a_session_with_approvals_is_one_trace(folder: Path, captured_spans):
    answers = scripted_input("y", "no, use invoices")  # two pauses -> three graph runs
    assert run(folder, _session_model(), input_fn=answers, output_fn=lambda *_: None) == 0

    spans = captured_spans.get_finished_spans()
    assert len({span.context.trace_id for span in spans}) == 1  # one trace, not one per resume

    roots = [span for span in spans if span.parent is None]
    assert [root.name for root in roots] == ["file-organizer"]
    root = roots[0]
    assert root.attributes["session.id"] == f"organize:{folder.resolve()}"
    assert root.attributes["output.value"] == "Moved the invoice to finance."
    assert json.loads(root.attributes["metadata"])["folder"] == str(folder.resolve())

    # The start and each resume are graph runs directly under the root.
    graph_runs = [span.name for span in spans if span.parent and span.parent.span_id == root.context.span_id]
    assert graph_runs[0] == "organize"
    assert graph_runs[1:] == ["resume: approved Create directory 'finance'",
                              "resume: rejected Move 'invoice.txt' -> 'finance'"]
    names = {span.name for span in spans}
    assert {"summarize", "agent", "review", "tools", "read_file", "create_directory",
            "ScriptedChatModel"} <= names

    decisions = [dict(event.attributes) for event in root.events if event.name == "human decision"]
    assert [(d["approved"], d["feedback"]) for d in decisions] == [(True, ""), (False, "no, use invoices")]


def test_an_aborted_session_is_marked_on_the_root_span(folder: Path, captured_spans):
    def ctrl_c(prompt=""):
        raise KeyboardInterrupt

    assert run(folder, _session_model(), input_fn=ctrl_c, output_fn=lambda *_: None) == 130
    root = next(s for s in captured_spans.get_finished_spans() if s.parent is None)
    assert root.attributes["output.value"] == "Aborted by the user"


def test_a_failed_session_is_an_error_on_the_root_span(folder: Path, captured_spans):
    from opentelemetry.trace import StatusCode

    model = ScriptedChatModel(responses=[])  # raises on the first call
    assert run(folder, model, input_fn=scripted_input(), output_fn=lambda *_: None) == 1
    spans = captured_spans.get_finished_spans()
    assert len({span.context.trace_id for span in spans}) == 1
    root = next(s for s in spans if s.parent is None)
    assert root.status.status_code is StatusCode.ERROR


def test_trace_session_is_a_no_op_while_tracing_is_off():
    with trace_session("s") as trace:
        trace.decision({"description": "x"}, {"approved": True, "feedback": None})
        trace.finish("done")
