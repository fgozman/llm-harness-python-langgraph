"""Logging configuration for the file-organizer command line.

Follows the standard-library recommendations:
* Modules only create loggers (``logging.getLogger(__name__)``); this module, called once
  from ``main()``, is the only place that configures handlers.
* The configuration is declarative (``logging.config.dictConfig``).
* Non-blocking: the root logger has a QueueHandler, and a background QueueListener thread
  does the slow disk writes. Python 3.12+ builds both from the "queue_handler" entry below.
* The file rotates (1 MB x 3 backups) so it can't grow without limit.
* Timestamps are ISO-8601 in UTC; each line shows level, logger, and source location.
* Two formats: "text" for people, "json" (one JSON object per line) for log tools such
  as ELK, Datadog, or Loki, which can read the fields without parsing the text.
* Logs go to a file only, so they never mix with the interactive prompts on the terminal.
"""

from __future__ import annotations

import copy
import json
import logging
import logging.config
import time
from collections.abc import Callable
from contextvars import ContextVar
from logging.handlers import QueueHandler
from pathlib import Path
from typing import Any

DEFAULT_LOG_FILE = ".file-organizer.log"  # hidden, so the agent never lists or moves it
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_LOG_FORMAT = "text"
LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
LOG_FORMATS = ["text", "json"]

# e.g. 2026-11-17T18:03:13.832Z INFO     file_organizer.workspace:move_file:108 - Moved a -> b
TEXT_FORMAT = "%(asctime)s %(levelname)-8s %(name)s:%(funcName)s:%(lineno)d - %(message)s"


class UTCFormatter(logging.Formatter):
    """Formats ``asctime`` as ISO-8601 UTC with milliseconds: 2026-11-17T18:03:13.832Z."""

    converter = time.gmtime
    default_time_format = "%Y-%m-%dT%H:%M:%S"
    default_msec_format = "%s.%03dZ"


# Correlation ids, set by the CLI; ContextFilter copies them onto every record.
# run_id:    a fresh UUID per process run (the "run_id" metadata of the Phoenix root span).
# thread_id: the LangGraph thread, one per folder; it outlives a run (resume after Ctrl-C)
#            and is the Phoenix session id.
run_id: ContextVar[str | None] = ContextVar("run_id", default=None)
thread_id: ContextVar[str | None] = ContextVar("thread_id", default=None)

try:  # OpenTelemetry comes with the optional "phoenix" extra
    from opentelemetry.trace import get_current_span
except ImportError:  # pragma: no cover - depends on the installed extras
    get_current_span = None


def _trace_id() -> str | None:
    """The id of the current OpenTelemetry trace, in the hex form Phoenix shows; None
    while tracing is off. (No span id: the LangChain instrumentation doesn't make its
    node/LLM/tool spans current, so it would always be the session's root span.)"""
    if get_current_span is None:
        return None
    ctx = get_current_span().get_span_context()
    return format(ctx.trace_id, "032x") if ctx.is_valid else None


class ContextFilter(logging.Filter):
    """Adds ``run_id``, ``thread_id`` and, while tracing, ``trace_id`` to each
    record, so one run can be filtered out of a shared log and a line can be looked up
    in Phoenix. Runs in the logging thread (before queueing), where these are current."""

    def filter(self, record: logging.LogRecord) -> bool:
        for name, var in (("run_id", run_id), ("thread_id", thread_id)):
            if (value := var.get()) is not None:
                setattr(record, name, value)
        if (trace := _trace_id()) is not None:
            record.trace_id = trace
        return True  # never drops records, only enriches them


# Attributes that every LogRecord has; anything else was passed by the caller via `extra=`.
_STANDARD_ATTRS = set(vars(logging.LogRecord("", 0, "", 0, "", None, None))) | {"message", "asctime"}


class JSONFormatter(UTCFormatter):
    """One JSON object per line ("JSON Lines"), e.g.

    {"timestamp": "2026-11-17T18:03:13.832Z", "level": "INFO", "logger": "file_organizer.cli",
     "function": "run", "line": 96, "message": "Session finished"}

    A traceback goes into an "exception" field, and values passed with
    ``log.info(..., extra={"key": value})`` become extra top-level fields.
    """

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "function": record.funcName,
            "line": record.lineno,
            "message": record.getMessage(),
        }
        if record.exc_info and not record.exc_text:
            record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            entry["exception"] = record.exc_text
        if record.stack_info:
            entry["stack"] = record.stack_info
        entry |= {k: v for k, v in vars(record).items() if k not in _STANDARD_ATTRS}
        return json.dumps(entry, default=str, ensure_ascii=False)


class TracebackKeepingQueueHandler(QueueHandler):
    """The standard QueueHandler pastes the traceback into the message text. This one
    keeps it in ``exc_text`` (a plain string, safe to queue) so the file's formatter
    decides how to show it: appended in text mode, its own field in JSON mode."""

    def prepare(self, record: logging.LogRecord) -> logging.LogRecord:
        record = copy.copy(record)  # don't change the record other handlers see
        record.msg, record.args = record.getMessage(), None  # merge args (may not be picklable)
        if record.exc_info:
            record.exc_text = logging.Formatter().formatException(record.exc_info)
            record.exc_info = None  # traceback objects can't be queued between processes
        return record


def logging_config(log_file: str | Path, level: str, fmt: str = DEFAULT_LOG_FORMAT) -> dict[str, Any]:
    """The dictConfig schema: root -> queue_handler -> (background thread) -> file."""
    return {
        "version": 1,
        "disable_existing_loggers": False,  # keep loggers that modules created at import
        "formatters": {
            "text": {"()": UTCFormatter, "format": TEXT_FORMAT},
            "json": {"()": JSONFormatter},
        },
        "filters": {
            "context": {"()": ContextFilter},
        },
        "handlers": {
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "filename": str(log_file),
                "maxBytes": 1_000_000,
                "backupCount": 3,
                "encoding": "utf-8",
                "formatter": fmt,
            },
            "queue_handler": {
                "class": TracebackKeepingQueueHandler,
                "handlers": ["file"],  # dictConfig wraps these in a QueueListener
                "filters": ["context"],  # runs in the caller's thread, before queueing
                "respect_handler_level": True,
            },
        },
        "loggers": {
            # Our own code logs at the requested level ...
            "file_organizer": {"level": level},
        },
        # ... third-party libraries (langchain, httpx, ...) only from WARNING up.
        "root": {"level": "WARNING", "handlers": ["queue_handler"]},
    }


def configure_logging(
    log_file: str | Path = DEFAULT_LOG_FILE,
    level: str = DEFAULT_LOG_LEVEL,
    fmt: str = DEFAULT_LOG_FORMAT,
) -> Callable[[], None]:
    """Apply the configuration and start the background writer.

    Returns a ``stop`` function that writes any queued records and stops the thread.
    Call it before the program exits.
    """
    log_path = Path(log_file).expanduser()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.config.dictConfig(logging_config(log_path, level.upper(), fmt.lower()))
    logging.captureWarnings(True)  # warnings.warn(...) messages go to the log as well

    queue_handler = logging.getHandlerByName("queue_handler")
    queue_handler.listener.start()  # dictConfig creates the listener but doesn't start it

    def stop() -> None:
        queue_handler.listener.stop()  # writes everything still queued
        logging.captureWarnings(False)
        logging.getLogger().removeHandler(queue_handler)
        for handler in (queue_handler, *queue_handler.listener.handlers):
            handler.close()

    return stop
