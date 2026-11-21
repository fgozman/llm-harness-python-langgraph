# File Organizer Agent (Python · LangGraph)

A tool-calling agent that sorts the files of a folder into category sub-directories
(finance, health, travel, …) based on each file's **content**.
The model works only through tools (`list_files`, `read_file`, `create_directory`, `move_file`),
and **every change to the filesystem is approved by a human first**.

## Quick start

```bash
uv sync --extra anthropic          # or: --extra openai
# put your API key (and optionally the model) in .env, see .env.example
uv run file-organizer ~/Downloads/inbox
uv run file-organizer ~/Downloads/inbox --model openai:gpt-4.1   # any init_chat_model id
```

Settings come from `.env` in the working directory (git-ignored; template: `.env.example`).
Variables already set in your shell override `.env`. Precedence for the model:
`--model` > `FILE_ORGANIZER_MODEL` > `anthropic:claude-sonnet-4-5`. The cheap summarizer
model: `--summary-model` > `FILE_ORGANIZER_SUMMARY_MODEL` > `anthropic:claude-haiku-5-5`
(for an Anthropic main model) / `openai:gpt-4.1-mini` (OpenAI) / the main model (others).

```dotenv
FILE_ORGANIZER_MODEL=anthropic:claude-sonnet-4-5
ANTHROPIC_API_KEY=sk-ant-...
```

### Logging

Each run writes a log to `.file-organizer.log` in the working directory. The file name is
hidden, so the agent never lists or moves it. Logging is set up once in `main()` with
`logging.config.dictConfig` (see `log.py`), following the standard-library recommendations:

- **Non-blocking:** the root logger has a `QueueHandler`, and a background `QueueListener`
  thread writes to disk, so logging never slows down the agent or the prompts.
- **Rotating file:** `RotatingFileHandler`, 1 MB × 3 backups, UTF-8. Nothing is printed to
  the terminal.
- **Standard line format:** ISO-8601 UTC timestamps, then level, logger, function, and line:

  ```
  2026-11-17T17:08:53.807Z INFO     file_organizer.workspace:move_file:109 - Moved /inbox/notes.txt -> /inbox/health/notes.txt
  ```
- **Per-module loggers:** each module uses `getLogger(__name__)`, and the package adds a
  `NullHandler`, so using it as a library never configures logging for you. The app logs
  at the chosen level, third-party libraries only from `WARNING` up, and `warnings.warn`
  messages are captured too.
- **Two formats:** `--log-format text` (the default, for reading) or `--log-format json`
  (one JSON object per line, for log tools such as ELK, Datadog, Grafana Loki, or Splunk,
  which then need no parsing rules). In JSON, a traceback goes into its own `exception`
  field instead of spanning several lines, and every line of a run has a `session_id`,
  so you can filter one session out of a shared log file:

  ```json
  {"timestamp": "2026-11-17T17:11:50.645Z", "level": "INFO", "logger": "file_organizer.graph", "function": "run_tools", "line": 113, "message": "User approved: Move 'notes.txt' -> 'health' (feedback: None)", "session_id": "be1578fa-..."}
  ```

The log records:

- the start and end of the session, and aborts
- every summarization (messages summarized/kept, summarizer tokens) and every time the
  trimming safety net fires
- every LLM call (`LLM call #N`): when it starts, how long it took, its input/output/total
  tokens, and the running session totals; failed calls too. In JSON these are also
  separate fields (`llm_call`, `duration_s`, `input_tokens`, `output_tokens`,
  `total_tokens`, `session_total_tokens`, ...)
- the session's total token usage when it ends, aborts, or fails
- every tool the model requests, and the model's final summary
- every approval or rejection, including feedback
- every directory created and file moved, with absolute paths (an audit trail)
- blocked attempts to reach outside the folder, and tool errors
- full tracebacks for failures; the console shows only a short message
- warnings from third-party libraries

```bash
uv run file-organizer ~/inbox --log-level debug --log-file /tmp/organizer.log
uv run file-organizer ~/inbox --log-format json --log-file /var/log/file-organizer.jsonl
```

The options can also be set with `FILE_ORGANIZER_LOG_FILE`, `FILE_ORGANIZER_LOG_LEVEL`, and
`FILE_ORGANIZER_LOG_FORMAT`.
The default level is `INFO`; `DEBUG` adds the sizes of tool results.
File contents are never logged.

```
Organizing /Users/me/Downloads/inbox
  [llm call #1: 1,950 in + 40 out tokens; session total 1,990]
  · list_files()
  [llm call #2: 2,080 in + 102 out tokens; session total 4,172]
  · read_file(path='invoice.txt')
  · read_file(path='flight.txt')

? Create directory 'finance'
  approve? [y]es / [n]o / or type feedback to reject: y

? Move 'flight.txt' -> 'finance'
  approve? [y]es / [n]o / or type feedback to reject: no, that's travel
...
```

After each LLM call the console shows its token counts and the session total; when the
session ends (or is aborted / fails) it prints `Tokens used: … in + … out = … tokens in N
LLM calls`. Token counts come from the provider (LangChain's `usage_metadata`); totals
include earlier runs of a resumed session.

For each proposed change you can answer `y`, answer `n`, or type feedback. Feedback
rejects the change and is passed to the model. Ctrl-C aborts the run, and nothing
else is changed.

### Tracing with Arize Phoenix

The log records what happened. [Arize Phoenix](https://arize.com/docs/phoenix) adds a UI
where you can see **how** it happened. One session is **one trace**, even though the graph
pauses for every approval:

```
file-organizer                                    root span: input folder, output = final message,
├─ organize                                       events = each human decision
│  ├─ summarize, agent (LLM call), review, tools (read_file, ...) ...
├─ resume: approved Create directory 'finance'    one graph run per answer
├─ resume: rejected Move 'a.txt' -> 'finance'
└─ ...
```

Every pause for a human answer ends a `graph.stream()` call and the answer starts a new
one. The instrumentation would report each call as a separate trace, so the CLI wraps the
session in a root span (`tracing.trace_session`) and the graph runs nest under it. The
node spans contain the LLM calls (full prompt, response, tool calls, tokens, latency) and
the tool calls (arguments and results). If the session is aborted, the root span says so;
if it fails, the root span has error status and the exception. Tracing uses OpenTelemetry
through the OpenInference LangChain instrumentor, which also covers LangGraph. It is
optional and off by default.

Phoenix runs locally in Docker (`docker/phoenix/compose.yaml`):

```bash
docker compose -f docker/phoenix/compose.yaml up -d   # Phoenix 22, traces kept in a Docker volume
uv sync --extra anthropic --extra phoenix
uv run file-organizer demo/inbox --phoenix
open http://localhost:6006            # project "file-organizer"
docker compose -f docker/phoenix/compose.yaml down    # stop (add -v to delete all traces)
```

- **Sessions:** every span of a folder's session has the session id
  `organize:<absolute folder path>`. A session resumed in a later process (after Ctrl-C
  or a crash) is a new trace (its first graph run is `resume after restart`), and both
  traces appear together in Phoenix's *Sessions* tab. The metadata `run_id` matches the `session_id` in the log file.
- **Settings:** `--phoenix` / `--no-phoenix` (`FILE_ORGANIZER_PHOENIX=true`),
  `--phoenix-endpoint` (`PHOENIX_COLLECTOR_ENDPOINT`, default `http://localhost:6006`), and
  `--phoenix-project` (`PHOENIX_PROJECT_NAME`, default `file-organizer`).
- **Never in the way:** spans are exported in a background thread. If Phoenix is down,
  spans are dropped and the session goes on. Remaining spans are flushed at exit.
- **Privacy:** unlike the log, traces contain prompts and therefore **file contents**.
  The container publishes its ports on `127.0.0.1` only, and Phoenix's own telemetry is
  off. To keep content out of the traces, set `OPENINFERENCE_HIDE_INPUTS`,
  `OPENINFERENCE_HIDE_OUTPUTS`, `OPENINFERENCE_HIDE_INPUT_MESSAGES`, and
  `OPENINFERENCE_HIDE_OUTPUT_MESSAGES` to `true` (see `.env.example`).
- **Auth:** Phoenix 22 turns on login by default and rejects traces without an API key
  (HTTP 401). The local container sets `PHOENIX_ENABLE_AUTH=false`. If you turn auth on
  (see the comments in `docker/phoenix/compose.yaml`), put `PHOENIX_API_KEY=<key>` in `.env`.

### Resuming an interrupted session

LangGraph saves the agent's state to a SQLite file (`.file-organizer.sqlite` in the
working directory) after every step. If a run stops early because of Ctrl-C, a crash, a
lost connection, or a model/API error, run the same command again:

```
Organizing /Users/me/Downloads/inbox
Resume the interrupted session for this folder? [Y/n] y

? Create directory 'travel'        <- the question you hadn't answered yet
  approve? [y]es / [n]o / or type feedback to reject:
```

- Changes approved before the interruption are not asked about or applied again. The
  pending question is asked again, and a failed model call is retried.
- Each folder has its own saved session, keyed by its absolute path. Answering `n` to
  the resume question starts over.
- The saved state is deleted when a session finishes, so the file stays small. The log
  keeps the history.
- Set the file with `--state-db PATH` or `FILE_ORGANIZER_STATE_DB`.
- If a crash happens in the middle of applying an approved batch, resuming re-runs that
  batch. Steps already done then return harmless errors to the model, such as "already
  exists" or "not found"; nothing is applied twice.

### Memory and context management

Long sessions (many files, big tool results) would otherwise resend an ever-growing
history on every LLM call. Three layers keep the context small (`memory.py`):

1. **Summarization with a cheap model** (persistent). A `summarize` node runs before every
   agent call. Once the history reaches `--summarize-at` tokens (default 60,000, approximate
   count), the cheap model condenses the older messages into short notes (files grouped by
   category, changes made/rejected, next steps; about 300 words at most), and those
   messages are **deleted from the state** (`RemoveMessage`), so later checkpoints and
   requests stay small. The newest half of the threshold (in tokens) stays verbatim, so
   the history drops well below the threshold and the next summary needs real growth. A
   tool call is never separated from its result. A summary is **skipped** when it would
   condense less than a quarter of the threshold of new content, e.g. while one big batch
   of file reads fills the whole kept window; otherwise it would re-summarize the previous
   summary on every step. This reuses LangChain's
   [`SummarizationMiddleware`](https://docs.langchain.com/oss/python/langchain/middleware/built-in#summarization)
   (`model=` main model, `summarizer=` cheap model): middleware hooks are plain
   `state -> update` functions, so its `before_model` hook runs as a node of this
   hand-written graph. Each summary is shown with its cost and its effect, and logged as
   `Summarizer call #N`:

   ```
     [summarizer call #1: 3,478 in + 995 out tokens; summarizer total 4,473]
     [history ~3,645 -> ~1,855 tokens: 15 older messages summarized, 4 kept]
   ```
2. **Trimming at the LLM call** (transient safety net). If a request would exceed
   `--max-context-tokens` (default 150,000), `fit_context()` first shortens old tool
   results to a 300-character preview (a file's path and first lines), then, only if
   needed, drops the oldest turns and adds a note to the system prompt saying so. The
   newest turn (the results the model is waiting for) is always sent in full, and the
   state is never changed; a warning is logged.

The limits are checked to fit together (`ContextPolicy`): the cap must be at least twice
the summarize threshold, so trimming can't fire constantly and drop messages that no
summary has captured yet (with tiny limits, the agent used to forget the files it had
just read and re-read them). Both layers count tokens the same way (about 3.3 characters
per token for Claude, 4 for others).
3. **Finalized decisions in a store** (long-term memory). Pending answers live in the
   graph state only until their batch has run. Then each decision (approved/rejected,
   feedback, outcome) moves to a LangGraph [store](https://docs.langchain.com/oss/python/langgraph/add-memory#add-long-term-memory)
   (`SqliteStore`, namespace per folder), and the decision log is added to the system
   prompt on every call. Summaries are lossy, but "the human rejected X" is never lost.

Retention, following LangGraph's [TTL settings](https://docs.langchain.com/langsmith/configure-ttl):

- **Checkpoints: `keep_latest`.** Whenever the run pauses for an answer, old checkpoints of
  the thread are pruned; only the newest (plus its pending writes, e.g. answers already
  given in a batch) is needed to resume. A finished session is deleted (`delete`).
  `SqliteSaver` doesn't implement `BaseCheckpointSaver.prune()` yet, so
  `persistence.PruningSqliteSaver` adds it.
- **Store items: TTL.** Decisions expire 7 days after they were last written
  (`TTLConfig(default_ttl=10080, refresh_on_read=True)`; the agent's own reads on every
  LLM call pass `refresh_ttl=False` to avoid a database write each time). Expired items
  are swept when the store is opened. Starting a session over clears that folder's decisions right away.

```bash
# watch summarization happen on the demo (tiny limits)
uv run file-organizer demo/inbox --summarize-at 3000 --max-context-tokens 6000
```

## Demo with real data

```bash
uv run python demo/demo.py reset      # copy demo/sample -> demo/inbox (git-ignored scratch)
uv run file-organizer demo/inbox
uv run python demo/demo.py check      # compare the result with demo/expected.json
```

The 14 sample files include tricky cases: misleading names, a binary file, a long
(truncated) file, an existing `finance/` folder, a hidden file, and an ambiguous file.
See [`demo/README.md`](demo/README.md).

## Architecture

```
START → summarize → agent ──(no tool calls)──▶ END
            ▲          │
            │          └─(tool calls)─▶ review ─▶ tools ─┐
            └────────────────────────────────────────────┘
```

| Module | Responsibility |
|---|---|
| `workspace.py` | Filesystem operations sandboxed to the target folder. It rejects `..`, absolute paths, and symlink escapes. It moves files only into existing directories, so it never renames or overwrites them. It truncates long files and doesn't dump binary files. |
| `tools.py` | Wraps the workspace as LangChain tools; each function's docstring becomes the tool description. `WorkspaceError` is a `ToolException`, so errors are sent back to the model as `Error: …` tool results. `MUTATING_TOOLS = {create_directory, move_file}`. |
| `graph.py` | The LangGraph `StateGraph` (state schema, nodes, wiring), compiled with a checkpointer (required) and a store. `summarize` shrinks the history when needed. `agent` calls the LLM with the decision log in the system prompt and a trimmed request. `review` calls `interrupt()` once for each mutating tool call and does nothing else, so LangGraph can safely re-run it on resume. `tools` runs read-only calls and approved changes; for rejected changes it returns a `Rejected by the user …` message, including any feedback. It then writes the final decisions to the store. |
| `memory.py` | Context management: `ContextPolicy` (limits), the summarize node (wrapping `SummarizationMiddleware`), `fit_context()` (trimming safety net), and `DecisionLog` (finalized decisions in the store). |
| `persistence.py` | `sqlite_checkpointer()` (`PruningSqliteSaver`: `SqliteSaver` + `prune(keep_latest/delete)`) and `sqlite_store()` (`SqliteStore` with a TTL), both in one SQLite file. |
| `approval.py` | The `Decision` type, parsing of resume values, and escaped descriptions of proposed changes. |
| `usage.py` | `TokenUsage` totals, the `add_usage` state reducer, and `collect_usage()` (counts LLM calls made inside library code via a callback). |
| `cli.py` | Streams progress, turns each interrupt into a y/n/feedback prompt, and resumes the graph with `Command(resume={interrupt_id: decision})`. It runs the graph with `durability="sync"` on the SQLite checkpointer and offers to resume an interrupted session. |
| `prompts.py` | System prompt, decision-log section, and the summarization prompt. |
| `tracing.py` | Optional Arize Phoenix tracing: `configure_tracing()` registers an OTLP exporter (`phoenix.otel.register`) and the OpenInference `LangChainInstrumentor`, and returns a `stop()` that flushes spans. `trace_session()` opens one root span per CLI session, so the start and every resume after an approval form a single trace. |
| `log.py` | Logging configuration (`dictConfig`): queue handler, background thread, rotating file, UTC timestamps, text or JSON-lines format, `session_id` on every record. `configure_logging()` returns a `stop()` function that flushes the log. |

Safety properties (all covered by tests):
- Nothing is written to disk before approval. Only an explicit `True` or `{"approved": True}` approves a change; any other resume value counts as a rejection.
- Every change in a batch of parallel tool calls is approved separately, and approved changes run only once.
- The agent can't touch anything outside the target folder, nor hidden files (`.env`, `.git/`, its own state/log files) or symlinks inside it.
- Bad paths (null bytes, overlong) and malformed tool arguments are returned to the model as errors instead of crashing the run.
- Reads and listings are bounded (only a prefix of a file is read; listings are capped), and `create_directory` nests at most 3 levels deep.
- Model-supplied text shown in the approval prompt has control characters escaped, so newlines/ANSI escapes can't fake or hide part of the question.
- File contents are wrapped in `<file>` tags, and the system prompt tells the model to treat them as untrusted data, not instructions (a prompt-injection mitigation; the approval gate is still the real safeguard).
- Files are never renamed or overwritten: `move_file` only moves into an existing directory, even if a bad move is approved.
- Summaries and trimming never drop the human's decisions: those come from the store, and the system prompt says they win over the summary. The summary prompt tells the cheap model to record only categories, never instructions found in files.
- After a restart, answers are matched to their question by interrupt id, so an old "yes" can never approve a different change. A plain `Command(resume=value)` followed by `stream(None)` after a restart *would* reuse it; `test_resume.py` guards against that.

Library use:

```python
from langgraph.types import Command
from file_organizer import ContextPolicy, Workspace, build_graph, sqlite_checkpointer, sqlite_store

with sqlite_checkpointer("state.sqlite") as checkpointer, sqlite_store("state.sqlite") as store:
    graph = build_graph(model, Workspace("~/inbox"), checkpointer=checkpointer, store=store,
                        summary_model=cheap_model, policy=ContextPolicy(summarize_at_tokens=60_000))
    cfg = {"configurable": {"thread_id": "1"}}
    result = graph.invoke({"messages": [("user", "Organize this folder")]}, cfg)
    while result.get("__interrupt__"):
        pending = result["__interrupt__"][0]     # .value = {tool, args, description, tool_call_id}
        result = graph.invoke(Command(resume={pending.id: {"approved": True}}), cfg)
```

`checkpointer` is required: there is no in-memory fallback, so state is always persisted.
`store` defaults to an `InMemoryStore` (no TTL; lost on exit), `summary_model` to `model`.


## Development (TDD)

```bash
uv run pytest
```

The tests use a scripted fake chat model (`tests/conftest.py`), so they're deterministic
and don't call any LLM. They cover:
- `test_workspace.py`: sandbox and filesystem semantics
- `test_tools.py`: the tool surface and error handling
- `test_graph.py`: the agent loop, interrupts, approval and rejection, batches, a full session, the required checkpointer, and reopening the SQLite file. Graph tests use the real `SqliteSaver` on an in-memory SQLite database.
- `test_cli.py`: prompt parsing, a full interactive session, Ctrl-C, and the entry point
- `test_memory.py`: summarization with the cheap model (state shrinks, histories stay valid, token totals survive), the limit checks, skipping pointless summaries, the trimming safety net (previews first, then dropping with a note), a regression test that tiny limits never make the agent lose what it read, decisions moving to the store and reaching the prompt even after summarization, `keep_latest` pruning of a half-answered batch, store TTL sweeping, and the CLI wiring
- `test_resume.py`: resuming in a later run after a crash or Ctrl-C, no stale answers, declining to resume, cleanup after a finished session, one session per folder
- `test_log.py`: the background log file, text and JSON formats, tracebacks, `session_id` tagging, cleanup, the audit entries, and each decision logged once
- `test_tracing.py`: the Phoenix options (off by default, flag, env vars, missing extra, a failing flush), and that a session with approvals is exactly one trace: one root span, one graph run per answer, the decisions as events, and aborted/failed sessions marked (uses an in-memory span exporter, no Phoenix needed)
- `test_demo.py`: keeps the demo sample, `expected.json`, and `demo.py` consistent
