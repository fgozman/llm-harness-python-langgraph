"""Where the agent's memory lives on disk: one SQLite file, two LangGraph components.

* Checkpointer (short-term memory, per thread): the graph state after every step, so
  a paused or crashed run can resume. Old checkpoints are pruned with the
  ``"keep_latest"`` strategy: resuming only ever needs the newest one.
* Store (long-term memory, across threads): finalized human decisions, as JSON
  documents under a namespace. Items expire after a TTL (time to live).

LangGraph defines both retention policies (``BaseCheckpointSaver.prune(thread_ids,
strategy="keep_latest" | "delete")`` and the store's ``TTLConfig``); LangSmith
deployments configure them in ``langgraph.json`` (``checkpointer.ttl``/``store.ttl``).
Here we run them ourselves: ``SqliteStore`` implements TTLs natively, while
``SqliteSaver`` does not implement ``prune()`` yet, so ``PruningSqliteSaver`` adds it.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

# SqliteSaver and SqliteStore come from a separate package: langgraph-checkpoint-sqlite
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.base import TTLConfig
from langgraph.store.sqlite import SqliteStore

log = logging.getLogger(__name__)

DEFAULT_STATE_DB = ".file-organizer.sqlite"  # hidden, so the agent never lists or moves it

#: Store items (finalized decisions) expire 7 days after they were last written or read.
STORE_TTL: TTLConfig = {"default_ttl": 7 * 24 * 60, "refresh_on_read": True}


class PruningSqliteSaver(SqliteSaver):
    """``SqliteSaver`` plus the ``prune()`` method of LangGraph's checkpointer interface.

    The SQLite saver stores every checkpoint as a full snapshot of the state (no delta
    channels are used by this graph), so the newest checkpoint of a thread, together with
    its pending writes (e.g. answers to interrupts), is all a resume needs.
    """

    def prune(self, thread_ids: Sequence[str], *, strategy: str = "keep_latest") -> None:
        """Prune checkpoints of ``thread_ids``.

        ``"keep_latest"`` keeps only the most recent checkpoint per namespace (the one
        ``get_tuple`` loads) and its writes; ``"delete"`` removes the threads entirely.
        """
        if strategy == "delete":
            for thread_id in thread_ids:
                self.delete_thread(thread_id)
            return
        if strategy != "keep_latest":
            raise ValueError(f"unknown prune strategy {strategy!r}")
        # "Latest" = the highest checkpoint_id, exactly how get_tuple() picks it.
        latest = (
            "(SELECT MAX(c.checkpoint_id) FROM checkpoints c "
            "WHERE c.thread_id = {t}.thread_id AND c.checkpoint_ns = {t}.checkpoint_ns)"
        )
        with self.cursor() as cur:  # one transaction, under the saver's lock
            for thread_id in thread_ids:
                cur.execute(
                    f"DELETE FROM writes WHERE thread_id = ? AND checkpoint_id < {latest.format(t='writes')}",
                    (str(thread_id),),
                )
                writes = cur.rowcount
                cur.execute(
                    "DELETE FROM checkpoints WHERE thread_id = ? AND checkpoint_id < "
                    f"{latest.format(t='checkpoints')}",
                    (str(thread_id),),
                )
                log.debug("Pruned thread %s (keep_latest): %d checkpoints, %d writes removed",
                          thread_id, cur.rowcount, writes)


def _db_path(path: str | Path) -> str:
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


@contextmanager
def sqlite_checkpointer(path: str | Path = DEFAULT_STATE_DB) -> Iterator[PruningSqliteSaver]:
    """Open LangGraph's SQLite checkpointer on ``path`` (created if missing).

    The graph state is saved there after every step, per thread_id, so a stopped run
    can be resumed later. The connection is closed when the ``with`` block ends.
    """
    # from_conn_string is a context manager that opens (and later closes) the SQLite
    # connection. Other backends share the interface: InMemorySaver (tests, nothing
    # persisted), PostgresSaver, ... The graph code doesn't care which one.
    with PruningSqliteSaver.from_conn_string(_db_path(path)) as checkpointer:
        yield checkpointer


@contextmanager
def sqlite_store(path: str | Path = DEFAULT_STATE_DB, *, ttl: TTLConfig | None = STORE_TTL) -> Iterator[SqliteStore]:
    """Open LangGraph's SQLite key-value store on ``path`` (its own tables, so it can share
    the checkpointer's file). Expired items are swept once on open; a long-running
    server would call ``store.start_ttl_sweeper()`` instead."""
    with SqliteStore.from_conn_string(_db_path(path), ttl=ttl) as store:
        store.setup()  # create/migrate the tables
        if ttl and (expired := store.sweep_ttl()):
            log.info("Removed %d expired store items", expired)
        yield store
