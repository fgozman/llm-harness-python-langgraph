"""Undo: reverse the changes a session applied, using the decision log in the store.

Every applied change is recorded with an ``undo`` entry (see ``workspace.Change``):
a move remembers where the file came from, a new directory which directories it
created. Undoing walks those records newest first:

* a moved file goes back to its original path (never overwriting anything there);
* a created directory is removed only if it is empty again (anything put there since,
  even a hidden file, keeps it).

A step that can't be undone (the file was moved or deleted since, the original name is
taken again, ...) is reported and its record kept, so a later undo can try again once
the conflict is fixed. Records of undone changes are deleted.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .approval import _safe
from .memory import DecisionLog, DecisionRecord
from .workspace import Workspace, WorkspaceError

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class UndoStep:
    """One reversal: ``key`` is the decision's tool_call_id in the store."""

    key: str
    description: str
    op: str  # "move" or "rmdir"
    paths: tuple[str, ...]  # move: (current path, original path); rmdir: (directory,)


def plan_undo(entries: list[tuple[str, DecisionRecord]]) -> tuple[list[UndoStep], list[str]]:
    """Turn the decision log (oldest first) into undo steps (newest first).

    Returns (steps, keys of applied records that can't be undone because they have no
    undo information, e.g. written by an older version). A created directory gives one
    step per directory, deepest first.
    """
    steps: list[UndoStep] = []
    unknown: list[str] = []
    for key, record in reversed(entries):
        if record.get("outcome") != "applied":
            continue
        undo = record.get("undo")
        if record.get("tool") == "create_directory" and undo is None and "undo" in record:
            continue  # the directory already existed: nothing changed on disk
        if not isinstance(undo, dict):
            unknown.append(key)
        elif undo.get("op") == "move":
            steps.append(UndoStep(key, f"Move '{_safe(undo['to'])}' back to '{_safe(undo['from'])}'",
                                  "move", (undo["to"], undo["from"])))
        elif undo.get("op") == "mkdir":
            for path in reversed(undo["paths"]):
                steps.append(UndoStep(key, f"Remove directory '{_safe(path)}' (if empty)", "rmdir", (path,)))
        else:
            unknown.append(key)
    return steps, unknown


def apply_step(workspace: Workspace, step: UndoStep) -> str:
    """Apply one undo step; raises WorkspaceError if it can't be done safely."""
    if step.op == "move":
        return workspace.undo_move(*step.paths)
    return workspace.remove_empty_directory(step.paths[0])


@dataclass
class UndoResult:
    done: list[UndoStep]
    failed: list[tuple[UndoStep, str]]


def undo_all(workspace: Workspace, decisions: DecisionLog, steps: list[UndoStep]) -> UndoResult:
    """Apply ``steps`` in order. A record is deleted as soon as its last step succeeded
    (and none of its steps failed), so an undo stopped half-way (Ctrl-C, an unexpected
    OSError, which propagates) leaves only the records of what was not undone yet.

    Only ``WorkspaceError`` (a conflict on disk the user can fix) is caught: it is
    reported in the result and the record kept for a retry."""
    result = UndoResult([], [])
    failed_keys: set[str] = set()
    pending: dict[str, int] = {}
    for step in steps:
        pending[step.key] = pending.get(step.key, 0) + 1
    for step in steps:
        pending[step.key] -= 1
        try:
            apply_step(workspace, step)
        except WorkspaceError as exc:
            log.warning("Undo failed: %s: %s", step.description, exc)
            result.failed.append((step, str(exc)))
            failed_keys.add(step.key)
        else:
            result.done.append(step)
        if pending[step.key] == 0 and step.key not in failed_keys:
            decisions.delete(step.key)
    return result
