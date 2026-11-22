"""Human approval: what a decision looks like and how proposed changes are shown."""

from __future__ import annotations

from typing import Any, NotRequired, TypedDict


class Decision(TypedDict):
    """The human's answer to one proposed change (the resume value of one interrupt)."""

    approved: bool
    feedback: str | None
    # move_file only: the directories the move leaves empty, shown to the human with
    # the move; approving the move approves removing exactly these (deepest first).
    remove_dirs: NotRequired[list[str]]


def _safe(value: Any, limit: int = 200) -> str:
    """Render a model-supplied value for the terminal: control characters (newlines, ANSI
    escapes, ...) are escaped so they can't fake or hide parts of the approval prompt."""
    text = "".join(ch if ch.isprintable() else ch.encode("unicode_escape").decode() for ch in str(value))
    return text if len(text) <= limit else text[:limit] + "..."


def describe(tool_name: str, args: dict[str, Any]) -> str:
    """Human-readable summary of a proposed filesystem change."""
    if not isinstance(args, dict):
        args = {}
    if tool_name == "create_directory":
        return f"Create directory '{_safe(args.get('path'))}'"
    if tool_name == "move_file":
        return f"Move '{_safe(args.get('source'))}' -> '{_safe(args.get('destination'))}'"
    return _safe(f"{tool_name}({args})")


def parse_decision(value: Any) -> Decision:
    """Normalize a resume value. Only True or {"approved": True} approve; all else rejects."""
    if isinstance(value, dict):
        return {"approved": value.get("approved") is True, "feedback": value.get("feedback") or None}
    return {"approved": value is True, "feedback": None}
