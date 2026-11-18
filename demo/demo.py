"""Demo helper: prepare a scratch inbox from the sample files and check the result.

    uv run python demo/demo.py reset      # copy demo/sample -> demo/inbox (fresh)
    uv run file-organizer demo/inbox      # let the agent organize it
    uv run python demo/demo.py check      # compare demo/inbox with demo/expected.json

`demo/sample` is never modified; `demo/inbox` is git-ignored scratch space.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

DEMO = Path(__file__).resolve().parent
SAMPLE = DEMO / "sample"
INBOX = DEMO / "inbox"
EXPECTED = DEMO / "expected.json"


def load_expected() -> dict[str, list[str | None]]:
    raw = json.loads(EXPECTED.read_text())
    raw.pop("_comment", None)
    return {name: v if isinstance(v, list) else [v] for name, v in raw.items()}


def reset() -> int:
    if INBOX.exists():
        shutil.rmtree(INBOX)
    shutil.copytree(SAMPLE, INBOX)
    count = sum(1 for p in INBOX.rglob("*") if p.is_file() and not p.name.startswith("."))
    print(f"Reset {INBOX} with {count} sample files.")
    print(f"Next: uv run file-organizer {os.path.relpath(INBOX)}")
    return 0


def check() -> int:
    if not INBOX.is_dir():
        print("No demo/inbox yet - run: uv run python demo/demo.py reset")
        return 2

    # Where each (basename) ended up: None = top level, else its directory.
    found: dict[str, str | None] = {}
    for p in INBOX.rglob("*"):
        if p.is_file() and not p.name.startswith("."):
            parent = p.parent.relative_to(INBOX).as_posix()
            found[p.name] = None if parent == "." else parent

    ok = bad = 0
    for name, allowed in load_expected().items():
        base = Path(name).name
        if base not in found:
            print(f"  MISSING  {name}")
            bad += 1
            continue
        actual = found.pop(base)
        where = actual or "(top level)"
        shown = ", ".join(a or "(top level)" for a in allowed)
        if actual in allowed:
            print(f"  ok       {name:28} -> {where}")
            ok += 1
        else:
            print(f"  WRONG    {name:28} -> {where}   (expected: {shown})")
            bad += 1
    for extra, where in found.items():
        print(f"  EXTRA    {extra:28} -> {where or '(top level)'}")
        bad += 1

    print(f"\n{ok} ok, {bad} problem(s)")
    return 0 if bad == 0 else 1


def main(argv: list[str]) -> int:
    commands = {"reset": reset, "check": check}
    if len(argv) != 1 or argv[0] not in commands:
        print(__doc__)
        return 2
    return commands[argv[0]]()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
