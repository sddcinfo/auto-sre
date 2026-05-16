"""PostToolUse audit hook — logs executed Bash commands.

Writes to ``autosre.paths.hook_audit_log()`` (XDG). Always fails open —
the audit log is a convenience, not a correctness surface.
"""

from __future__ import annotations

import os
import sys
from datetime import UTC, datetime

from autosre import paths
from autosre.claude_hooks import _io


def main() -> int:
    inv = _io.parse_stdin(__file__)
    command = inv.tool_input.get("command", "") if inv.tool_input else ""

    if not command:
        return _io.emit_continue()

    summary = command[:200].replace("\n", " ").strip()
    if len(command) > 200:
        summary += "..."

    session_id = (
        inv.session_id
        if inv.session_id != "unknown"
        else os.environ.get("CLAUDE_SESSION_ID") or os.environ.get("AUTOSRE_RUN_ID", "unknown")
    )
    timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    log_path = paths.hook_audit_log()
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a") as f:
            f.write(f"{timestamp} | {session_id} | {summary}\n")
    except OSError as exc:
        _io.log(inv, f"audit log write failed: {exc}", level="warn")

    return _io.emit_continue()


if __name__ == "__main__":
    sys.exit(main())
