"""PreCompact hook — injects CLAUDE.md + git state as a systemMessage.

Fail-open on any error.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from autosre.claude_hooks import _io

_MAX_CLAUDE_MD = 1500


def _read_claude_md(cwd: Path) -> str:
    path = cwd / "CLAUDE.md"
    try:
        content = path.read_text()[:_MAX_CLAUDE_MD]
    except (OSError, UnicodeDecodeError):
        return ""
    if len(content) >= _MAX_CLAUDE_MD:
        content += "\n... (truncated)"
    return content


def _git_work_state(cwd: Path) -> str:
    parts: list[str] = []

    try:
        branch = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if branch.returncode == 0 and branch.stdout.strip():
            parts.append(f"Branch: {branch.stdout.strip()}")
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass

    try:
        status = subprocess.run(
            ["git", "status", "--short"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if status.returncode == 0 and status.stdout.strip():
            lines = status.stdout.strip().splitlines()
            if len(lines) > 15:
                summary = "\n".join(lines[:15]) + f"\n... ({len(lines) - 15} more)"
            else:
                summary = status.stdout.strip()
            parts.append(f"Uncommitted:\n{summary}")
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass

    return "\n".join(parts) if parts else ""


def main() -> int:
    inv = _io.parse_stdin(__file__)

    try:
        cwd = inv.cwd if inv.cwd.is_dir() else Path.cwd()
        claude_md = _read_claude_md(cwd)
        work_state = _git_work_state(cwd)

        if not claude_md and not work_state:
            return _io.emit_ok()

        parts = ["PRESERVE ACROSS COMPACTION"]
        if claude_md:
            parts.append(f"## Project Config (CLAUDE.md)\n{claude_md}")
        if work_state:
            parts.append(f"## Current Work State\n{work_state}")

        return _io.emit_system_message("\n\n---\n\n".join(parts))
    except Exception as exc:
        return _io.fail_open_continue(inv, f"precompact error: {exc}")


if __name__ == "__main__":
    sys.exit(main())
