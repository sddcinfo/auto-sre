"""UserPromptSubmit hook — emits "on branch X" at most once per hour.

Marker lives under :func:`autosre.paths.branch_warned_marker` (XDG
state). Fail-open on any error.
"""

from __future__ import annotations

import subprocess
import sys
import time
from typing import TYPE_CHECKING

from autosre import paths
from autosre.claude_hooks import _io

if TYPE_CHECKING:
    from pathlib import Path

_ONE_HOUR_SECONDS = 3600


def _get_current_branch() -> str | None:
    try:
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return None
    if result.returncode == 0:
        return result.stdout.strip() or None
    return None


def _touch(marker: Path) -> None:
    try:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.touch()
    except OSError:
        pass


def main() -> int:
    inv = _io.parse_stdin(__file__)

    try:
        marker = paths.branch_warned_marker()
        if marker.exists():
            try:
                if (time.time() - marker.stat().st_mtime) < _ONE_HOUR_SECONDS:
                    return _io.emit_continue()
            except OSError:
                pass

        branch = _get_current_branch()
        if branch and branch not in ("dev", "main"):
            _touch(marker)
            return _io.emit_continue(message=f"On feature branch: {branch}")
        if branch == "main":
            _touch(marker)
            return _io.emit_continue(
                message=(
                    "WARNING: On 'main' branch. Consider a feature branch for "
                    "work that might land in a PR."
                ),
            )
        return _io.emit_continue()
    except Exception as exc:
        return _io.fail_open_continue(inv, f"branch-check error: {exc}")


if __name__ == "__main__":
    sys.exit(main())
