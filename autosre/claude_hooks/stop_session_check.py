"""Stop hook — session-end completion checklist.

Thin wrapper around ``autosre hooks-backend stop-check``. Fail-open:
any error produces a ``continue`` so a broken backend doesn't trap the
user at the end of their session.
"""

from __future__ import annotations

import subprocess
import sys

from autosre.claude_hooks import _io


def main() -> int:
    inv = _io.parse_stdin(__file__)

    try:
        result = subprocess.run(
            [sys.executable, "-m", "autosre.cli", "hooks-backend", "stop-check"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return _io.emit_continue(message="[session-end] Stop check timed out.")
    except (FileNotFoundError, OSError) as exc:
        return _io.fail_open_continue(inv, f"stop-check binary not available: {exc}")
    except Exception as exc:
        return _io.fail_open_continue(inv, f"stop-check error: {exc}")

    if result.stdout.strip():
        return _io.emit_raw_stdout(result.stdout, exit_code=0)
    return _io.emit_continue()


if __name__ == "__main__":
    sys.exit(main())
