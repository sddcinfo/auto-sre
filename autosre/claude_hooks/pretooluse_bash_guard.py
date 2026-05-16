"""PreToolUse bash guard — delegates to ``autosre hooks-backend guard``.

Security-critical: fails CLOSED on any error (timeout, missing binary,
empty subprocess stdout) so a broken guard can't bypass the allow/deny
check.

The backend parses the hook JSON itself, so this shim reads stdin once,
archives/logs it via :mod:`._io`, then forwards the bytes to the
subprocess.
"""

from __future__ import annotations

import io
import subprocess
import sys

from autosre.claude_hooks import _io


def main() -> int:
    try:
        payload = sys.stdin.read()
    except OSError:
        payload = ""

    # Re-parse via _io so logging + archival are consistent with the other hooks.
    real_stdin = sys.stdin
    sys.stdin = io.StringIO(payload)
    try:
        inv = _io.parse_stdin(__file__)
    finally:
        sys.stdin = real_stdin

    try:
        result = subprocess.run(
            [sys.executable, "-m", "autosre.cli", "hooks-backend", "guard"],
            input=payload,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return _io.fail_closed_pretooluse(inv, "Guard timed out (fail-closed).")
    except (FileNotFoundError, OSError) as exc:
        return _io.fail_closed_pretooluse(inv, f"Guard binary not available (fail-closed): {exc}")
    except Exception as exc:
        return _io.fail_closed_pretooluse(inv, f"Guard error (fail-closed): {exc}")

    if not result.stdout.strip():
        return _io.fail_closed_pretooluse(inv, "Guard returned no output (fail-closed).")

    return _io.emit_raw_stdout(result.stdout, exit_code=result.returncode)


if __name__ == "__main__":
    sys.exit(main())
