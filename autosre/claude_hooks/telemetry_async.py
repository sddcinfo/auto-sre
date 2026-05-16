"""Telemetry hook — **stub**.

autosre ships no telemetry sink, so this hook is a no-op that returns
``{"result": "continue"}``. Routed through :mod:`._io` so its input
still gets archived for drift analysis.
"""

from __future__ import annotations

import sys

from autosre.claude_hooks import _io


def main() -> int:
    _io.parse_stdin(__file__)
    return _io.emit_continue()


if __name__ == "__main__":
    sys.exit(main())
