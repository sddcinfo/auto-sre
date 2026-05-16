"""Post-commit scan-update hook — **stub**.

No-op returning ``{"result": "continue"}``. Routed through :mod:`._io`
so its input still gets archived for drift analysis. Kept as a named
hook so the hooks installer has a stable target if a real post-commit
scan subsystem lands later.
"""

from __future__ import annotations

import sys

from autosre.claude_hooks import _io


def main() -> int:
    _io.parse_stdin(__file__)
    return _io.emit_continue()


if __name__ == "__main__":
    sys.exit(main())
