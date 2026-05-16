"""Tmux split-pane helpers.

Used by the TUI to "self-split" its pane once so the top stays live with
compact metrics and the bottom runs a spawned child (Claude Code, swarm,
eval run). Outside tmux every helper degrades to a no-op so the rest of
the codebase never needs to branch on ``in_tmux()``.

The TUI is the supervisor. Service-start and other one-shot commands
never call anything in this module. Two safeguards prevent layout
surprises:

1. The self-split is gated on a tmux window option
   ``@autosre_split_done`` so restarting the TUI in the same window does
   not stack splits.
2. ``AUTOSRE_NO_SPLIT=1`` in the environment disables all splitting,
   which is the escape hatch for anyone who wants to run the TUI
   full-screen in a single pane.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from dataclasses import dataclass

__all__ = [
    "SplitResult",
    "bottom_pane_from_flag",
    "focus_pane",
    "in_tmux",
    "is_splitting_disabled",
    "pane_height",
    "pane_id",
    "pane_width",
    "run_in_pane",
    "self_split_once",
    "set_pane_title",
    "wait_for_pane_exit",
    "window_size",
]


def in_tmux() -> bool:
    """True iff we are running inside a tmux session."""
    return bool(os.environ.get("TMUX"))


def is_splitting_disabled() -> bool:
    """Honor the ``AUTOSRE_NO_SPLIT`` env var."""
    return os.environ.get("AUTOSRE_NO_SPLIT", "").strip() not in ("", "0")


def _tmux() -> str | None:
    return shutil.which("tmux")


def _run_tmux(*args: str) -> subprocess.CompletedProcess[str]:
    tmux = _tmux()
    if tmux is None:
        raise RuntimeError("tmux not on PATH")
    return subprocess.run(
        [tmux, *args],
        check=False,
        capture_output=True,
        text=True,
    )


def pane_id() -> str | None:
    """Return the current pane id, e.g. ``%42``, or ``None`` if unavailable."""
    if not in_tmux() or _tmux() is None:
        return None
    r = _run_tmux("display-message", "-p", "#{pane_id}")
    if r.returncode != 0:
        return None
    out = r.stdout.strip()
    return out or None


def pane_height(pane: str | None = None) -> int | None:
    """Return the height in rows of the named pane (default: current)."""
    if not in_tmux() or _tmux() is None:
        return None
    target = ["-t", pane] if pane else []
    r = _run_tmux("display-message", "-p", *target, "#{pane_height}")
    if r.returncode != 0:
        return None
    try:
        return int(r.stdout.strip())
    except ValueError:
        return None


def pane_width(pane: str | None = None) -> int | None:
    """Return the width in columns of the named pane (default: current)."""
    if not in_tmux() or _tmux() is None:
        return None
    target = ["-t", pane] if pane else []
    r = _run_tmux("display-message", "-p", *target, "#{pane_width}")
    if r.returncode != 0:
        return None
    try:
        return int(r.stdout.strip())
    except ValueError:
        return None


def window_size() -> tuple[int, int] | None:
    """Return the (cols, rows) of the current tmux window, or ``None``."""
    if not in_tmux() or _tmux() is None:
        return None
    r = _run_tmux("display-message", "-p", "#{window_width} #{window_height}")
    if r.returncode != 0:
        return None
    try:
        cols, rows = r.stdout.strip().split()
        return int(cols), int(rows)
    except (ValueError, IndexError):
        return None


def focus_pane(pane: str) -> bool:
    """Make ``pane`` the active pane so subsequent keystrokes go to it.

    Used on demo dispatch so the user can immediately type into the
    just-respawned demo pane without needing to mouse-click or press a
    prefix-arrow. Returns True on success.
    """
    if not in_tmux() or _tmux() is None:
        return False
    r = _run_tmux("select-pane", "-t", pane)
    return r.returncode == 0


def _window_option_set(name: str) -> bool:
    r = _run_tmux("show-options", "-w", "-v", name)
    return r.returncode == 0 and r.stdout.strip() in ("on", "1", "true")


def _set_window_option(name: str, value: str) -> None:
    _run_tmux("set-option", "-w", name, value)


@dataclass(frozen=True)
class SplitResult:
    """Outcome of :func:`self_split_once`."""

    performed: bool
    reason: str
    top_pane: str | None = None
    bottom_pane: str | None = None
    orientation: str = "horizontal"


_SPLIT_FLAG = "@autosre_split_done"
_BOTTOM_PANE_KEY = "@autosre_bottom_pane"
_ORIENTATION_KEY = "@autosre_split_orientation"


def _get_window_option_value(name: str) -> str | None:
    r = _run_tmux("show-options", "-w", "-v", name)
    if r.returncode != 0:
        return None
    out = r.stdout.strip()
    return out or None


def _pane_exists(pane: str) -> bool:
    r = _run_tmux("list-panes", "-t", pane, "-F", "#{pane_id}")
    return r.returncode == 0 and pane in r.stdout


def bottom_pane_from_flag() -> str | None:
    """Recover the stored bottom-pane id from the tmux window option.

    Returns the pane id if it's still alive, ``None`` otherwise. When the
    stored pane has died (e.g. user closed it manually) both the bottom-
    pane key and the split flag are cleared so the next ``self_split_once``
    call re-splits cleanly instead of assuming an already-split layout.
    """
    if not in_tmux() or _tmux() is None:
        return None
    stored = _get_window_option_value(_BOTTOM_PANE_KEY)
    if stored is None:
        return None
    if _pane_exists(stored):
        return stored
    # Stale — clear both flags so the next run re-splits.
    _run_tmux("set-option", "-w", "-u", _BOTTOM_PANE_KEY)
    _run_tmux("set-option", "-w", "-u", _SPLIT_FLAG)
    return None


def _preflight_split() -> str | None:
    """Return a short reason why we should skip splitting, or ``None``.

    Separated so :func:`self_split_once` can stay under the PLR0911
    return-count budget while still honoring every safety gate.
    """
    if not in_tmux():
        return "not in tmux"
    if is_splitting_disabled():
        return "AUTOSRE_NO_SPLIT set"
    if _tmux() is None:
        return "tmux binary missing"
    if _window_option_set(_SPLIT_FLAG):
        return "already split"
    return None


def self_split_once(
    *,
    orientation: str = "horizontal",
    bottom_pct: int = 60,
    top_rows: int = 28,
    sidebar_cols: int = 60,
    placeholder_cmd: str = "bash -l",
) -> SplitResult:
    """Split the current pane once. Idempotent across reruns of the TUI.

    ``orientation="horizontal"`` creates a top/bottom split with the TUI
    on top and the demo/interactive pane on the bottom (``top_rows`` tall).

    ``orientation="vertical"`` creates a left/right split with the TUI on
    the left (``sidebar_cols`` wide) and the demo on the right. This is
    the preferred layout on wide terminals because Claude Code's
    experimental agent-team mode creates its own tmux sub-panes by
    splitting the active pane; keeping the TUI in a pinned left sidebar
    means those team splits subdivide the right region only.

    On idempotent re-entry the stored orientation is reused regardless
    of the arguments, so passing a different ``orientation`` on a second
    call has no effect unless the previous split was cleared (e.g. the
    stored bottom pane died, triggering :func:`bottom_pane_from_flag`
    to clear both flags).

    The right/bottom pane runs ``placeholder_cmd`` (default: an
    interactive shell) so there is always something alive there. When
    the TUI is ready to spawn a real child it calls :func:`run_in_pane`
    on the stored pane id.
    """
    skip = _preflight_split()
    if skip is not None:
        return SplitResult(performed=False, reason=skip, orientation=orientation)

    top = pane_id()
    if top is None:
        return SplitResult(
            performed=False,
            reason="cannot determine current pane id",
            orientation=orientation,
        )

    if orientation == "vertical":
        split_args = [
            "split-window",
            "-h",
            "-l",
            f"{100 - int(sidebar_cols * 100 / max(sidebar_cols + 60, 120))}%",
            "-P",
            "-F",
            "#{pane_id}",
            placeholder_cmd,
        ]
    else:
        split_args = [
            "split-window",
            "-v",
            "-l",
            f"{bottom_pct}%",
            "-P",
            "-F",
            "#{pane_id}",
            placeholder_cmd,
        ]

    r = _run_tmux(*split_args)
    if r.returncode != 0:
        return SplitResult(
            performed=False,
            reason=f"split-window failed: {r.stderr.strip()}",
            orientation=orientation,
        )
    bottom = r.stdout.strip() or None

    # Resize the TUI pane (original) to its fixed dimension.
    if orientation == "vertical":
        _run_tmux("resize-pane", "-t", top, "-x", str(sidebar_cols))
    else:
        _run_tmux("resize-pane", "-t", top, "-y", str(top_rows))

    # Focus the TUI pane so keypresses reach the Rich event loop
    # immediately instead of going to the placeholder shell.
    _run_tmux("select-pane", "-t", top)

    _set_window_option(_SPLIT_FLAG, "on")
    _set_window_option(_ORIENTATION_KEY, orientation)
    if bottom is not None:
        _set_window_option(_BOTTOM_PANE_KEY, bottom)

    return SplitResult(
        performed=True,
        reason="split performed",
        top_pane=top,
        bottom_pane=bottom,
        orientation=orientation,
    )


def run_in_pane(
    pane: str,
    cmd: str,
    *,
    env: dict[str, str] | None = None,
    focus: bool = True,
) -> bool:
    """Replace the given pane's current process with ``cmd``.

    Uses ``tmux respawn-pane -k`` so the child owns the pane cleanly
    and the TUI is undisturbed. When ``focus`` is True (the default)
    the target pane is also selected so the user's next keystroke
    goes to the freshly spawned demo instead of the Rich TUI sidebar.
    Returns True on success.
    """
    if not in_tmux() or _tmux() is None:
        return False
    args = ["respawn-pane", "-k", "-t", pane]
    if env:
        for k, v in env.items():
            args.extend(["-e", f"{k}={v}"])
    args.append(cmd)
    r = _run_tmux(*args)
    if r.returncode != 0:
        return False
    if focus:
        _run_tmux("select-pane", "-t", pane)
    return True


def wait_for_pane_exit(pane: str, *, poll_interval: float = 0.5) -> int:
    """Poll ``pane`` until it has no live process, return the exit status.

    Uses ``#{pane_dead}`` + ``#{pane_dead_status}``. If the pane has
    gone away entirely we return ``-1`` as a sentinel.
    """
    if not in_tmux() or _tmux() is None:
        return -1
    while True:
        r = _run_tmux(
            "display-message",
            "-p",
            "-t",
            pane,
            "#{pane_dead}|#{pane_dead_status}",
        )
        if r.returncode != 0:
            return -1
        parts = r.stdout.strip().split("|", 1)
        if parts and parts[0] == "1":
            try:
                return int(parts[1]) if len(parts) > 1 else 0
            except ValueError:
                return 0
        time.sleep(poll_interval)


def set_pane_title(title: str, *, pane: str | None = None) -> None:
    """Set the ``#T`` title of a pane (for display in the status line)."""
    if not in_tmux() or _tmux() is None:
        return
    target = ["-t", pane] if pane else []
    _run_tmux("select-pane", *target, "-T", title)
