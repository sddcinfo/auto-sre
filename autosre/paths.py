"""XDG-compliant path helpers for autosre state, config, cache, and logs.

All state-producing subsystems in autosre route their paths through this
module so that test fixtures can redirect everything with a single
`XDG_DATA_HOME` / `XDG_CONFIG_HOME` / `XDG_STATE_HOME` / `XDG_CACHE_HOME`
override.

Reuses the same env-var semantics as ``autosre/backends/base.py:35``
(``ACTIVE_STATE_FILE``) so both modules agree on where autosre's XDG root
lives.
"""

from __future__ import annotations

import os
from pathlib import Path


def _xdg(env_var: str, default_relative: str) -> Path:
    """Return ``$env_var`` if set, otherwise ``$HOME/default_relative``."""
    raw = os.environ.get(env_var)
    if raw:
        return Path(raw)
    return Path.home() / default_relative


def data_dir() -> Path:
    """XDG data directory for autosre (``$XDG_DATA_HOME/autosre``)."""
    directory = _xdg("XDG_DATA_HOME", ".local/share") / "autosre"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def config_dir() -> Path:
    """XDG config directory for autosre (``$XDG_CONFIG_HOME/autosre``)."""
    directory = _xdg("XDG_CONFIG_HOME", ".config") / "autosre"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def state_dir() -> Path:
    """XDG state directory for autosre (``$XDG_STATE_HOME/autosre``)."""
    directory = _xdg("XDG_STATE_HOME", ".local/state") / "autosre"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def cache_dir() -> Path:
    """XDG cache directory for autosre (``$XDG_CACHE_HOME/autosre``)."""
    directory = _xdg("XDG_CACHE_HOME", ".cache") / "autosre"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


# ---------------------------------------------------------------------------
# Subsystem-specific subdirs. Every caller uses these helpers instead of
# hand-rolling paths so the XDG overrides work transparently.
# ---------------------------------------------------------------------------


def review_state_dir() -> Path:
    """Plan-review iteration state (``<data_dir>/review-state``)."""
    directory = data_dir() / "review-state"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def review_log_dir() -> Path:
    """Per-iteration plan-review logs (``<data_dir>/review-log``)."""
    directory = data_dir() / "review-log"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def guard_rules_file() -> Path:
    """User's guard-rules.yaml (``<config_dir>/guard-rules.yaml``)."""
    return config_dir() / "guard-rules.yaml"


def guard_approvals_dir() -> Path:
    """Pre-approved command cache (``<state_dir>/approvals``)."""
    directory = state_dir() / "approvals"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def hook_audit_log() -> Path:
    """Bash command audit trail (``<state_dir>/hook-audit.log``)."""
    return state_dir() / "hook-audit.log"


def hook_blocked_log() -> Path:
    """Blocked-command audit log (``<state_dir>/hook-blocked.log``)."""
    return state_dir() / "hook-blocked.log"


def hook_errors_log() -> Path:
    """Fail-open error log for hook scripts (``<state_dir>/hook-errors.log``)."""
    return state_dir() / "hook-errors.log"


def branch_warned_marker() -> Path:
    """UserPromptSubmit branch-warn marker (``<state_dir>/branch-warned``)."""
    return state_dir() / "branch-warned"


def hooks_log_dir() -> Path:
    """Per-event Claude Code hook logs (``<data_dir>/hooks``)."""
    directory = data_dir() / "hooks"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def hooks_raw_jsonl() -> Path:
    """Size-capped JSONL archive of every hook invocation's raw stdin."""
    return hooks_log_dir() / "hooks-raw.jsonl"


def claude_config_dir() -> Path:
    """Claude Code config dir (``$CLAUDE_CONFIG_DIR`` or ``~/.claude``)."""
    raw = os.environ.get("CLAUDE_CONFIG_DIR")
    if raw:
        return Path(raw)
    return Path.home() / ".claude"


def claude_plans_dir() -> Path:
    """Directory Claude Code writes plan-mode plan files to."""
    return claude_config_dir() / "plans"


def capabilities_index_file() -> Path:
    """On-disk cache for the MCP capabilities catalog."""
    return data_dir() / "capabilities-index.json"
