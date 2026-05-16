"""Installer for autosre Claude Code hooks.

Writes autosre hook entries into ``~/.claude/settings.json`` (or a
project-level ``.claude/settings.json`` when ``--project`` is passed) so
that **bare** ``claude`` — not just ``autosre claude`` — fires the
plan-review loop, the bash guard, and the other autosre hooks.

Design invariants:

- **Wrapped schema.** Per Claude Code's documented settings schema, event
  handlers live under a top-level ``"hooks"`` object — ``settings["hooks"]
  ["PreToolUse"]``, not ``settings["PreToolUse"]``. Entries written at
  the top level are silently ignored by Claude Code.
- **PATH-resolved command.** Hook entries invoke ``autosre hooks run
  <module>`` and rely on ``autosre`` being on ``$PATH`` (mise shim or
  venv). No absolute Python path is embedded, so upgrading Python doesn't
  require re-running ``autosre hooks install``.
- **Additive merge.** Existing user hook entries are preserved verbatim;
  we only append our entries to each event's list. Other user settings
  (``permissions``, ``enabledPlugins``, etc.) are untouched.
- **Sidecar marker.** A JSON file at ``~/.claude/.autosre-hooks-installed.json``
  records exactly which ``(event, matcher, command)`` triples we added, so
  ``uninstall`` can remove them surgically without touching the user's own
  hooks.
- **Idempotent.** Running ``install`` twice does not duplicate entries or
  rewrite the file.
- **Online-mode chain.** When run by bare ``claude``, the review chain
  defaults to ``DEFAULT_CHAINS["plan"] = ["codex", "local", "gemini", "claude"]``
  (codex gpt-5.4 xhigh primary, local fallback) because
  ``AUTOSRE_REVIEW_CHAIN`` is not set. ``autosre claude`` sets the env var
  to ``"local"`` to force local-only behavior.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

USER_SETTINGS_PATH = Path.home() / ".claude" / "settings.json"
SIDECAR_PATH = Path.home() / ".claude" / ".autosre-hooks-installed.json"


# ---------------------------------------------------------------------------
# Hook entry definitions — single source of truth for what we install
# ---------------------------------------------------------------------------


def _hook_cmd(module: str) -> str:
    """Render a hook command: ``autosre hooks run <module>``."""
    return f"autosre hooks run {module}"


def _planned_entries() -> list[dict[str, Any]]:
    """Return the full list of hook entries we want installed.

    Each entry is a dict with:
      - ``event``: the Claude Code hook event key (PreToolUse, Stop, …),
        nested under ``settings["hooks"][event]`` at write time.
      - ``matcher``: optional matcher string (None → no matcher)
      - ``command``: the shell command to run
      - ``timeout``: hook timeout in seconds
    """
    return [
        {
            "event": "PreToolUse",
            "matcher": "Bash",
            "command": _hook_cmd("pretooluse_bash_guard"),
            "timeout": 10,
        },
        {
            "event": "PreToolUse",
            "matcher": "Edit",
            "command": _hook_cmd("pretooluse_bash_guard"),
            "timeout": 10,
        },
        {
            "event": "PreToolUse",
            "matcher": "Write",
            "command": _hook_cmd("pretooluse_bash_guard"),
            "timeout": 10,
        },
        {
            "event": "PreToolUse",
            "matcher": "ExitPlanMode",
            "command": _hook_cmd("pretooluse_plan_review"),
            "timeout": 1260,
        },
        {
            "event": "PostToolUse",
            "matcher": "Bash",
            "command": _hook_cmd("posttooluse_audit"),
            "timeout": 5,
        },
        {
            "event": "PostToolUse",
            "matcher": "Bash",
            "command": _hook_cmd("telemetry_async"),
            "timeout": 5,
        },
        {
            "event": "PostToolUse",
            "matcher": "Bash",
            "command": _hook_cmd("post_commit_scan_update"),
            "timeout": 5,
        },
        {
            "event": "Stop",
            "matcher": None,
            "command": _hook_cmd("stop_session_check"),
            "timeout": 10,
        },
        {
            "event": "Stop",
            "matcher": None,
            "command": _hook_cmd("telemetry_async"),
            "timeout": 5,
        },
        {
            "event": "UserPromptSubmit",
            "matcher": None,
            "command": _hook_cmd("user_prompt_submit_branch_check"),
            "timeout": 5,
        },
        {
            "event": "PreCompact",
            "matcher": None,
            "command": _hook_cmd("precompact_context"),
            "timeout": 10,
        },
        {
            "event": "SubagentStart",
            "matcher": "Plan",
            "command": _hook_cmd("subagent_plan_context"),
            "timeout": 10,
        },
        {
            "event": "SubagentStart",
            "matcher": "Explore",
            "command": _hook_cmd("subagent_plan_context"),
            "timeout": 10,
        },
    ]


# ---------------------------------------------------------------------------
# Low-level settings merge / strip
# ---------------------------------------------------------------------------


def _load_settings(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(data, dict):
        return {}
    return data


def _save_settings(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def _find_matcher_block(
    event_list: list[dict[str, Any]],
    matcher: str | None,
) -> dict[str, Any] | None:
    """Find the first existing block in ``event_list`` with the given matcher.

    ``None`` means "no matcher key" (the universal Stop/UserPromptSubmit shape).
    """
    for block in event_list:
        if not isinstance(block, dict):
            continue
        if matcher is None and "matcher" not in block:
            return block
        if matcher is not None and block.get("matcher") == matcher:
            return block
    return None


def _merge_entry(
    settings: dict[str, Any],
    entry: dict[str, Any],
) -> bool:
    """Add a single hook entry to ``settings``. Returns True if anything changed."""
    event = entry["event"]
    matcher = entry["matcher"]
    command = entry["command"]
    timeout = entry["timeout"]

    hooks_root = settings.setdefault("hooks", {})
    if not isinstance(hooks_root, dict):
        return False
    event_list = hooks_root.setdefault(event, [])
    if not isinstance(event_list, list):
        # The user has something non-standard here; don't clobber.
        return False

    block = _find_matcher_block(event_list, matcher)
    if block is None:
        block = {"hooks": []}
        if matcher is not None:
            block["matcher"] = matcher
        event_list.append(block)

    hooks = block.setdefault("hooks", [])
    if not isinstance(hooks, list):
        return False

    # Skip if our exact command is already present
    for h in hooks:
        if isinstance(h, dict) and h.get("command") == command:
            return False

    hooks.append({"type": "command", "command": command, "timeout": timeout})
    return True


def _remove_entry(
    settings: dict[str, Any],
    entry: dict[str, Any],
) -> bool:
    """Remove a single hook entry from ``settings``. Returns True if found & removed.

    Looks in both the correct location (``settings["hooks"][event]``) and the
    legacy top-level location (``settings[event]``) so that sidecars written
    by older installer versions still clean up cleanly.
    """
    event = entry["event"]
    matcher = entry["matcher"]
    command = entry["command"]

    changed = False
    hooks_root = settings.get("hooks")

    def _strip(event_list: list[Any]) -> bool:
        local_changed = False
        for block in list(event_list):
            if not isinstance(block, dict):
                continue
            if matcher is None and "matcher" in block:
                continue
            if matcher is not None and block.get("matcher") != matcher:
                continue
            hooks = block.get("hooks")
            if not isinstance(hooks, list):
                continue
            before = len(hooks)
            block["hooks"] = [
                h for h in hooks if not (isinstance(h, dict) and h.get("command") == command)
            ]
            if len(block["hooks"]) != before:
                local_changed = True
            if not block["hooks"]:
                event_list.remove(block)
        return local_changed

    if isinstance(hooks_root, dict):
        event_list = hooks_root.get(event)
        if isinstance(event_list, list):
            changed = _strip(event_list) or changed
            if not event_list:
                hooks_root.pop(event, None)
        if not hooks_root:
            settings.pop("hooks", None)

    # Legacy top-level shape (pre-wrapper installer). Clean up in place.
    legacy_list = settings.get(event)
    if isinstance(legacy_list, list):
        changed = _strip(legacy_list) or changed
        if not legacy_list:
            settings.pop(event, None)

    return changed


# ---------------------------------------------------------------------------
# Sidecar tracking
# ---------------------------------------------------------------------------


def _load_sidecar(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(data, dict):
        return None
    return data


def _save_sidecar(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def install(
    settings_path: Path | None = None,
    sidecar_path: Path | None = None,
) -> dict[str, Any]:
    """Install autosre hook entries into Claude Code settings.

    Idempotent behavior:

    - Fresh install (no sidecar): add every planned entry to settings.
    - Re-install with the same planned set: no-op — the file is only
      rewritten if a drifted entry had to be re-added.
    - Re-install with a changed planned set (upgrade changed the hook
      list): strip previously-tracked entries and install the fresh set.
      Existing user hook entries are preserved throughout.
    """
    resolved_settings = settings_path or USER_SETTINGS_PATH
    resolved_sidecar = sidecar_path or SIDECAR_PATH

    planned = _planned_entries()

    settings = _load_settings(resolved_settings)
    existing_sidecar = _load_sidecar(resolved_sidecar)

    # Fast path: sidecar claims the exact same entries are already installed.
    if existing_sidecar is not None and existing_sidecar.get("entries") == planned:
        # Re-verify the entries are live (the user may have pruned them by
        # hand). Re-add any drifted-out entries.
        added: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        for entry in planned:
            if _merge_entry(settings, entry):
                added.append(entry)
            else:
                skipped.append(entry)
        if added:
            _save_settings(resolved_settings, settings)
        return {
            "added": added,
            "skipped": skipped,
            "settings_path": str(resolved_settings),
            "sidecar_path": str(resolved_sidecar),
        }

    # Planned set changed (or no sidecar): strip old tracked entries, then
    # install the fresh set. User-owned hook entries are preserved because
    # they're not in the old sidecar's entries list.
    if existing_sidecar is not None:
        for entry in existing_sidecar.get("entries", []):
            _remove_entry(settings, entry)

    added = []
    skipped = []
    for entry in planned:
        if _merge_entry(settings, entry):
            added.append(entry)
        else:
            skipped.append(entry)

    _save_settings(resolved_settings, settings)

    sidecar_data = {
        "version": 2,
        "entries": planned,
    }
    _save_sidecar(resolved_sidecar, sidecar_data)

    return {
        "added": added,
        "skipped": skipped,
        "settings_path": str(resolved_settings),
        "sidecar_path": str(resolved_sidecar),
    }


def uninstall(
    settings_path: Path | None = None,
    sidecar_path: Path | None = None,
) -> dict[str, Any]:
    """Remove autosre hook entries from Claude Code settings.

    Uses the sidecar to know which entries were ours. Entries the user
    added by hand are preserved.
    """
    resolved_settings = settings_path or USER_SETTINGS_PATH
    resolved_sidecar = sidecar_path or SIDECAR_PATH

    sidecar = _load_sidecar(resolved_sidecar)
    if sidecar is None:
        return {
            "removed": [],
            "settings_path": str(resolved_settings),
            "sidecar_path": str(resolved_sidecar),
            "note": "no sidecar found — nothing to uninstall",
        }

    settings = _load_settings(resolved_settings)
    removed: list[dict[str, Any]] = []
    for entry in sidecar.get("entries", []):
        if _remove_entry(settings, entry):
            removed.append(entry)

    _save_settings(resolved_settings, settings)

    if resolved_sidecar.exists():
        resolved_sidecar.unlink()

    return {
        "removed": removed,
        "settings_path": str(resolved_settings),
        "sidecar_path": str(resolved_sidecar),
    }


def status(
    settings_path: Path | None = None,
    sidecar_path: Path | None = None,
) -> dict[str, Any]:
    """Report installer state.

    Returns ``installed`` (bool), ``entries`` (list from sidecar), and
    ``drift`` (list of sidecar entries whose command no longer appears in
    the live settings file — indicating the user modified it by hand).
    """
    resolved_settings = settings_path or USER_SETTINGS_PATH
    resolved_sidecar = sidecar_path or SIDECAR_PATH

    sidecar = _load_sidecar(resolved_sidecar)
    if sidecar is None:
        return {
            "installed": False,
            "entries": [],
            "drift": [],
            "settings_path": str(resolved_settings),
            "sidecar_path": str(resolved_sidecar),
        }

    settings = _load_settings(resolved_settings)
    hooks_root = settings.get("hooks")
    entries = sidecar.get("entries", [])
    drift: list[dict[str, Any]] = []
    for entry in entries:
        event_list = hooks_root.get(entry["event"], []) if isinstance(hooks_root, dict) else []
        found = False
        if isinstance(event_list, list):
            for block in event_list:
                if not isinstance(block, dict):
                    continue
                if entry["matcher"] is None and "matcher" in block:
                    continue
                if entry["matcher"] is not None and block.get("matcher") != entry["matcher"]:
                    continue
                hooks = block.get("hooks", [])
                if isinstance(hooks, list):
                    for h in hooks:
                        if isinstance(h, dict) and h.get("command") == entry["command"]:
                            found = True
                            break
                if found:
                    break
        if not found:
            drift.append(entry)

    return {
        "installed": True,
        "entries": entries,
        "drift": drift,
        "settings_path": str(resolved_settings),
        "sidecar_path": str(resolved_sidecar),
    }
