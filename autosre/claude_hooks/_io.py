"""Shared IO layer for Claude Code hook scripts.

Every ``autosre/claude_hooks/*.py`` entry-point parses the same stdin JSON
shape, writes to the same set of state directories, and emits one of a
handful of decision envelopes. Before this module existed each hook
hand-rolled those three concerns, which (a) drifted apart over time, (b)
silently fell back to fail-open when Claude Code changed an input key
name, and (c) gave us no way to diff "what CC actually sent" against
"what the hook expected".

Design rules
------------

- **No hardcoded home-relative paths.** Plan discovery honours
  ``$CLAUDE_CONFIG_DIR`` via :func:`autosre.paths.claude_plans_dir`;
  logs honour ``$XDG_DATA_HOME`` via :func:`autosre.paths.hooks_log_dir`.
- **Every invocation's raw input is archived** to
  ``$XDG_DATA_HOME/autosre/hooks/hooks-raw.jsonl`` (size-capped). If the
  ``AUTOSRE_HOOKS_DUMP_DIR`` env var is set, a per-invocation JSON file
  is also dropped there — intended for collecting golden test fixtures.
- **Strict mode** (``AUTOSRE_HOOKS_STRICT=1``) turns every
  :func:`fail_open` path into a visible ``systemMessage`` on top of the
  allow/continue payload so silent no-ops become observable.
- **Security-critical hooks fail closed.** :func:`fail_closed` exists for
  the bash guard; it emits a deny with the error baked into the reason.

The module has no side effects at import time beyond creating the log
directory (via :mod:`autosre.paths`, which is already test-friendly).
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from autosre import paths

_RAW_JSONL_MAX_BYTES = 20 * 1024 * 1024  # 20 MB; rotated to .1 on overflow

_KNOWN_TOP_LEVEL_KEYS = frozenset(
    {
        "session_id",
        "transcript_path",
        "cwd",
        "hook_event_name",
        "tool_name",
        "tool_input",
        "tool_response",
        "prompt",
        "matcher",
        "message",
        "stop_hook_active",
        # plan-mode convention seen in some CC versions:
        "planFilePath",
        "plan_file_path",
    },
)


@dataclasses.dataclass(frozen=True)
class HookInvocation:
    """Parsed view of a Claude Code hook invocation.

    ``event`` is the logical event name (e.g. ``"PreToolUse"``); it is
    pulled from ``hook_event_name`` when present, otherwise inferred from
    the caller module's filename so tests don't have to fake the field.
    ``raw`` retains the full input dict so hook-specific logic can reach
    into fields :class:`HookInvocation` doesn't promote.
    """

    event: str
    tool_name: str | None
    session_id: str
    cwd: Path
    tool_input: dict[str, Any]
    tool_response: dict[str, Any] | None
    transcript_path: Path | None
    raw: dict[str, Any]
    hook_script: str


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


_SCRIPT_PREFIX_TO_EVENT: tuple[tuple[str, str], ...] = (
    ("pretooluse_", "PreToolUse"),
    ("posttooluse_", "PostToolUse"),
    ("stop_", "Stop"),
    ("user_prompt_submit_", "UserPromptSubmit"),
    ("precompact_", "PreCompact"),
    ("subagent_", "SubagentStart"),
    ("post_commit_", "PostCommit"),
    ("telemetry_", "Telemetry"),
)


def _infer_event_from_script(hook_script: str) -> str:
    """Map a hook-script path to an event name when CC didn't send one."""
    name = Path(hook_script).stem.lower()
    for prefix, event in _SCRIPT_PREFIX_TO_EVENT:
        if name.startswith(prefix):
            return event
    return "Unknown"


def parse_stdin(hook_script: str) -> HookInvocation:
    """Read and parse Claude Code's stdin JSON.

    On JSON errors returns a :class:`HookInvocation` with ``raw={}`` so
    callers can decide whether to fail-open or fail-closed; the error is
    logged to the per-event log.
    """

    try:
        payload = sys.stdin.read()
    except OSError as exc:
        _log_event(_infer_event_from_script(hook_script), f"stdin read failed: {exc}")
        payload = ""

    raw: dict[str, Any] = {}
    if payload:
        try:
            parsed = json.loads(payload)
            if isinstance(parsed, dict):
                raw = parsed
            else:
                _log_event(
                    _infer_event_from_script(hook_script),
                    f"stdin not a JSON object: {type(parsed).__name__}",
                )
        except json.JSONDecodeError as exc:
            _log_event(
                _infer_event_from_script(hook_script),
                f"stdin JSON decode failed: {exc}; first 120 chars: {payload[:120]!r}",
            )

    event = raw.get("hook_event_name") or _infer_event_from_script(hook_script)
    tool_name = raw.get("tool_name")
    session_id = str(raw.get("session_id", "unknown"))
    cwd_raw = raw.get("cwd") or str(Path.cwd())
    transcript_raw = raw.get("transcript_path")

    tool_input = raw.get("tool_input")
    if not isinstance(tool_input, dict):
        tool_input = {}

    tool_response = raw.get("tool_response")
    if not isinstance(tool_response, dict):
        tool_response = None

    inv = HookInvocation(
        event=str(event),
        tool_name=str(tool_name) if tool_name else None,
        session_id=session_id,
        cwd=Path(cwd_raw),
        tool_input=tool_input,
        tool_response=tool_response,
        transcript_path=Path(transcript_raw) if transcript_raw else None,
        raw=raw,
        hook_script=hook_script,
    )

    # Archive the raw input for post-mortem diffing and warn on unknown keys.
    _archive_raw(inv)
    _warn_on_unknown_keys(inv)
    return inv


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def plans_dir() -> Path:
    """Directory Claude Code writes plan files to. Honors $CLAUDE_CONFIG_DIR."""
    return paths.claude_plans_dir()


def resolve_plan_file(inv: HookInvocation) -> Path | None:
    """Find the ExitPlanMode plan file across CC-version shapes.

    Fallback order, no hardcoded home-relative paths:

    1. ``tool_input.planFilePath`` / ``plan_file_path``.
    2. Top-level ``raw.planFilePath``.
    3. ``$CLAUDE_PLAN_FILE`` env var.
    4. Newest ``*.md`` in :func:`plans_dir` (ignoring files older than the
       ``$AUTOSRE_HOOKS_PLAN_MAX_AGE_SECONDS`` cutoff — default 3600s —
       so stale plans from previous sessions don't get grabbed).

    Returns ``None`` if no candidate exists; callers decide whether to
    fail-open or fail-loud.
    """

    for key in ("planFilePath", "plan_file_path"):
        cand = inv.tool_input.get(key)
        if cand and Path(cand).exists():
            log(inv, f"plan from tool_input.{key}: {cand}")
            return Path(cand)
        if cand:
            log(inv, f"tool_input.{key} not usable ({cand!r})", level="warn")

    top = inv.raw.get("planFilePath")
    if top and Path(top).exists():
        log(inv, f"plan from raw.planFilePath: {top}")
        return Path(top)

    env_path = os.environ.get("CLAUDE_PLAN_FILE")
    if env_path and Path(env_path).exists():
        log(inv, f"plan from $CLAUDE_PLAN_FILE: {env_path}")
        return Path(env_path)

    try:
        max_age = float(os.environ.get("AUTOSRE_HOOKS_PLAN_MAX_AGE_SECONDS", "3600"))
    except ValueError:
        max_age = 3600.0

    pdir = plans_dir()
    if pdir.is_dir():
        now = time.time()
        candidates = [p for p in pdir.glob("*.md") if (now - p.stat().st_mtime) <= max_age]
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        if candidates:
            chosen = candidates[0]
            age = now - chosen.stat().st_mtime
            log(inv, f"plan from {pdir} (newest, age={age:.0f}s): {chosen}")
            return chosen
        log(inv, f"no recent *.md in {pdir} (max_age={max_age:.0f}s)", level="warn")
    else:
        log(inv, f"plans dir does not exist: {pdir}", level="warn")

    log(inv, f"raw keys: {sorted(inv.raw.keys())}", level="warn")
    if inv.tool_input:
        log(inv, f"tool_input keys: {sorted(inv.tool_input.keys())}", level="warn")
    return None


def resolve_transcript_path(inv: HookInvocation) -> Path | None:
    """Return ``inv.transcript_path`` if it exists on disk, else ``None``."""
    if inv.transcript_path and inv.transcript_path.exists():
        return inv.transcript_path
    return None


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def _event_log_path(event: str) -> Path:
    return paths.hooks_log_dir() / f"{event}.log"


def _log_event(event: str, msg: str, *, level: str = "info") -> None:
    """Write a line to the per-event human-readable log. Never raises."""
    ts = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    line = f"[{ts}] [{level}] {msg}"
    with contextlib.suppress(OSError):
        print(f"[autosre-hook:{event}] {msg}", file=sys.stderr)
    try:
        path = _event_log_path(event)
        with path.open("a") as f:
            f.write(line + "\n")
    except OSError:
        pass


def log(inv: HookInvocation, msg: str, *, level: str = "info") -> None:
    """Log a line tagged with the current invocation's event."""
    _log_event(inv.event, f"[{inv.session_id}] {msg}", level=level)


def _archive_raw(inv: HookInvocation) -> None:
    """Append one JSONL line to hooks-raw.jsonl, rotating at the size cap.

    Also writes a per-invocation JSON file to ``$AUTOSRE_HOOKS_DUMP_DIR``
    if that env var is set (used to collect fixtures for Phase B1).
    """

    record = {
        "ts": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "event": inv.event,
        "tool_name": inv.tool_name,
        "session_id": inv.session_id,
        "hook_script": inv.hook_script,
        "raw": inv.raw,
    }
    try:
        path = paths.hooks_raw_jsonl()
        if path.exists() and path.stat().st_size > _RAW_JSONL_MAX_BYTES:
            path.replace(path.with_suffix(path.suffix + ".1"))
        with path.open("a") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except OSError:
        pass

    dump_dir = os.environ.get("AUTOSRE_HOOKS_DUMP_DIR")
    if dump_dir:
        try:
            target_dir = Path(dump_dir)
            target_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
            fname = f"{inv.event}-{ts}-{os.getpid()}.json"
            (target_dir / fname).write_text(
                json.dumps(inv.raw, indent=2, default=str),
            )
        except OSError:
            pass


def _warn_on_unknown_keys(inv: HookInvocation) -> None:
    """Log (don't fail) if CC sent a top-level key we don't recognize."""
    unknown = [k for k in inv.raw if k not in _KNOWN_TOP_LEVEL_KEYS]
    if unknown:
        log(
            inv,
            f"unknown top-level keys from Claude Code (update _io.py): {sorted(unknown)}",
            level="warn",
        )


# ---------------------------------------------------------------------------
# Decision emission
# ---------------------------------------------------------------------------


def _write_stdout(payload: dict[str, Any]) -> None:
    try:
        sys.stdout.write(json.dumps(payload))
        sys.stdout.flush()
    except OSError:
        pass


def emit_pretooluse_allow(
    additional_context: str | None = None,
    *,
    system_message: str | None = None,
) -> int:
    """Emit a PreToolUse ``allow`` decision. Returns 0 so ``return emit_*(...)`` works."""
    payload: dict[str, Any] = {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "allow",
        },
    }
    if additional_context:
        payload["hookSpecificOutput"]["additionalContext"] = additional_context
    if system_message:
        payload["systemMessage"] = system_message
    _write_stdout(payload)
    return 0


def emit_pretooluse_deny(
    reason: str,
    *,
    additional_context: str | None = None,
) -> int:
    """Emit a PreToolUse ``deny`` decision. Returns 0."""
    payload: dict[str, Any] = {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        },
    }
    if additional_context:
        payload["hookSpecificOutput"]["additionalContext"] = additional_context
    _write_stdout(payload)
    return 0


def emit_pretooluse_ask(reason: str) -> int:
    _write_stdout(
        {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "ask",
                "permissionDecisionReason": reason,
            },
        },
    )
    return 0


def emit_continue(message: str | None = None) -> int:
    payload: dict[str, Any] = {"result": "continue"}
    if message:
        payload["message"] = message
    _write_stdout(payload)
    return 0


def emit_ok() -> int:
    _write_stdout({"result": "ok"})
    return 0


def emit_system_message(message: str) -> int:
    _write_stdout({"systemMessage": message})
    return 0


def emit_subagent_context(context: str) -> int:
    _write_stdout(
        {
            "hookSpecificOutput": {
                "hookEventName": "SubagentStart",
                "additionalContext": context,
            },
        },
    )
    return 0


def emit_raw_stdout(stdout: str, exit_code: int = 0) -> int:
    """Forward a subprocess's stdout verbatim (used by the guard/stop shims)."""
    try:
        sys.stdout.write(stdout)
        sys.stdout.flush()
    except OSError:
        pass
    return exit_code


# ---------------------------------------------------------------------------
# Fail paths
# ---------------------------------------------------------------------------


def _strict_mode() -> bool:
    return os.environ.get("AUTOSRE_HOOKS_STRICT", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def fail_open_pretooluse(inv: HookInvocation, reason: str) -> int:
    """PreToolUse fail-open: allow + advisory additionalContext. Strict
    mode additionally raises a visible systemMessage.
    """
    log(inv, f"fail_open: {reason}", level="warn")
    sys_msg = f"[autosre-hooks] {inv.event} skipped: {reason}" if _strict_mode() else None
    return emit_pretooluse_allow(
        additional_context=f"{inv.event}: {reason}. Proceeding without review.",
        system_message=sys_msg,
    )


def fail_closed_pretooluse(inv: HookInvocation, reason: str) -> int:
    """PreToolUse fail-closed: deny (for security-critical hooks)."""
    log(inv, f"fail_closed: {reason}", level="error")
    return emit_pretooluse_deny(reason)


def fail_open_continue(inv: HookInvocation, reason: str) -> int:
    """Non-PreToolUse fail-open: continue + (loud when strict) message."""
    log(inv, f"fail_open: {reason}", level="warn")
    msg = f"[autosre-hooks] {inv.event} skipped: {reason}" if _strict_mode() else None
    return emit_continue(message=msg)
