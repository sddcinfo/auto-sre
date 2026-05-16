# Claude Code hook inputs — schema + autosre conventions

This document describes what Claude Code sends to each of the autosre
hook entry points, how the hooks resolve paths, and how they fail. The
shared :mod:`autosre.claude_hooks._io` layer enforces every rule below.

## Goals

1. **No hardcoded home-relative paths.** All path discovery goes
   through :func:`autosre.paths.claude_plans_dir` (honours
   ``$CLAUDE_CONFIG_DIR``) or :func:`autosre.paths.hooks_log_dir`
   (honours ``$XDG_DATA_HOME``).
2. **Every invocation is archived.** The full raw stdin JSON is
   appended to ``<data_dir>/hooks/hooks-raw.jsonl`` (20 MB cap, rotated
   to ``.1``). When ``$AUTOSRE_HOOKS_DUMP_DIR`` is set, each
   invocation also drops a per-hook JSON file there — use this to
   collect golden fixtures for tests.
3. **Schema drift is visible.** Unknown top-level keys produce a `warn`
   line in ``<data_dir>/hooks/<Event>.log`` instead of silently being
   ignored.
4. **Silent no-ops become loud in strict mode.** When
   ``AUTOSRE_HOOKS_STRICT=1`` is exported, every fail-open path surfaces
   a visible ``systemMessage`` (PreToolUse) or ``message`` (continue)
   so the user sees when a hook decided not to act.

## Environment variables

| Variable | Purpose |
|---|---|
| ``CLAUDE_CONFIG_DIR`` | Base for Claude Code state (``plans/`` lives here). Falls back to ``~/.claude``. |
| ``CLAUDE_PLAN_FILE`` | If set, overrides plan-file discovery in :func:`_io.resolve_plan_file`. |
| ``AUTOSRE_HOOKS_DUMP_DIR`` | If set, every hook invocation writes its raw input here as a JSON file. |
| ``AUTOSRE_HOOKS_STRICT`` | ``1``/``true``/``yes``/``on`` — make fail-open paths visible. |
| ``AUTOSRE_HOOKS_PLAN_MAX_AGE_SECONDS`` | Cutoff for the plans-dir fallback (default 3600s). |
| ``XDG_DATA_HOME`` | Standard XDG var; hook logs live at ``<XDG_DATA_HOME>/autosre/hooks``. |

## Events and resolution rules

### PreToolUse(ExitPlanMode) — `pretooluse_plan_review.py`

**Input keys used**

- ``tool_input.planFilePath`` / ``tool_input.plan_file_path`` (primary).
- ``planFilePath`` at top level (some CC versions).
- ``$CLAUDE_PLAN_FILE`` env var.
- Newest ``*.md`` in :func:`autosre.paths.claude_plans_dir` whose mtime
  is within ``$AUTOSRE_HOOKS_PLAN_MAX_AGE_SECONDS`` (filesystem
  fallback).

**Fail policy:** fail-open. The `additionalContext` on allow says
"Proceeding without review." When a P0/P1 finding comes back the hook
issues a `deny` with findings baked into `additionalContext`.

### PreToolUse(Bash|*) — `pretooluse_bash_guard.py`

Forwards stdin to ``autosre hooks-backend guard``. Security-critical:
fail-closed on timeout / missing binary / empty stdout. The shared
archive/log layer still captures the input.

### PostToolUse(Bash) — `posttooluse_audit.py`

Reads ``tool_input.command``; logs to
:func:`autosre.paths.hook_audit_log`. Always fails open.

### Stop — `stop_session_check.py`

Thin shim around ``autosre hooks-backend stop-check``. Fail-open on
timeout / missing binary.

### UserPromptSubmit — `user_prompt_submit_branch_check.py`

Reads ``git branch --show-current``; emits "on feature branch X" or
"on main" at most once per hour (marker at
:func:`autosre.paths.branch_warned_marker`).

### PreCompact — `precompact_context.py`

Reads ``CLAUDE.md`` + ``git branch`` + ``git status --short`` from
``inv.cwd`` (or ``Path.cwd()`` as fallback). Emits a ``systemMessage``.

### SubagentStart — `subagent_plan_context.py`

Reads CLAUDE.md, ``.claude/rules/*.md``, git state, top-level dir listing
from ``inv.cwd``. Emits ``hookSpecificOutput.additionalContext`` within
a 12K budget.

### PostCommit — `post_commit_scan_update.py`

No-op stub that still archives input (for drift analysis).

### Telemetry — `telemetry_async.py`

No-op stub that still archives input.

## Fail policy matrix

| Hook | On error |
|---|---|
| `pretooluse_plan_review` | **Fail-open loud** — allow + advisory context; strict-mode adds `systemMessage`. |
| `pretooluse_bash_guard`  | **Fail-closed** — deny. Security-critical. |
| `posttooluse_audit`      | Fail-open silent (audit log is a convenience). |
| `stop_session_check`     | Fail-open — never trap the user at session end. |
| `user_prompt_submit_branch_check` | Fail-open. |
| `precompact_context`     | Fail-open — must not block compaction. |
| `subagent_plan_context`  | Fail-open — must not block subagent start. |
| `post_commit_scan_update`| No-op. |
| `telemetry_async`        | No-op. |

## Collecting golden fixtures

```
export AUTOSRE_HOOKS_DUMP_DIR=$HOME/tmp/hook-fixtures
# drive Claude Code through each event once
ls $AUTOSRE_HOOKS_DUMP_DIR   # per-event JSON files
```

Sanitize (strip session IDs, personal paths) and commit selected
samples to ``tests/fixtures/claude_hooks/`` for regression tests.
