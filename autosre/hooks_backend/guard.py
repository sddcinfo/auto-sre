"""Guard logic for PreToolUse hook.

Reads rules from ``autosre/hooks_backend/config/guard-rules.yaml`` (user override
at ``$XDG_CONFIG_HOME/autosre/guard-rules.yaml``) and evaluates Bash commands
against them. Called via ``autosre hooks-backend guard`` from the installed
wrapper at ``autosre/claude_hooks/pretooluse_bash_guard.py``.

The rule evaluator honors ``AUTOSRE_GUARD_RULES`` to point at an alternate
rules file (used by tests and for local experimentation).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import traceback
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import click
import yaml

from autosre import paths

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration loading
# ============================================================================

_CONFIG_CACHE: dict[str, Any] | None = None
_CONFIG_CACHE_PATH: Path | None = None
_CONFIG_CACHE_MTIME: float = 0


def _find_config_path() -> Path:
    """Return the guard-rules.yaml path.

    Priority:
      1. ``AUTOSRE_GUARD_RULES`` env var (explicit override, for tests).
      2. ``$XDG_CONFIG_HOME/autosre/guard-rules.yaml`` (user-installed).
      3. The packaged default at ``autosre/hooks_backend/config/guard-rules.yaml``
         (used the first time, before ``autosre hooks-backend init`` is run).
    """
    env_path = os.environ.get("AUTOSRE_GUARD_RULES")
    if env_path:
        from pathlib import Path as _Path

        return _Path(env_path)

    user_path = paths.guard_rules_file()
    if user_path.exists():
        return user_path

    from pathlib import Path as _Path

    return _Path(__file__).resolve().parent / "config" / "guard-rules.yaml"


def load_config() -> dict[str, Any]:
    """Load and cache guard rules configuration.

    Re-reads on file change (mtime check) for hot-reload during development.
    """
    global _CONFIG_CACHE, _CONFIG_CACHE_PATH, _CONFIG_CACHE_MTIME  # noqa: PLW0603

    config_path = _find_config_path()

    try:
        current_mtime = config_path.stat().st_mtime
    except OSError:
        current_mtime = 0

    if (
        _CONFIG_CACHE is not None
        and config_path == _CONFIG_CACHE_PATH
        and current_mtime == _CONFIG_CACHE_MTIME
    ):
        return _CONFIG_CACHE

    if not config_path.exists():
        raise click.ClickException(
            f"Guard rules not found: {config_path}\nRun: autosre hooks-backend init",
        )

    with config_path.open() as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        config = {}

    _CONFIG_CACHE = config
    _CONFIG_CACHE_PATH = config_path
    _CONFIG_CACHE_MTIME = current_mtime
    return config


# ============================================================================
# Helper functions
# ============================================================================


def get_command_hash(command: str) -> str:
    """Generate a short hash for a command (for approval file names)."""
    return hashlib.sha256(command.encode()).hexdigest()[:16]


def check_approval(command: str, config: dict[str, Any]) -> bool:
    """Check if a command has been pre-approved by the user.

    Approvals are stored as files in ``<state_dir>/approvals/<hash>`` and
    expire after ``approval_expiry_seconds`` (default 60).
    """
    import time

    settings = config.get("settings", {})
    expiry = settings.get("approval_expiry_seconds", 60)

    approvals_dir = paths.guard_approvals_dir()
    cmd_hash = get_command_hash(command)
    approval_file = approvals_dir / cmd_hash

    if not approval_file.exists():
        return False

    try:
        file_age = time.time() - approval_file.stat().st_mtime
        if file_age > expiry:
            approval_file.unlink(missing_ok=True)
            return False
        # Valid approval -- one-time use
        approval_file.unlink(missing_ok=True)
        return True
    except OSError:
        return False


def log_block(command: str, reason: str, suggested: str | None, config: dict[str, Any]) -> None:
    """Log blocked command for metrics."""
    del config  # Reserved for future per-repo log overrides.
    log_file = paths.hook_blocked_log()

    try:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "command": command,
            "reason": reason,
            "suggested": suggested,
        }
        with log_file.open("a") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError:
        pass


def detect_environment(command: str) -> str:
    """Extract environment from command args (--env / -e flags)."""
    match = re.search(r"(?:--env[= ]|-e[= ]?)(\w+)", command)
    if match:
        env = match.group(1)
        if env == "production":
            return "prod"
        if env in ("dev", "test", "prod"):
            return env
    return "dev"


def extract_pr_number(command: str) -> str | None:
    """Extract a numeric ID (PR number, run ID) from a command."""
    match = re.search(r"\b(\d+)\b", command)
    return match.group(1) if match else None


def extract_git_directory(command: str) -> str | None:
    """Extract the -C directory from a git command, if present."""
    match = re.search(r"git\s+-C\s*[= ]?(\S+)", command)
    return match.group(1) if match else None


def split_chained_commands(command: str) -> list[str]:
    """Split a shell command on ``&&``, ``||``, ``;``, ``|``, ``&``, and newlines.

    Each sub-command is evaluated independently by the guard so that chaining
    (e.g. ``cd /path && git push origin dev``) cannot bypass pattern-based
    rules that are anchored with ``^``.

    The background operator ``&`` (single, distinct from ``&&``) is also
    treated as a separator so that ``git push origin dev &`` cannot evade
    rules with ``$`` or ``\\b`` anchors.
    """
    commands: list[str] = []
    current: list[str] = []
    in_single = False
    in_double = False
    escaped = False
    i = 0

    while i < len(command):
        ch = command[i]

        if escaped:
            current.append(ch)
            escaped = False
            i += 1
            continue

        if ch == "\\":
            escaped = True
            current.append(ch)
            i += 1
            continue

        if ch == "'" and not in_double:
            in_single = not in_single
            current.append(ch)
            i += 1
            continue

        if ch == '"' and not in_single:
            in_double = not in_double
            current.append(ch)
            i += 1
            continue

        if in_single or in_double:
            current.append(ch)
            i += 1
            continue

        if ch in ("\n", ";"):
            part = "".join(current).strip()
            if part:
                commands.append(part)
            current = []
            i += 1
            continue

        if ch == "&" and i + 1 < len(command) and command[i + 1] == "&":
            part = "".join(current).strip()
            if part:
                commands.append(part)
            current = []
            i += 2
            continue

        if ch == "&":
            part = "".join(current).strip()
            if part:
                commands.append(part)
            current = []
            i += 1
            continue

        if ch == "|" and i + 1 < len(command) and command[i + 1] == "|":
            part = "".join(current).strip()
            if part:
                commands.append(part)
            current = []
            i += 2
            continue

        if ch == "|":
            part = "".join(current).strip()
            if part:
                commands.append(part)
            current = []
            i += 1
            continue

        current.append(ch)
        i += 1

    part = "".join(current).strip()
    if part:
        commands.append(part)

    return commands


def normalize_for_patterns(command: str) -> str:
    """Normalize a command for YAML pattern matching.

    Strips ``git -C <path>`` so that rules anchored with ``^git\\s+push``
    still match when the caller passes ``git -C /some/path push ...``.
    """
    if not command.startswith("git "):
        return command
    normalized = re.sub(r"\s+-C\s*[= ]?\S+", "", command)
    return re.sub(r"\s+", " ", normalized).strip()


def get_current_branch(directory: str | None = None) -> str | None:
    """Get current git branch."""
    try:
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=directory,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass
    return None


# ============================================================================
# Rule evaluation
# ============================================================================


def evaluate_rules(
    command: str,
    config: dict[str, Any],
    cwd: str | None = None,
) -> tuple[str | None, str, str]:
    """Evaluate command against guard rules.

    Ordering:
    1. Optional branch enforcement (gated on ``settings.enforce_branch``).
    2. Python allowlist (code logic).
    3. Pattern-based rules from YAML (first match wins).

    Args:
        cwd: Effective working directory (e.g. from a preceding ``cd`` in a chain).

    Returns: (mapped_command, block_reason, decision)
    """
    del cwd  # Reserved for future per-subcommand CWD tracking.
    settings = config.get("settings", {})

    # --- Optional: branch enforcement ---
    enforce_branch = settings.get("enforce_branch")
    if enforce_branch and command.startswith("git "):
        git_target_dir = extract_git_directory(command)
        current_branch = get_current_branch(git_target_dir)
        if current_branch and current_branch != enforce_branch:
            # Allow recovery commands
            if re.match(
                rf"^git\s+(-C\s*\S+\s+)?checkout\s+(-b\s+)?{re.escape(enforce_branch)}\b",
                command,
            ) or re.match(r"^git\s+(-C\s*\S+\s+)?merge\s+--abort\b", command):
                pass
            else:
                return (
                    None,
                    f"FORBIDDEN: Currently on '{current_branch}'. "
                    f"Switch to {enforce_branch}: git checkout {enforce_branch}",
                    "deny",
                )

    # --- Python allowlist (complex logic) ---
    if re.match(r"^python3?\s+", command):
        if re.match(r"^python3?\s+scripts/", command):
            script_path_match = re.match(r"^python3?\s+(\S+)", command)
            if script_path_match:
                from pathlib import Path as _Path

                script_path = _Path(script_path_match.group(1))
                try:
                    resolved = script_path.resolve()
                    scripts_dir = _Path.cwd() / "scripts"
                    resolved.relative_to(scripts_dir.resolve())
                except (ValueError, OSError):
                    return (
                        None,
                        "Path traversal detected: script path escapes scripts/ directory",
                        "deny",
                    )
            return None, "", "allow"
        if re.match(
            r"^python3?\s+-m\s+(pytest|pip|venv|http\.server|py_compile|json\.tool)\b",
            command,
        ):
            return None, "", "allow"
        # Unknown python invocation — defer to pattern rules (may be caught there)

    # --- Pattern-based rules from YAML ---
    pattern_command = normalize_for_patterns(command)
    rules = config.get("rules", [])
    for rule in rules:
        pattern = rule.get("pattern", "")
        if not pattern:
            continue

        decision = rule.get("decision", "deny")
        use_search = rule.get("search", False)

        # Allow rules: if matched, skip the command (not blocked)
        if decision == "allow":
            if use_search:
                if re.search(pattern, pattern_command):
                    return None, "", "allow"
            elif re.match(pattern, pattern_command):
                return None, "", "allow"
            continue

        # Two-condition rules
        extra_search = rule.get("extra_search")
        if extra_search:
            first_match = (
                re.search(pattern, pattern_command)
                if use_search
                else re.match(pattern, pattern_command)
            )
            if first_match and re.search(extra_search, pattern_command, re.IGNORECASE):
                reason = rule.get("reason", "Blocked by guard")
                mapped_cmd = rule.get("mapped_cmd")
                return mapped_cmd, reason, decision
            continue

        # Standard single-pattern match
        matched = (
            re.search(pattern, pattern_command)
            if use_search
            else re.match(pattern, pattern_command)
        )
        if matched:
            reason = rule.get("reason", "Blocked by guard")
            mapped_cmd = rule.get("mapped_cmd")

            if mapped_cmd and rule.get("detect_env"):
                env = detect_environment(command)
                mapped_cmd = mapped_cmd.replace("{env}", env)
            if mapped_cmd and "<PR_NUMBER>" in mapped_cmd:
                pr_num = extract_pr_number(command)
                if pr_num:
                    mapped_cmd = mapped_cmd.replace("<PR_NUMBER>", pr_num)

            return mapped_cmd, reason, decision

    # Not blocked
    return None, "", "allow"


# ============================================================================
# Guard entry point (called by ``autosre hooks-backend guard``)
# ============================================================================


@click.command("guard")
@click.option(
    "--event",
    default="",
    help="Hook event name (unused, kept for compatibility with upstream wrapper).",
)
def guard_cmd(event: str) -> None:
    """Evaluate a Bash command against guard rules.

    Reads JSON from stdin (``tool_name``, ``tool_input``) and outputs a JSON
    decision (allow/deny/ask). Called by the installed wrapper hook at
    ``autosre/claude_hooks/pretooluse_bash_guard.py``.
    """
    del event
    command = ""
    try:
        config = load_config()

        # Block bypass environment variables
        blocked_vars = config.get("blocked_env_vars", [])
        for var in blocked_vars:
            if os.environ.get(var):
                log_block("BACKDOOR_ATTEMPT", f"Bypass via {var}", None, config)
                _output_deny(
                    f"SECURITY VIOLATION: Attempted to bypass guard via "
                    f"{var}={os.environ.get(var)}. Guard bypass is FORBIDDEN.",
                )
                return

        input_data = json.load(sys.stdin)
        tool_name = input_data.get("tool_name", "")
        tool_input = input_data.get("tool_input", {})

        # Guard Edit/Write on .gitignore: never add secrets/ or *.age patterns
        if tool_name in ("Edit", "Write"):
            file_path = tool_input.get("file_path", "")
            if file_path.endswith(".gitignore"):
                new_content = tool_input.get("new_string", "") or tool_input.get("content", "")
                if re.search(r"(?m)^[^#]*secrets", new_content):
                    _output_deny(
                        "FORBIDDEN: Do not gitignore the secrets/ directory. "
                        "Age-encrypted files (*.age) and recipients.txt are the "
                        "source of truth for secret management and MUST be committed to git.",
                    )
                    return

            # Guard Edit/Write on recipe/stage_config YAML files
            from autosre.hooks_backend import recipe_guard

            if recipe_guard.is_protected_recipe(file_path):
                from pathlib import Path as _Path

                recipe_path = _Path(file_path)
                file_exists = recipe_path.exists()

                try:
                    before = recipe_path.read_text()
                except (FileNotFoundError, OSError):
                    before = ""

                # A Write to a path that doesn't exist is a create.  New recipes
                # have no prior baseline to compare against — Phase 2 bench
                # runs mint the first baseline for a new model, so gating
                # creation on an approval token is circular.  Allow the create;
                # edits to existing recipes still require the token.
                if tool_name == "Write" and not file_exists:
                    _output_allow()
                    return

                if tool_name == "Write":
                    after = tool_input.get("content", "")
                else:  # Edit
                    old_s = tool_input.get("old_string", "")
                    new_s = tool_input.get("new_string", "")
                    after = before.replace(old_s, new_s, 1)

                changed = recipe_guard.diff_perf_values(before, after)
                if changed and not recipe_guard.has_perf_approval(file_path, after):
                    changed_str = ", ".join(changed)
                    _output_deny(
                        f"FORBIDDEN: Recipe edit changes performance-sensitive "
                        f"parameter(s): {changed_str}.\n\n"
                        "These require perf validation:\n"
                        "  1. Ask the user to make the change manually\n"
                        "  2. Run: autosre perf run\n"
                        "  3. If clean: autosre perf save-baseline <name>\n"
                        "  4. git add benchmarks/baselines/<name>.{json,md}\n"
                        "  5. Re-attempt the edit — the approval token will allow it\n",
                    )
                    return

            _output_allow()
            return

        # Only process Bash tool
        if tool_name != "Bash":
            _output_allow()
            return

        command = tool_input.get("command", "")
        if not command:
            _output_allow()
            return

        sub_commands = split_chained_commands(command)

        mapped_cmd: str | None = None
        reason = ""
        decision = "allow"
        chain_cwd: str | None = None

        for sub_cmd in sub_commands:
            if not sub_cmd:
                continue
            cd_match = re.match(r"^cd\s+(.+)$", sub_cmd)
            if cd_match:
                from pathlib import Path as _Path

                chain_cwd = str(_Path(cd_match.group(1).strip().strip("'\"")).expanduser())
            mapped_cmd, reason, decision = evaluate_rules(sub_cmd, config, cwd=chain_cwd)
            if decision != "allow":
                break

        if decision == "allow" or not reason:
            _output_allow()
        elif decision == "ask":
            if check_approval(command, config):
                _output_allow()
            else:
                cmd_hash = get_command_hash(command)
                message = f"APPROVAL_REQUIRED:{cmd_hash}:{reason}"
                log_block(command, reason, mapped_cmd, config)
                _output_deny(message)
        else:
            message = f"{reason}\n\nRun instead: {mapped_cmd}" if mapped_cmd else reason
            log_block(command, reason, mapped_cmd, config)
            _output_deny(message)

    except Exception as exc:
        # Log and fail CLOSED — guard exceptions must not bypass the deny.
        try:
            error_log = paths.hook_errors_log()
            error_log.parent.mkdir(parents=True, exist_ok=True)
            with error_log.open("a") as f:
                f.write(f"\n--- {datetime.now(UTC).isoformat()} ---\n")
                f.write(f"Command: {command or 'UNKNOWN'}\n")
                f.write(f"Exception: {exc}\n")
                traceback.print_exc(file=f)
        except OSError:
            pass
        _output_deny(f"Guard error (fail-closed): {exc}")


def _output_allow() -> None:
    """Output an allow decision."""
    print(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "allow",
                },
            },
        ),
    )


def _output_deny(reason: str) -> None:
    """Output a deny decision with reason."""
    print(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": reason,
                },
            },
        ),
    )
