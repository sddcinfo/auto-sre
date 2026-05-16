"""Stop hook logic — session-end completion checklist.

Called via ``autosre hooks-backend stop-check`` from
``autosre/claude_hooks/stop_session_check.py``. Every git repo is treated
uniformly; ``_detect_deploy_cmd`` / ``_detect_test_cmd`` hint at generic
``wrangler``/``pytest``/``pnpm test`` patterns inferred from files present
in the repo.
"""

from __future__ import annotations

import json
import subprocess
import time
from typing import TYPE_CHECKING, Any

import click

if TYPE_CHECKING:
    from pathlib import Path


# ============================================================================
# Git helpers
# ============================================================================


def _git(*args: str, cwd: str | None = None) -> str | None:
    """Run a git command, return stdout or None on failure."""
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return None
    if result.returncode == 0:
        return result.stdout.strip()
    return None


def _detect_repo(cwd: str) -> tuple[str | None, Path | None]:
    """Detect the repo name and toplevel path from ``cwd``."""
    from pathlib import Path as _Path

    toplevel = _git("rev-parse", "--show-toplevel", cwd=cwd)
    if not toplevel:
        return None, None
    toplevel_path = _Path(toplevel)
    return toplevel_path.name, toplevel_path


def _uncommitted_count(cwd: str) -> int | None:
    """Return number of uncommitted changes, or None if can't determine."""
    status = _git("status", "--porcelain", cwd=cwd)
    if status is None:
        return None
    if not status:
        return 0
    return len(status.splitlines())


def _push_status(cwd: str) -> tuple[str | None, int | None, str | None]:
    """Return (branch, ahead_count, reason) or (branch, None, reason) on error."""
    branch = _git("rev-parse", "--abbrev-ref", "HEAD", cwd=cwd)
    if not branch:
        return None, None, None

    upstream = _git("rev-parse", "--abbrev-ref", f"{branch}@{{upstream}}", cwd=cwd)
    if not upstream:
        return branch, None, "no upstream"

    ahead = _git("rev-list", "--count", f"{upstream}..{branch}", cwd=cwd)
    if ahead is not None:
        return branch, int(ahead), None

    return branch, None, "could not determine"


# ============================================================================
# Repo type detection (dynamic, no hardcoded lists)
# ============================================================================


def _detect_repo_type(repo_path: Path) -> str:
    """Detect repo type from file markers."""
    # Check for Cloudflare Workers (wrangler.toml at root or in apps/)
    if (repo_path / "wrangler.toml").exists():
        return "cloudflare-worker"
    apps_dir = repo_path / "apps"
    if apps_dir.exists():
        for child in apps_dir.iterdir():
            if child.is_dir() and (child / "wrangler.toml").exists():
                return "cloudflare-worker"

    # Python package
    if (repo_path / "pyproject.toml").exists():
        return "python"

    # Node project (without wrangler = generic node)
    if (repo_path / "package.json").exists():
        return "node"

    return "unknown"


def _detect_deploy_hint(repo_path: Path, repo_type: str) -> str | None:
    """Return a generic deploy command hint based on repo type."""
    del repo_path
    if repo_type == "cloudflare-worker":
        return "wrangler deploy"
    return None


def _detect_test_hint(repo_path: Path, repo_type: str) -> str | None:
    """Return a generic test command hint based on repo type."""
    if (repo_path / "playwright.config.ts").exists() or (
        repo_path / "playwright.config.js"
    ).exists():
        return "pnpm playwright test"
    if repo_type == "python" and (repo_path / "tests").exists():
        return "pytest"
    if repo_type == "node" and (repo_path / "package.json").exists():
        return "pnpm test"
    return None


def _has_recent_local_plan(repo_path: Path) -> dict[str, str] | None:
    """Check for plan files modified within the last 30 minutes (project-local only).

    Returns an info item if found, None otherwise. Does NOT check
    ``~/.claude/plans/`` — that would create cross-repo false positives.
    """
    plans_dir = repo_path / ".claude" / "plans"
    if not plans_dir.exists():
        return None

    cutoff = time.time() - 1800  # 30 minutes

    for plan_file in plans_dir.glob("*.md"):
        try:
            if plan_file.stat().st_mtime > cutoff:
                return {
                    "status": "info",
                    "text": (
                        f"Recent plan file detected ({plan_file.name}) — "
                        "if still in plan mode, exit it before stopping."
                    ),
                }
        except OSError:
            continue

    return None


# ============================================================================
# Checklist builder
# ============================================================================


def build_checklist(repo_name: str, repo_path: Path) -> dict[str, Any]:
    """Build a completion checklist for the repo.

    Returns a dict with: repo_name, repo_type, items (list of {status, text}),
    and actions (list of suggested next steps).
    """
    repo_type = _detect_repo_type(repo_path)
    deploy_hint = _detect_deploy_hint(repo_path, repo_type)
    test_hint = _detect_test_hint(repo_path, repo_type)
    cwd = str(repo_path)

    items: list[dict[str, str]] = []
    actions: list[str] = []

    # 1. Committed?
    uncommitted = _uncommitted_count(cwd)
    if uncommitted is None:
        items.append({"status": "unknown", "text": "Could not check commit status"})
    elif uncommitted == 0:
        items.append({"status": "done", "text": "All changes committed"})
    else:
        items.append(
            {
                "status": "todo",
                "text": f"{uncommitted} uncommitted change(s) — commit first",
            },
        )
        actions.append("Commit outstanding changes")

    # 2. Pushed?
    branch, ahead, reason = _push_status(cwd)
    if branch is None:
        items.append({"status": "unknown", "text": "Could not check push status"})
    elif reason:
        items.append({"status": "todo", "text": f"Branch `{branch}` not pushed ({reason})"})
        actions.append(f"Push branch `{branch}` to remote")
    elif ahead and ahead > 0:
        items.append(
            {"status": "todo", "text": f"Branch `{branch}` is {ahead} commit(s) ahead of remote"},
        )
        actions.append(f"Push branch `{branch}` to remote")
    else:
        items.append({"status": "done", "text": f"Branch `{branch}` is up to date with remote"})

    # 3. Deploy hint (info only)
    if deploy_hint:
        items.append(
            {
                "status": "info",
                "text": f"Consider deploy: `{deploy_hint}`",
            },
        )

    # 4. Test hint (info only)
    if test_hint:
        items.append({"status": "info", "text": f"Run tests? (`{test_hint}`)"})
        actions.append(f"Run tests: {test_hint}")

    # 5. Commit frequency reminder (if large uncommitted count)
    if uncommitted and uncommitted > 5:
        items.append(
            {
                "status": "todo",
                "text": (
                    f"{uncommitted} uncommitted files — commit frequently to get "
                    "pre-commit checks on smaller changesets"
                ),
            },
        )
        actions.append("Break changes into logical commits for better review coverage")

    # 6. Recent local plan files (info only — may indicate unfinished plan mode)
    recent_plan = _has_recent_local_plan(repo_path)
    if recent_plan:
        items.append(recent_plan)

    return {
        "repo_name": repo_name,
        "repo_type": repo_type,
        "branch": branch,
        "items": items,
        "actions": actions,
    }


def format_checklist(data: dict[str, Any]) -> str:
    """Format checklist data into a human-readable message."""
    repo_name = data["repo_name"]
    repo_type = data["repo_type"]
    items = data["items"]
    actions = data["actions"]

    lines = []
    for item in items:
        if item["status"] == "done":
            lines.append(f"[x] {item['text']}")
        else:
            lines.append(f"[ ] {item['text']}")

    checklist = "\n".join(lines)

    blocking = [i for i in items if i["status"] == "todo"]

    if blocking:
        header = (
            f"STOP — {len(blocking)} blocking item(s) must be resolved before ending session.\n"
            f"Completion checklist for `{repo_name}` ({repo_type}):"
        )
    else:
        header = f"[session-end] Completion checklist for `{repo_name}` ({repo_type}):"

    message = f"{header}\n{checklist}\n"

    if actions:
        label = "REQUIRED before stopping" if blocking else "Suggested next steps"
        message += f"\n{label}:\n" + "\n".join(f"  - {a}" for a in actions)

    message += (
        "\n\nReminders:"
        "\n  - Commit frequently for smaller, better-reviewed changesets"
        "\n  - Address lint/type/review findings immediately — don't accumulate"
    )

    return message


# ============================================================================
# CLI entry point
# ============================================================================


@click.command("stop-check")
def stop_check_cmd() -> None:
    """Evaluate session-end completion checklist.

    Detects the current repo, its type, and what steps remain
    (commit, push, deploy, test). Output is JSON for the Claude Code
    Stop hook. Non-blocking by default — only produces ``result: block``
    when there are uncommitted changes or unpushed commits.
    """
    from pathlib import Path as _Path

    cwd = str(_Path.cwd())
    repo_name, repo_path = _detect_repo(cwd)

    if not repo_name or not repo_path:
        print(json.dumps({"result": "continue"}))
        return

    data = build_checklist(repo_name, repo_path)

    # If everything is done (no todo or info items), minimal message
    non_done = [i for i in data["items"] if i["status"] != "done"]
    if not non_done:
        print(
            json.dumps(
                {
                    "result": "continue",
                    "message": (
                        f"[session-end] {repo_name} ({data['repo_type']}): "
                        "All clear — nothing outstanding."
                    ),
                },
            ),
        )
        return

    message = format_checklist(data)
    blocking = [i for i in data["items"] if i["status"] == "todo"]
    output: dict[str, str] = {"result": "block" if blocking else "continue"}
    output["reason" if blocking else "message"] = message
    print(json.dumps(output))
