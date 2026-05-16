"""SubagentStart hook — injects project context into Plan/Explore subagents.

Reads CLAUDE.md, ``.claude/rules/*.md``, recent git state, and the
top-level directory listing; emits them as ``additionalContext`` within
a 12K character budget. Fail-open on any error.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from autosre.claude_hooks import _io

TOTAL_BUDGET = 12000


def _capped(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n... (truncated)"


def _read_claude_md(cwd: Path, max_chars: int = 2000) -> str:
    path = cwd / "CLAUDE.md"
    try:
        content = path.read_text()[:max_chars]
    except (OSError, UnicodeDecodeError):
        return "(no CLAUDE.md found)"
    if len(content) >= max_chars:
        content += "\n... (truncated)"
    return content


def _list_rules(cwd: Path) -> str:
    rules_dir = cwd / ".claude" / "rules"
    if not rules_dir.is_dir():
        return "none"

    lines: list[str] = []
    try:
        for entry in sorted(rules_dir.iterdir()):
            if not entry.name.endswith(".md"):
                continue
            try:
                with entry.open() as f:
                    first_line = f.readline().strip().lstrip("# ")
                lines.append(f"- {entry.name}: {first_line}")
            except (OSError, UnicodeDecodeError):
                lines.append(f"- {entry.name}")
    except OSError:
        return "none"
    return "\n".join(lines) if lines else "none"


def _git_context(cwd: Path) -> str:
    parts: list[str] = []
    try:
        branch = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if branch.returncode == 0 and branch.stdout.strip():
            parts.append(f"Branch: {branch.stdout.strip()}")
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass

    try:
        log = subprocess.run(
            ["git", "log", "--oneline", "-5"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if log.returncode == 0 and log.stdout.strip():
            parts.append(f"Last commits:\n{log.stdout.strip()}")
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass

    return " | ".join(parts) if parts else "(not a git repo)"


def _top_level_dirs(cwd: Path) -> str:
    try:
        entries = sorted(cwd.iterdir(), key=lambda p: p.name)
    except OSError:
        return "(unable to list)"

    dirs = [e.name for e in entries if e.is_dir() and not e.name.startswith(".")]
    files = [e.name for e in entries if e.is_file() and not e.name.startswith(".")]
    result: list[str] = []
    if dirs:
        result.append("Dirs: " + ", ".join(dirs[:20]))
    if files:
        result.append("Files: " + ", ".join(files[:15]))
    return "\n".join(result) if result else "(empty)"


def main() -> int:
    inv = _io.parse_stdin(__file__)

    try:
        cwd = inv.cwd if inv.cwd.is_dir() else Path.cwd()

        claude_md = _capped(_read_claude_md(cwd, 2000), 2000)
        rules = _capped(_list_rules(cwd), 500)
        git_state = _capped(_git_context(cwd), 500)
        structure = _capped(_top_level_dirs(cwd), 500)

        context_parts = [
            "=== PROJECT CONTEXT (auto-injected) ===",
            f"## CLAUDE.md (truncated)\n{claude_md}",
            f"## Rules\n{rules}",
            f"## Recent Activity\n{git_state}",
            f"## Structure\n{structure}",
            "=== END PROJECT CONTEXT ===",
        ]

        context = "\n\n".join(context_parts)
        if len(context) > TOTAL_BUDGET:
            context = context[:TOTAL_BUDGET] + "\n... (context truncated)"

        return _io.emit_subagent_context(context)
    except Exception as exc:
        return _io.fail_open_continue(inv, f"subagent-context error: {exc}")


if __name__ == "__main__":
    sys.exit(main())
