"""Claude Code settings management — deny/allow built-in web tools."""

from __future__ import annotations

import json
from pathlib import Path

CLAUDE_USER_SETTINGS_PATH = Path.home() / ".claude" / "settings.json"
BUILTIN_WEB_TOOLS = ["WebFetch", "WebSearch"]


def get_claude_settings_path() -> Path:
    """Return the path to the project-level Claude Code settings file.

    WebFetch/WebSearch deny rules go here (not user-level) so they only
    apply to autosre local sessions, not the user's Anthropic-connected sessions.
    """
    # Walk up from CWD to find the auto-sre repo root (has pyproject.toml with name=autosre)
    # Fall back to the directory containing this file's package
    pkg_dir = Path(__file__).resolve().parent.parent.parent
    project_settings = pkg_dir / ".claude" / "settings.json"
    if project_settings.parent.exists() or (pkg_dir / "pyproject.toml").exists():
        return project_settings
    return CLAUDE_USER_SETTINGS_PATH


def load_claude_settings(path: Path | None = None) -> dict[str, object]:
    """Load Claude Code settings, returning empty dict if missing."""
    settings_path = path or get_claude_settings_path()
    if not settings_path.exists():
        return {}
    text = settings_path.read_text(encoding="utf-8").strip()
    if not text:
        return {}
    result: dict[str, object] = json.loads(text)
    return result


def save_claude_settings(settings: dict[str, object], path: Path | None = None) -> None:
    """Write Claude Code settings to disk."""
    settings_path = path or get_claude_settings_path()
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(
        json.dumps(settings, indent=2) + "\n",
        encoding="utf-8",
    )


def deny_builtin_web_tools(settings: dict[str, object] | None = None) -> dict[str, object]:
    """Add WebFetch and WebSearch to the deny list. Idempotent."""
    if settings is None:
        settings = load_claude_settings()

    permissions = settings.setdefault("permissions", {})
    if not isinstance(permissions, dict):
        permissions = {}
        settings["permissions"] = permissions

    deny = permissions.setdefault("deny", [])
    if not isinstance(deny, list):
        deny = []
        permissions["deny"] = deny

    for tool in BUILTIN_WEB_TOOLS:
        if tool not in deny:
            deny.append(tool)

    return settings


def allow_builtin_web_tools(settings: dict[str, object] | None = None) -> dict[str, object]:
    """Remove WebFetch and WebSearch from the deny list."""
    if settings is None:
        settings = load_claude_settings()

    permissions = settings.get("permissions")
    if not isinstance(permissions, dict):
        return settings

    deny = permissions.get("deny")
    if not isinstance(deny, list):
        return settings

    permissions["deny"] = [t for t in deny if t not in BUILTIN_WEB_TOOLS]

    return settings
