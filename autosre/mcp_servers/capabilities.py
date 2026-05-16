"""MCP server for autosre capability discovery.

Exposes 3 tools for progressive command discovery:

1. ``list_modules()`` — Categorized overview of all autosre subcommands.
2. ``search_commands(query, module=None)`` — Fuzzy keyword search.
3. ``get_command(command_path)`` — Full details for a specific command.

The backend introspects the live ``autosre.cli.cli`` click group at
startup instead of reading a static JSON catalog — there's nothing to
regenerate, the catalog is always in sync with the installed package.
"""

from __future__ import annotations

from typing import Any

import click
from mcp.server.fastmcp import FastMCP

from autosre.cli import cli as autosre_click_group

mcp = FastMCP(
    "autosre-capabilities",
    instructions=(
        "autosre MCP server for command discovery. "
        "Use these tools to find the correct autosre command before guessing. "
        "Flow: list_modules() -> search_commands(query) -> get_command(path) -> execute via Bash."
    ),
)


# ---------------------------------------------------------------------------
# Catalog introspection — walks the click group tree
# ---------------------------------------------------------------------------


def _walk(
    cmd: click.Command,
    path: str,
) -> list[dict[str, Any]]:
    """Return a flat list of ``{path, name, help, is_group, ...}`` dicts."""
    entries: list[dict[str, Any]] = []
    entry: dict[str, Any] = {
        "path": path,
        "name": cmd.name or "",
        "help": (cmd.help or "").strip(),
        "short_help": (cmd.get_short_help_str() or "").strip(),
        "is_group": isinstance(cmd, click.Group),
        "params": [_param_dict(p) for p in cmd.params],
        "subcommands": [],
    }
    entries.append(entry)
    if isinstance(cmd, click.Group):
        for sub_name in sorted(cmd.commands):
            sub = cmd.commands[sub_name]
            sub_path = f"{path} {sub_name}".strip()
            entry["subcommands"].append(sub_name)
            entries.extend(_walk(sub, sub_path))
    return entries


def _param_dict(param: click.Parameter) -> dict[str, Any]:
    kind = "argument" if isinstance(param, click.Argument) else "option"
    result: dict[str, Any] = {
        "kind": kind,
        "name": param.name or "",
        "help": (getattr(param, "help", "") or "").strip(),
        "required": param.required,
        "default": None if param.default is None else str(param.default),
        "opts": list(getattr(param, "opts", [])),
    }
    if isinstance(param.type, click.Choice):
        result["choices"] = list(param.type.choices)
    return result


def _build_catalog() -> list[dict[str, Any]]:
    """Return a flat catalog of all autosre commands and subcommands.

    The root entry has ``path = ""`` and represents the top-level ``autosre``
    command itself. Every other entry has a space-separated path like
    ``review plan`` or ``hooks-backend guard``.
    """
    return _walk(autosre_click_group, path="")


# Cache the catalog at import time — it doesn't change for the life of the
# server process since it's derived from the live click group.
_CATALOG: list[dict[str, Any]] | None = None


def _get_catalog() -> list[dict[str, Any]]:
    global _CATALOG  # noqa: PLW0603
    if _CATALOG is None:
        _CATALOG = _build_catalog()
    return _CATALOG


# ---------------------------------------------------------------------------
# Search helpers
# ---------------------------------------------------------------------------


def _score(entry: dict[str, Any], tokens: list[str]) -> int:
    """Simple substring-based scoring. No synonyms, no alias layer."""
    if not tokens:
        return 0
    path_parts = entry["path"].lower().split()
    short = entry["short_help"].lower()
    help_text = entry["help"].lower()
    score = 0
    for token in tokens:
        if token in path_parts:
            score += 10
            continue
        if token in short:
            score += 5
            continue
        if token in help_text:
            score += 2
            continue
        # Partial/fuzzy match against path parts
        if any(token in p for p in path_parts):
            score += 3
    return score


def _format_search_result(entry: dict[str, Any]) -> str:
    opts = [p for p in entry["params"] if p["kind"] == "option"]
    group_tag = " (group)" if entry["is_group"] else ""
    return (
        f"  autosre {entry['path']}{group_tag}\n"
        f"    {entry['short_help']}\n"
        f"    {len(opts)} options, {len(entry['subcommands'])} subcommands"
    )


def _format_command_detail(entry: dict[str, Any]) -> str:
    lines: list[str] = [f"autosre {entry['path']}", f"  {entry['short_help']}", ""]

    args = [p for p in entry["params"] if p["kind"] == "argument"]
    opts = [p for p in entry["params"] if p["kind"] == "option"]

    if args:
        lines.append("Arguments:")
        for p in args:
            req = " (required)" if p["required"] else ""
            lines.append(f"  {p['name'].upper()}{req}  {p['help']}")
        lines.append("")

    if opts:
        lines.append("Options:")
        for p in opts:
            opt_str = ", ".join(p["opts"]) or f"--{p['name']}"
            default = p.get("default")
            default_str = f" [default: {default}]" if default not in (None, "None") else ""
            choices = p.get("choices")
            choices_str = f" [{', '.join(choices)}]" if choices else ""
            req_str = " (REQUIRED)" if p["required"] else ""
            lines.append(f"  {opt_str}{req_str}  {p['help']}{default_str}{choices_str}")
        lines.append("")

    if entry["is_group"] and entry["subcommands"]:
        lines.append("Subcommands:")
        for sub in entry["subcommands"]:
            lines.append(f"  {sub}")
        lines.append("")

    if entry["help"] and entry["help"] != entry["short_help"]:
        lines.append("Full help:")
        lines.append(entry["help"])

    return "\n".join(lines).rstrip()


# ---------------------------------------------------------------------------
# MCP tools
# ---------------------------------------------------------------------------


@mcp.tool()
def list_modules() -> str:
    """List top-level autosre subcommand groups.

    Use this first to get an overview of what autosre can do. Returns each
    top-level command with its short help and subcommand count.
    """
    catalog = _get_catalog()
    top_level = [e for e in catalog if e["path"] and " " not in e["path"]]
    if not top_level:
        return "No commands found in autosre CLI."

    lines: list[str] = [f"autosre commands ({len(top_level)} top-level groups):\n"]
    for entry in sorted(top_level, key=lambda e: e["path"]):
        group_tag = " (group)" if entry["is_group"] else ""
        subs = f" ({len(entry['subcommands'])} subcommands)" if entry["subcommands"] else ""
        lines.append(f"  {entry['path']:<20} {entry['short_help']}{group_tag}{subs}")
    return "\n".join(lines)


@mcp.tool()
def search_commands(query: str, module: str | None = None) -> str:
    """Search autosre commands by keyword.

    Args:
        query: Keywords to search for (e.g. "review plan", "hooks guard",
            "vllm start"). Tokenized and matched against command paths, short
            help, and full help text.
        module: Optional top-level command name to restrict the search
            (e.g. ``module="review"`` only searches inside ``autosre review``).

    Returns up to 15 matching commands with their paths and descriptions.
    """
    catalog = _get_catalog()
    tokens = query.lower().split()
    if not tokens:
        return "No search terms provided."

    candidates = (
        catalog
        if module is None
        else [e for e in catalog if e["path"] == module or e["path"].startswith(f"{module} ")]
    )

    scored = [(e, _score(e, tokens)) for e in candidates if e["path"]]
    scored = [(e, s) for e, s in scored if s > 0]
    scored.sort(key=lambda t: -t[1])

    if not scored:
        return f"No commands matching '{query}'."

    header = f"Found {len(scored)} match(es) for '{query}':"
    lines = [header, ""]
    for entry, _score_val in scored[:15]:
        lines.append(_format_search_result(entry))
    return "\n".join(lines)


@mcp.tool()
def get_command(command_path: str) -> str:
    """Get full details for a specific autosre command.

    Args:
        command_path: The command path, e.g. ``review plan`` or
            ``hooks-backend guard``. The ``autosre`` prefix is optional —
            both ``autosre review plan`` and ``review plan`` work.

    Returns arguments, options with defaults and choices, subcommands (if
    it's a group), and the full help text.
    """
    catalog = _get_catalog()
    path = command_path.removeprefix("autosre ").strip()
    for entry in catalog:
        if entry["path"] == path:
            return _format_command_detail(entry)

    # Fallback: suggest close matches
    from difflib import get_close_matches

    all_paths = [e["path"] for e in catalog if e["path"]]
    matches = get_close_matches(path, all_paths, n=3, cutoff=0.5)
    if matches:
        lines = [f"Command '{path}' not found. Did you mean:"]
        for m in matches:
            lines.append(f"  autosre {m}")
        return "\n".join(lines)
    return f"Command '{path}' not found."


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the MCP server over stdio."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
