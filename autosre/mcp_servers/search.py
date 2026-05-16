"""Local web search MCP server — searches via DuckDuckGo, no API key needed."""

from __future__ import annotations

import os

from ddgs import DDGS
from mcp.server.fastmcp import FastMCP

MAX_RESULTS = int(os.environ.get("AUTOSRE_SEARCH_MAX_RESULTS", "10"))

server = FastMCP("autosre-search")


def _search(query: str, max_results: int) -> str:
    """Perform a DuckDuckGo search and format results as markdown."""
    capped = min(max_results, MAX_RESULTS)
    results = DDGS().text(query, max_results=capped)

    if not results:
        return f"No results found for: {query}"

    parts: list[str] = []
    for r in results:
        title = r.get("title", "Untitled")
        href = r.get("href", "")
        body = r.get("body", "")
        parts.append(f"### [{title}]({href})\n\n{body}")

    return "\n\n---\n\n".join(parts)


@server.tool()
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web using DuckDuckGo.

    Returns formatted markdown results with title, URL, and snippet
    for each result. No API key required.
    """
    try:
        return _search(query, max_results)
    except Exception as e:
        return f"Error searching for '{query}': {e}"


def main() -> None:
    """Entry point for autosre-mcp-search."""
    server.run()


if __name__ == "__main__":
    main()
