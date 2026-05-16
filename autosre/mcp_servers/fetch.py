"""Local web fetch MCP server — fetches URLs and converts HTML to markdown.

Uses curl_cffi to impersonate real browser TLS fingerprints, bypassing
bot detection (Cloudflare, Akamai, etc.) that blocks plain HTTP clients.
"""

from __future__ import annotations

import os
from typing import Any

from bs4 import BeautifulSoup
from curl_cffi import requests as curl_requests
from markdownify import markdownify
from mcp.server.fastmcp import FastMCP

MAX_LENGTH = int(os.environ.get("AUTOSRE_FETCH_MAX_LENGTH", "100000"))
BROWSER_IMPERSONATE: Any = "chrome131"
STRIP_TAGS = ["script", "style", "nav", "header", "footer", "aside"]

server = FastMCP("autosre-fetch")


def _fetch_and_convert(url: str) -> str:
    """Fetch a URL and convert its content to markdown."""
    response = curl_requests.get(
        url,
        impersonate=BROWSER_IMPERSONATE,
        timeout=30,
        allow_redirects=True,
    )
    response.raise_for_status()  # type: ignore[no-untyped-call]

    content_type = response.headers.get("content-type", "")

    # Non-HTML: return raw text
    if "text/html" not in content_type:
        text = response.text
        if len(text) > MAX_LENGTH:
            return text[:MAX_LENGTH] + "\n\n[Content truncated]"
        return text

    # HTML: strip junk tags, convert to markdown
    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup.find_all(STRIP_TAGS):
        tag.decompose()

    md: str = markdownify(str(soup), strip=["img"])
    # Collapse excessive blank lines
    while "\n\n\n" in md:
        md = md.replace("\n\n\n", "\n\n")
    md = md.strip()

    if len(md) > MAX_LENGTH:
        return md[:MAX_LENGTH] + "\n\n[Content truncated]"
    return md


@server.tool()
def web_fetch(url: str) -> str:
    """Fetch a URL and return its content as markdown.

    Fetches the given URL using browser TLS fingerprint impersonation
    to bypass bot detection. Strips non-content HTML elements and
    converts the result to clean markdown text.
    """
    try:
        return _fetch_and_convert(url)
    except curl_requests.errors.RequestsError as e:
        msg = str(e).lower()
        if "timeout" in msg:
            return f"Error: Request timed out fetching {url}"
        if "resolve" in msg or "connect" in msg:
            return f"Error: Could not connect to {url}"
        return f"Error fetching {url}: {e}"
    except Exception as e:
        return f"Error fetching {url}: {e}"


def main() -> None:
    """Entry point for autosre-mcp-fetch."""
    server.run()


if __name__ == "__main__":
    main()
