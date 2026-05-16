"""Tiny lenient-JSON cleaner used as fallback A in findings extraction.

This is intentionally NOT a JSON5 parser. It is ~20 lines of surgical
regex + string work that handles the three common failure modes we see
when agents emit "almost-valid" JSON:

1. Markdown code fences around the document (```json ... ```).
2. ``//`` line comments and ``/* ... */`` block comments.
3. Trailing commas before ``}`` / ``]``.

Anything else is left alone and the standard ``json.loads`` pass decides.
No new runtime dependency.
"""

from __future__ import annotations

import json
import re
from typing import Any

_FENCE_OPEN = re.compile(r"^\s*```(?:json|json5|jsonc)?\s*\n", re.IGNORECASE)
_FENCE_CLOSE = re.compile(r"\n\s*```\s*$")
_LINE_COMMENT = re.compile(r"(?m)(?<!:)//[^\n]*$")
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_TRAILING_COMMA = re.compile(r",(\s*[}\]])")


def clean(text: str) -> str:
    """Return a cleaned-up version of ``text`` more likely to parse as JSON."""
    text = text.strip()
    text = _FENCE_OPEN.sub("", text)
    text = _FENCE_CLOSE.sub("", text)
    text = _BLOCK_COMMENT.sub("", text)
    text = _LINE_COMMENT.sub("", text)
    text = _TRAILING_COMMA.sub(r"\1", text)
    return text.strip()


def loads(text: str) -> Any:
    """Parse ``text`` with the cleaner applied first.

    Raises ``json.JSONDecodeError`` on failure, same as ``json.loads``,
    so callers can catch the standard exception.
    """
    return json.loads(clean(text))


def try_loads(text: str) -> Any | None:
    """Same as :func:`loads` but returns ``None`` on failure instead of raising."""
    try:
        return loads(text)
    except json.JSONDecodeError:
        return None
