"""Finding schema + deterministic id + normalization.

A ``Finding`` is the single unit of eval output. Every agent in every suite
produces a list of them, and the differ consumes those lists to compute
per-suite buckets. The schema is deliberately small and strict so that
two providers can produce findings that are comparable without manual
post-processing.

``Finding.id`` is a content-addressed hash so that two runs of the same
provider on the same input tree produce byte-identical ids for the same
underlying issue. That stability is what makes the differ's
"both" bucket trivial in the easy case (exact id match).
"""

from __future__ import annotations

import hashlib
import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator

Severity = Literal["info", "low", "medium", "high", "critical"]
Category = Literal[
    "security",
    "leakage",
    "quality",
    "duplication",
    "dead-code",
    "tech-debt",
    "coverage",
    "ui-consistency",
    "a11y",
    "i18n",
]
Provider = Literal["local", "anthropic"]


_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_MD_NOISE = re.compile(r"[`*_~]")
_WS = re.compile(r"\s+")


def _normalize_path(path: str) -> str:
    """Canonicalize a file path for stable comparison.

    Strips leading ``./``, collapses ``//``, normalizes backslashes to
    forward slashes, and strips a leading ``/`` so that findings always
    use repo-relative paths regardless of how the agent emitted them.
    Does not resolve symlinks — the snapshot layer has already pinned
    the tree.
    """
    p = path.replace("\\", "/")
    while "//" in p:
        p = p.replace("//", "/")
    while p.startswith("./"):
        p = p[2:]
    while p.startswith("/"):
        p = p[1:]
    return p


def _normalize_title(title: str) -> str:
    t = _ANSI.sub("", title)
    t = _MD_NOISE.sub("", t)
    t = _WS.sub(" ", t).strip()
    return t[:200]


def _token_set(text: str) -> frozenset[str]:
    t = _MD_NOISE.sub(" ", text.lower())
    return frozenset(tok for tok in _WS.split(t) if tok)


def compute_id(
    *,
    suite: str,
    category: str,
    file: str,
    line: int | None,
    title: str,
) -> str:
    """Deterministic hash used as ``Finding.id``.

    Uses a token *set* of the title (not the raw string) so trivial
    reordering or punctuation differences don't produce different ids.
    File path and line are canonicalized first.
    """
    title_tokens = sorted(_token_set(_normalize_title(title)))
    key = "|".join(
        [
            suite,
            category,
            _normalize_path(file),
            str(line) if line is not None else "",
            " ".join(title_tokens),
        ]
    )
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


class Finding(BaseModel):
    """One finding produced by one eval agent."""

    id: str = ""
    suite: str
    category: Category
    severity: Severity = "medium"
    file: str
    line: int | None = None
    line_end: int | None = None
    title: str = Field(..., max_length=200)
    description: str = ""
    evidence: str = ""
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    agent: str = ""
    provider: Provider = "local"

    @field_validator("file", mode="after")
    @classmethod
    def _canon_file(cls, v: str) -> str:
        return _normalize_path(v)

    @field_validator("title", mode="after")
    @classmethod
    def _canon_title(cls, v: str) -> str:
        return _normalize_title(v)

    def with_id(self) -> Finding:
        """Return a copy with ``id`` recomputed from the current fields."""
        new_id = compute_id(
            suite=self.suite,
            category=self.category,
            file=self.file,
            line=self.line,
            title=self.title,
        )
        return self.model_copy(update={"id": new_id})


def normalize_findings(
    raw: list[dict[str, object]],
    *,
    suite: str,
    agent: str,
    provider: Provider,
) -> list[Finding]:
    """Parse a list of agent-produced dicts into validated ``Finding`` objects.

    Missing fields are defaulted. Invalid rows are dropped silently at
    this layer — the caller (findings extraction pipeline) is responsible
    for reporting parse failures via ``parse_report.json``.
    """
    out: list[Finding] = []
    for row in raw:
        if not isinstance(row, dict):
            continue
        payload = dict(row)  # copy so we can mutate
        payload.setdefault("suite", suite)
        payload.setdefault("agent", agent)
        payload.setdefault("provider", provider)
        try:
            f = Finding.model_validate(payload).with_id()
        except Exception:  # pydantic.ValidationError + any unexpected
            continue
        out.append(f)
    return out
