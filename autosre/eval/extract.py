"""Findings extraction pipeline with explicit fallback chain.

The agent contract is: each sub-agent writes its findings as JSON to a
pre-created file path inside the run directory via the Write tool, and
does not print findings in chat. This module implements the extractor
that enforces that contract — with graceful fallbacks for the real
world, where sometimes agents get sloppy.

Five fallback tiers, in strict order:

    1. primary               — file exists, parses cleanly as JSON
    2. recovered_json        — file exists, cleaned JSON parses
    3. recovered_truncated   — file exists, slice from first ``{`` to
                               last matching ``}`` parses
    4. recovered_chat        — file missing; scan ``turns.jsonl`` for a
                               JSON blob whose top level has ``findings``
    5. failed                — nothing recovered; Finding count is zero
                               and the failure is recorded explicitly

The per-agent status is written to ``parse_report.json`` alongside
``findings.jsonl`` so no failure is ever silent.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from autosre.eval.capture import TurnRecord
from autosre.eval.findings import Finding, normalize_findings
from autosre.eval.lenient_json import try_loads

if TYPE_CHECKING:
    from pathlib import Path

ExtractStatus = Literal[
    "ok",
    "recovered_json",
    "recovered_truncated",
    "recovered_chat",
    "failed",
]


@dataclass
class AgentExtraction:
    role: str
    status: ExtractStatus
    source: str
    finding_count: int
    reason: str = ""


@dataclass
class ExtractionReport:
    suite: str
    agents: list[AgentExtraction]

    def to_dict(self) -> dict[str, object]:
        return {
            "suite": self.suite,
            "agents": [
                {
                    "role": a.role,
                    "status": a.status,
                    "source": a.source,
                    "finding_count": a.finding_count,
                    "reason": a.reason,
                }
                for a in self.agents
            ],
        }

    @property
    def any_failed(self) -> bool:
        return any(a.status == "failed" for a in self.agents)


def extract_agent_findings(
    agent_file: Path,
    *,
    role: str,
    suite: str,
    provider: str,
    turns_path: Path | None = None,
) -> tuple[list[Finding], AgentExtraction]:
    """Extract findings for one agent using the tiered fallback chain."""
    # Tier 1: strict JSON from the expected file.
    if agent_file.exists():
        try:
            data = json.loads(agent_file.read_text())
            findings = _findings_from_payload(
                data,
                suite=suite,
                role=role,
                provider=provider,
            )
            return findings, AgentExtraction(
                role=role,
                status="ok",
                source=str(agent_file),
                finding_count=len(findings),
            )
        except json.JSONDecodeError:
            pass

        # Tier 2: lenient JSON cleaner.
        raw = agent_file.read_text()
        data = try_loads(raw)
        if data is not None:
            findings = _findings_from_payload(
                data,
                suite=suite,
                role=role,
                provider=provider,
            )
            return findings, AgentExtraction(
                role=role,
                status="recovered_json",
                source=str(agent_file),
                finding_count=len(findings),
            )

        # Tier 3: brace-slice fallback.
        sliced = _slice_braces(raw)
        if sliced is not None:
            data = try_loads(sliced)
            if data is not None:
                findings = _findings_from_payload(
                    data,
                    suite=suite,
                    role=role,
                    provider=provider,
                )
                return findings, AgentExtraction(
                    role=role,
                    status="recovered_truncated",
                    source=str(agent_file),
                    finding_count=len(findings),
                )

    # Tier 4: scrape the transcript for a JSON blob in the final
    # assistant messages.
    if turns_path is not None and turns_path.exists():
        chat_data, chat_source = _scrape_turns_for_findings(turns_path)
        if chat_data is not None:
            findings = _findings_from_payload(
                chat_data,
                suite=suite,
                role=role,
                provider=provider,
            )
            return findings, AgentExtraction(
                role=role,
                status="recovered_chat",
                source=chat_source,
                finding_count=len(findings),
            )

    # Tier 5: failed — no findings, but the failure is recorded loudly.
    return [], AgentExtraction(
        role=role,
        status="failed",
        source=str(agent_file),
        finding_count=0,
        reason=(
            "expected findings file missing or unparseable; no JSON "
            "blob found in recent transcript messages either"
        ),
    )


def _findings_from_payload(
    data: object,
    *,
    suite: str,
    role: str,
    provider: str,
) -> list[Finding]:
    """Normalize a parsed JSON payload into validated ``Finding`` rows.

    Accepts either the full contract shape ``{suite, agent, findings}``
    or a bare list ``[{...}, ...]`` — agents sometimes emit just the
    list.
    """
    if isinstance(data, dict):
        raw = data.get("findings")
        if raw is None and "suite" not in data and "agent" not in data:
            raw = [data]
    elif isinstance(data, list):
        raw = data
    else:
        raw = None

    if not isinstance(raw, list):
        return []

    return normalize_findings(
        raw,
        suite=suite,
        agent=role,
        provider=provider,  # type: ignore[arg-type]
    )


def _slice_braces(text: str) -> str | None:
    """Return the substring from the first ``{`` to its matching ``}``."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


_FENCE_JSON_RE = re.compile(
    r"```(?:json|jsonc|json5)?\s*(\{.*?\})\s*```",
    re.DOTALL,
)


def _scrape_turns_for_findings(turns_path: Path) -> tuple[object | None, str]:
    """Scan the last few assistant turns for a JSON blob with ``findings``."""
    lines = turns_path.read_text().splitlines()
    tail: list[TurnRecord] = []
    for raw_line in reversed(lines):
        stripped = raw_line.strip()
        if not stripped:
            continue
        try:
            tail.append(TurnRecord.model_validate_json(stripped))
        except Exception:
            continue
        if len(tail) >= 5:
            break

    for n, turn in enumerate(tail):
        text = turn.response_prefix or ""
        if "findings" not in text.lower():
            continue
        # Try fenced code block first.
        m = _FENCE_JSON_RE.search(text)
        candidate = m.group(1) if m else _slice_braces(text)
        if candidate is None:
            continue
        data = try_loads(candidate)
        if isinstance(data, (dict, list)):
            return data, f"{turns_path}#tail[{n}]"
    return None, ""


def write_findings_jsonl(
    findings: list[Finding],
    out_path: Path,
) -> None:
    """Append-style writer for one suite's merged findings.jsonl."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for finding in findings:
            f.write(finding.model_dump_json() + "\n")


def write_parse_report(report: ExtractionReport, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report.to_dict(), indent=2))
