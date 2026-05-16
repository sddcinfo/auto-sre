"""LLM-as-judge wrapper around ``claude -p`` for finding comparison.

The judge is only used during the compare step (never during capture).
It takes two :class:`Finding` objects and returns a verdict of whether
they describe the same underlying issue: ``yes`` / ``partial`` / ``no``
plus a confidence score and a short rationale.

We invoke ``claude -p`` as a headless subprocess with a locked system
prompt and a JSON-shaped output contract. No tools, no agent teams, no
MCP servers — just a single pass. If the subprocess is missing, the
judge degrades gracefully to "no" so the differ can still run.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from autosre.eval.lenient_json import try_loads

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from autosre.eval.findings import Finding

JudgeVerdict = Literal["yes", "partial", "no"]

DEFAULT_JUDGE_MODEL = "claude-opus-4-6[1m]"


@dataclass
class JudgeResult:
    """One judge decision about one candidate pair."""

    same: JudgeVerdict
    confidence: float
    rationale: str
    raw: str = ""

    def score(self) -> float:
        """Map the verdict to a similarity score in ``[0, 1]``."""
        if self.same == "yes":
            return 1.0
        if self.same == "partial":
            return 0.6
        return 0.0


@dataclass
class JudgeConfig:
    model: str = DEFAULT_JUDGE_MODEL
    timeout_seconds: int = 120
    # The log file this judge instance appends to. Every call appends one
    # JSON row so compare runs are auditable after the fact.
    log_path: Path | None = None


class Judge:
    """Pluggable interface so tests can substitute a fake judge."""

    def __init__(self, config: JudgeConfig | None = None) -> None:
        self.config = config or JudgeConfig()
        self.calls: int = 0
        self._log_entries: list[dict[str, object]] = []

    def compare(self, a: Finding, b: Finding) -> JudgeResult:
        """Decide whether ``a`` and ``b`` describe the same issue.

        Subprocess is invoked as ``claude -p <prompt> --output-format=json``
        so we get a structured reply. On any failure path (missing binary,
        non-zero exit, unparseable output) we return ``same="no"``, which
        is the safe default for the differ (no spurious matches).
        """
        self.calls += 1

        prompt = _build_prompt(a, b)

        if shutil.which("claude") is None:
            result = JudgeResult(
                same="no",
                confidence=0.0,
                rationale="claude CLI not available; judge degraded to 'no'",
                raw="",
            )
            self._append_log(a, b, result)
            return result

        cmd = [
            "claude",
            "-p",
            prompt,
            "--model",
            self.config.model,
            "--output-format",
            "json",
        ]
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.timeout_seconds,
                check=False,
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            result = JudgeResult(
                same="no",
                confidence=0.0,
                rationale=f"judge subprocess failed: {exc}",
                raw="",
            )
            self._append_log(a, b, result)
            return result

        result = _parse_verdict(proc.stdout)
        result.raw = proc.stdout
        self._append_log(a, b, result)
        return result

    # ── Internals ──────────────────────────────────────────────────

    def _append_log(
        self,
        a: Finding,
        b: Finding,
        result: JudgeResult,
    ) -> None:
        entry = {
            "a_id": a.id,
            "b_id": b.id,
            "a_title": a.title,
            "b_title": b.title,
            "same": result.same,
            "confidence": result.confidence,
            "rationale": result.rationale,
        }
        self._log_entries.append(entry)
        if self.config.log_path is not None:
            self.config.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.config.log_path.open("a") as f:
                f.write(json.dumps(entry) + "\n")


def _build_prompt(a: Finding, b: Finding) -> str:
    return (
        "You are grading whether two code-review findings describe the "
        "same underlying issue. Return STRICT JSON with fields "
        '{"same": "yes"|"partial"|"no", "confidence": 0..1, "rationale": "..."}. '
        "Do not include any other text.\n\n"
        "Finding A:\n"
        f"  file: {a.file}\n"
        f"  line: {a.line}\n"
        f"  category: {a.category}\n"
        f"  title: {a.title}\n"
        f"  description: {a.description}\n\n"
        "Finding B:\n"
        f"  file: {b.file}\n"
        f"  line: {b.line}\n"
        f"  category: {b.category}\n"
        f"  title: {b.title}\n"
        f"  description: {b.description}\n"
    )


def _parse_verdict(raw: str) -> JudgeResult:
    data = try_loads(raw)
    if not isinstance(data, dict):
        return JudgeResult(
            same="no",
            confidence=0.0,
            rationale="judge output did not parse as JSON",
        )
    same_raw = str(data.get("same") or "no").lower()
    if same_raw == "yes":
        same: JudgeVerdict = "yes"
    elif same_raw == "partial":
        same = "partial"
    else:
        same = "no"
    conf_raw = data.get("confidence")
    try:
        conf_f = max(0.0, min(1.0, float(conf_raw)))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        conf_f = 0.0
    rationale = str(data.get("rationale") or "")
    return JudgeResult(same=same, confidence=conf_f, rationale=rationale)


class StubJudge(Judge):
    """Deterministic judge used by tests. Does not invoke any subprocess."""

    def __init__(
        self,
        verdict_fn: Callable[[Finding, Finding], JudgeResult] | None = None,
    ) -> None:
        super().__init__()
        self.verdict_fn = verdict_fn

    def compare(self, a: Finding, b: Finding) -> JudgeResult:
        self.calls += 1
        if self.verdict_fn is None:
            # Default: "yes" iff ids match.
            if a.id == b.id and a.id:
                return JudgeResult(same="yes", confidence=1.0, rationale="ids match")
            return JudgeResult(same="no", confidence=0.0, rationale="stub default")
        return self.verdict_fn(a, b)
