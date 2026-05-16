"""PreToolUse(ExitPlanMode) hook — runs the plan-review loop.

When the user exits plan mode, this hook resolves the plan file (via the
shared :mod:`._io` fallback chain so schema drift in Claude Code doesn't
silently bypass review), shells out to ``autosre review plan <path>
--json-output``, and translates the result into a Claude Code hook
decision:

- Blocking findings (exit 1): ``permissionDecision=deny`` with the
  findings + recommendations + any questions as ``additionalContext``.
  Claude re-enters plan mode, updates the plan, and tries ExitPlanMode
  again — the loop closes on the next call to this hook.
- Clean / P2-only (exit 0): ``permissionDecision=allow``. P2 findings
  are attached as advisory context but don't block.
- Provider chain failure (exit 2): fail-open via :func:`_io.fail_open_pretooluse`
  so a broken reviewer can't trap the user (visible systemMessage in
  strict mode).
"""

from __future__ import annotations

import json
import subprocess
import sys
from typing import Any

from autosre.claude_hooks import _io

_REVIEW_TIMEOUT_SECONDS = 1260  # MAX_CHAIN_SECONDS (1200) + 60s buffer


def _format_findings(findings: list[dict[str, Any]], provider: str) -> str:
    lines = [f"Plan review findings (via {provider}):\n"]
    counts: dict[str, int] = {}

    for f in findings:
        sev = f.get("severity", "P2")
        title = f.get("title", "Untitled")
        desc = f.get("description", "")
        rec = f.get("recommendation", "")
        counts[sev] = counts.get(sev, 0) + 1
        lines.append(f"  {sev}: {title}")
        if desc:
            lines.append(f"    {desc}")
        if rec:
            lines.append(f"    Recommendation: {rec}")
        lines.append("")

    summary_parts = [f"{counts.get(s, 0)} {s}" for s in ("P0", "P1", "P2") if counts.get(s)]
    lines.insert(1, f"  ({', '.join(summary_parts)})\n")
    lines.append(
        "BLOCKING: Address ALL findings above (P0, P1, and P2) in the plan, "
        "then call ExitPlanMode again for re-review.",
    )
    return "\n".join(lines)


def _format_questions(questions: list[str], provider: str) -> str:
    lines = [f"Questions from {provider} (use AskUserQuestion to ask the user):\n"]
    for i, q in enumerate(questions, 1):
        lines.append(f"  {i}. {q}")
    lines.append("")
    lines.append(
        "ACTION REQUIRED: Use AskUserQuestion to ask the user these questions. "
        "Incorporate their answers into the plan before calling ExitPlanMode again.",
    )
    return "\n".join(lines)


def main() -> int:
    inv = _io.parse_stdin(__file__)
    _io.log(inv, "plan review hook triggered")

    plan_file = _io.resolve_plan_file(inv)
    if plan_file is None:
        return _io.fail_open_pretooluse(inv, "no plan path in any fallback")

    plan_path = str(plan_file)
    _io.log(inv, f"reviewing: {plan_path} ({plan_file.stat().st_size} bytes)")

    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "autosre.cli",
                "review",
                "plan",
                plan_path,
                "--json-output",
            ],
            capture_output=True,
            text=True,
            timeout=_REVIEW_TIMEOUT_SECONDS,
            check=False,
        )
    except FileNotFoundError:
        return _io.fail_open_pretooluse(inv, "autosre binary not invocable")
    except subprocess.TimeoutExpired:
        return _io.fail_open_pretooluse(
            inv,
            f"autosre review plan timed out ({_REVIEW_TIMEOUT_SECONDS}s)",
        )
    except Exception as exc:
        return _io.fail_open_pretooluse(inv, f"autosre review plan unexpected error: {exc}")

    _io.log(
        inv,
        f"autosre review plan: rc={result.returncode}, "
        f"stdout={len(result.stdout)} chars, stderr={len(result.stderr)} chars",
    )
    if result.stderr:
        _io.log(inv, f"stderr: {result.stderr[:500]}", level="warn")

    if result.returncode == 1:
        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError:
            return _io.fail_open_pretooluse(inv, "review output not valid JSON")

        findings = data.get("findings", [])
        questions = data.get("questions", [])
        provider = data.get("provider", "unknown")
        iteration = data.get("iteration", 1)
        _io.log(
            inv,
            f"BLOCKING: {provider} iteration {iteration}: "
            f"{len(findings)} findings (P0/P1), {len(questions)} questions",
        )

        parts: list[str] = []
        if findings:
            parts.append(_format_findings(findings, provider))
        if questions:
            parts.append(_format_questions(questions, provider))

        context = "\n\n".join(parts) if parts else "Review returned empty results."
        reason = (
            f"Plan review ({provider}, iteration {iteration}) found "
            f"blocking issues. Revise the plan."
        )
        return _io.emit_pretooluse_deny(reason=reason, additional_context=context)

    if result.returncode == 0:
        try:
            data = json.loads(result.stdout) if result.stdout else {}
        except json.JSONDecodeError:
            data = {}

        findings = data.get("findings", [])
        questions = data.get("questions", [])
        provider = data.get("provider", "")
        iteration = data.get("iteration", 1)
        p2_only = data.get("p2_only", False)

        if p2_only:
            _io.log(
                inv,
                f"P2-ONLY: {provider} iteration {iteration}: "
                f"{len(findings)} advisory findings — allowing",
            )
            parts = [_format_findings(findings, provider)]
            parts.append(
                "NOTE: These are P2 (advisory) findings only — no blocking "
                "P0/P1 issues. Consider addressing during implementation but "
                "you may proceed.",
            )
            if questions:
                parts.append(_format_questions(questions, provider))
            return _io.emit_pretooluse_allow(additional_context="\n\n".join(parts))

        msg = f"Plan review ({provider}): no issues found." if provider else "Plan review: clean."
        _io.log(inv, msg)
        return _io.emit_pretooluse_allow(additional_context=msg)

    # rc=2 (all providers failed) or unexpected rc — fail-open loud.
    return _io.fail_open_pretooluse(
        inv,
        f"all providers failed (rc={result.returncode})",
    )


if __name__ == "__main__":
    sys.exit(main())
