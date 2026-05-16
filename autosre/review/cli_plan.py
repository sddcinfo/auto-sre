"""CLI command for plan review with iteration tracking.

State files live under ``autosre.paths.review_state_dir()`` keyed as
``_state_<sha256(abs_path)[:12]>_<plan_stem>.json`` so that two repos each
containing a ``plan.md`` don't share iteration state or contaminate
cached-clean decisions.

Usage:
    autosre review plan /path/to/plan.md                    # Review with default chain
    autosre review plan /path/to/plan.md --chain local      # Local vLLM/Ollama
    autosre review plan /path/to/plan.md --json-output      # JSON for hook consumption
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sys
import time
from typing import TYPE_CHECKING, Any

import click

from autosre import paths
from autosre.review.chain import DEFAULT_CHAINS, ChainResult, run_chain

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State file — tracks iteration count + previous findings per plan
# ---------------------------------------------------------------------------


def _state_key(plan_path: Path) -> str:
    """Compute a collision-free state filename fragment for ``plan_path``.

    Upstream keys state by ``plan_path.stem`` alone, which means two
    different repos containing ``plan.md`` would share state and contaminate
    iteration counts + cached-clean decisions. We prefix with a 12-char hash
    of the absolute path so each plan gets its own namespace while keeping
    the stem visible for human debugging.
    """
    abs_path = str(plan_path.resolve())
    path_hash = hashlib.sha256(abs_path.encode()).hexdigest()[:12]
    return f"{path_hash}_{plan_path.stem}"


def _state_path(plan_path: Path) -> Path:
    return paths.review_state_dir() / f"_state_{_state_key(plan_path)}.json"


def _load_state(plan_path: Path) -> dict[str, Any]:
    sp = _state_path(plan_path)
    if sp.exists():
        try:
            result: dict[str, Any] = json.loads(sp.read_text())
            return result
        except (json.JSONDecodeError, OSError):
            pass
    return {
        "iteration": 0,
        "previous_findings": [],
        "plan_mtime": 0,
        "plan_hash": "",
        "last_reviewed_at": "",
        "last_review_status": "not_reviewed",
        "last_review_findings_count": 0,
    }


def _save_state(plan_path: Path, state: dict[str, Any]) -> None:
    try:
        _state_path(plan_path).parent.mkdir(parents=True, exist_ok=True)
        _state_path(plan_path).write_text(json.dumps(state, indent=2) + "\n")
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Plan metadata helpers
# ---------------------------------------------------------------------------


def _extract_title(content: str) -> str:
    """Extract plan title from first markdown heading."""
    match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
    return match.group(1).strip() if match else "Untitled"


def _compute_hash(content: str) -> str:
    """SHA-256 of plan content."""
    return hashlib.sha256(content.encode()).hexdigest()


def _detect_project(content: str) -> str | None:
    """Try to detect project directory from plan content."""
    match = re.search(r"\*\*Project\*\*:\s*`([^`]+)`", content)
    return match.group(1).strip() if match else None


# ---------------------------------------------------------------------------
# Prompt templates (ported verbatim from upstream)
# ---------------------------------------------------------------------------

INITIAL_REVIEW_PROMPT = """\
Review this implementation plan. Identify issues by severity:
- P0 (Critical): Will cause data loss, security vulnerability, or system outage
- P1 (High): Will cause incorrect behavior or significant technical debt
- P2 (Medium): Design improvements or missing edge cases

Focus on: completeness, correctness, dependency ordering, security, risk.
Only flag genuine issues with concrete recommendations.

If you need clarification on any aspect of the plan before you can fully review it,
include those as questions in your response.

PLAN:
{plan_content}

Respond as JSON: {{"findings": [{{"severity": "P0|P1|P2", "title": "...", "description": "...", "recommendation": "..."}}], "questions": ["question 1", "question 2"]}}
Note: "questions" array is optional — only include if you genuinely need clarification."""

RE_REVIEW_PROMPT = """\
This is review iteration {iteration} of an implementation plan. The previous review \
found {prev_count} issues that the author has attempted to address in the updated plan below.

Your task is focused:
1. Verify each previously flagged issue was adequately addressed in the updated plan
2. Check that the fixes did not introduce NEW issues (regressions)
3. Do NOT raise entirely new concerns unrelated to the previous findings — \
the goal is convergence, not expanding scope

Previous findings that should now be resolved:
{previous_findings_text}

UPDATED PLAN:
{plan_content}

For each previous finding, check if it was fixed. If a fix introduced a regression, flag it.
Only flag genuinely NEW P0/P1 issues if they are critical and directly caused by the plan changes.

Respond as JSON: {{"findings": [{{"severity": "P0|P1|P2", "title": "...", "description": "...", "recommendation": "..."}}], "questions": ["question 1", "question 2"]}}
Note: "questions" array is optional."""


# ---------------------------------------------------------------------------
# CLI command
# ---------------------------------------------------------------------------


@click.command(name="plan")
@click.argument("plan_path", type=click.Path(exists=True, path_type=None))
@click.option(
    "--chain",
    "chain_str",
    default=None,
    help=f"Comma-separated provider chain. Default: {','.join(DEFAULT_CHAINS['plan'])}",
)
@click.option(
    "--json-output",
    "json_output",
    is_flag=True,
    help="Output raw JSON (for hook consumption). Exit code 1 if blocking findings.",
)
@click.option(
    "--reset",
    is_flag=True,
    help="Reset iteration counter (start fresh review).",
)
def plan_review_cmd(
    plan_path: str,
    chain_str: str | None,
    json_output: bool,
    reset: bool,
) -> None:
    """Review an implementation plan using AI providers.

    Tracks iterations per plan. After cycle 2, uses a focused re-review prompt
    that verifies previous findings were addressed without scope creep.

    P0/P1 findings block. P2-only findings are advisory (non-blocking).

    \b
    Examples:
      autosre review plan ~/.claude/plans/my-plan.md
      autosre review plan plan.md --chain local
      autosre review plan plan.md --json-output
      autosre review plan plan.md --reset
    """
    from pathlib import Path

    plan_path_obj = Path(plan_path)
    chain = chain_str.split(",") if chain_str else None

    plan_content = plan_path_obj.read_text()
    if len(plan_content.strip()) < 100:
        if json_output:
            json.dump({"findings": [], "provider": "", "error": "plan too short"}, sys.stdout)
            print()
        else:
            click.secho("Plan too short (<100 chars) — skipping review", fg="yellow")
        return

    # Extract plan metadata
    plan_title = _extract_title(plan_content)
    plan_hash = _compute_hash(plan_content)
    project = _detect_project(plan_content)
    plan_mtime = plan_path_obj.stat().st_mtime

    # Load iteration state
    state = _load_state(plan_path_obj)

    if reset or state.get("plan_mtime", 0) == 0:
        state = {
            "iteration": 0,
            "previous_findings": [],
            "plan_mtime": plan_mtime,
            "plan_hash": plan_hash,
            "last_reviewed_at": "",
            "last_review_status": "not_reviewed",
            "last_review_findings_count": 0,
        }

    state["iteration"] += 1
    iteration = state["iteration"]
    previous_findings = state.get("previous_findings", [])

    # Build metadata for logging
    metadata = {
        "plan_path": str(plan_path_obj),
        "plan_filename": plan_path_obj.name,
        "plan_title": plan_title,
        "iteration": iteration,
        "project": project,
        "plan_hash": plan_hash,
    }

    # Choose prompt based on iteration
    if iteration <= 2 or not previous_findings:
        prompt = INITIAL_REVIEW_PROMPT.format(plan_content=plan_content)
    else:
        prev_lines = []
        for i, f in enumerate(previous_findings, 1):
            sev = f.get("severity", "P2")
            title = f.get("title", "Untitled")
            desc = f.get("description", "")
            prev_lines.append(f"  {i}. [{sev}] {title}: {desc[:200]}")

        prompt = RE_REVIEW_PROMPT.format(
            iteration=iteration,
            prev_count=len(previous_findings),
            previous_findings_text="\n".join(prev_lines),
            plan_content=plan_content,
        )

    logger.info(
        "plan_review_start: plan=%r plan_size=%d iteration=%d chain=%r mode=%s",
        str(plan_path_obj),
        len(plan_content),
        iteration,
        chain,
        "re-review" if iteration > 2 else "initial",
    )

    result: ChainResult = run_chain(prompt, chain=chain, mode="plan", metadata=metadata)

    # Classify findings by severity
    findings = result.findings or []
    p0_p1 = [f for f in findings if f.get("severity") in ("P0", "P1")]
    p2_only = len(p0_p1) == 0 and len(findings) > 0
    has_blocking = len(p0_p1) > 0

    # Update and save state
    now = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    state["previous_findings"] = findings
    state["plan_mtime"] = plan_mtime
    state["plan_hash"] = plan_hash
    state["last_reviewed_at"] = now
    state["last_review_findings_count"] = len(findings)
    if has_blocking:
        state["last_review_status"] = "blocking"
    elif p2_only:
        state["last_review_status"] = "p2_only"
    elif result.provider:
        state["last_review_status"] = "clean"
    else:
        state["last_review_status"] = "provider_failed"
    _save_state(plan_path_obj, state)

    if json_output:
        output = {
            "findings": findings,
            "questions": result.questions or [],
            "provider": result.provider,
            "elapsed_seconds": result.elapsed_seconds,
            "iteration": iteration,
            "blocking": has_blocking,
            "p2_only": p2_only,
            "attempts": [
                {
                    "provider": a.provider,
                    "success": a.success,
                    "elapsed_seconds": a.elapsed_seconds,
                    "error": a.error,
                    "findings_count": a.findings_count,
                }
                for a in result.attempts
            ],
        }
        json.dump(output, sys.stdout, indent=2)
        print()
        if has_blocking:
            sys.exit(1)
        return

    # Text console output
    if has_blocking:
        click.secho(
            f"Iteration {iteration}: {len(p0_p1)} blocking findings (P0/P1)",
            fg="red",
        )
        click.echo(result.format_findings())
        _print_attempt_summary(result)
        sys.exit(1)
    elif p2_only:
        click.secho(
            f"Iteration {iteration}: {len(findings)} advisory findings (P2 only — non-blocking)",
            fg="yellow",
        )
        click.echo(result.format_findings())
        _print_attempt_summary(result)
    elif result.provider:
        click.secho(
            f"Iteration {iteration}: Plan review ({result.provider}) — no issues found.",
            fg="green",
        )
        _print_attempt_summary(result)
    else:
        click.secho("All providers failed or unavailable.", fg="red")
        _print_attempt_summary(result)
        sys.exit(2)


def _print_attempt_summary(result: ChainResult) -> None:
    if not result.attempts:
        return
    click.echo("Provider attempts:")
    for a in result.attempts:
        status = click.style("OK", fg="green") if a.success else click.style("FAIL", fg="red")
        findings_str = f" ({a.findings_count} findings)" if a.findings_count else ""
        error_str = f" — {a.error}" if a.error else ""
        click.echo(
            f"  {status} {a.provider}: {a.elapsed_seconds:.1f}s{findings_str}{error_str}",
        )
