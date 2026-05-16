"""Markdown rendering for eval runs and compares.

Deliberately boring: just enough markdown to let a human scan a
``report.md`` or ``compare.md`` without opening JSON. The source of
truth is always the JSON next to the markdown — the markdown is a
courtesy.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from autosre.eval.differ import CompareResult
    from autosre.eval.runner import RunResult

# ── Single-run report ─────────────────────────────────────────────


def render_single(result: RunResult) -> str:
    """Render the per-run ``report.md`` for one capture run."""
    lines: list[str] = []
    lines.append(f"# Eval run: {result.run_id}")
    lines.append("")
    lines.append(f"- Provider: **{result.provider}**")
    lines.append(f"- Target: `{result.target_repo}`")
    lines.append(f"- SHA: `{result.target_sha or 'n/a'}`")
    lines.append(f"- Snapshot digest: `{result.snapshot_digest or 'n/a'}`")
    lines.append("")

    total_findings = sum(len(sr.findings) for sr in result.suites)
    total_failures = sum(
        1 for sr in result.suites for a in sr.report.agents if a.status == "failed"
    )
    lines.append(f"**Findings:** {total_findings}")
    lines.append(f"**Parse failures:** {total_failures}")
    lines.append("")

    lines.append("## Per-suite summary")
    lines.append("")
    lines.append("| Suite | Findings | Parse failures | Snapshot |")
    lines.append("|---|---:|---:|---|")
    for sr in result.suites:
        failures = sum(1 for a in sr.report.agents if a.status == "failed")
        lines.append(f"| {sr.suite} | {len(sr.findings)} | {failures} | {sr.snapshot.mode} |")
    lines.append("")

    # Per-suite detail.
    for sr in result.suites:
        lines.append(f"## {sr.suite}")
        lines.append("")
        if sr.report.agents:
            lines.append("### Extraction status per agent")
            lines.append("")
            lines.append("| Agent | Status | Findings | Source |")
            lines.append("|---|---|---:|---|")
            for a in sr.report.agents:
                lines.append(f"| {a.role} | {a.status} | {a.finding_count} | `{a.source}` |")
            lines.append("")
        if sr.findings:
            lines.append("### Findings")
            lines.append("")
            lines.append("| Severity | File:Line | Category | Title |")
            lines.append("|---|---|---|---|")
            for f in sr.findings:
                loc = f"`{f.file}`" + (f":{f.line}" if f.line else "")
                lines.append(f"| {f.severity} | {loc} | {f.category} | {f.title} |")
            lines.append("")
        else:
            lines.append("_No findings reported._")
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_report_md(result: RunResult) -> Path:
    path = result.run_dir / "report.md"
    path.write_text(render_single(result))
    return path


# ── Compare report ────────────────────────────────────────────────


def render_compare(result: CompareResult) -> str:
    """Render the ``compare.md`` for a compare between two runs."""
    lines: list[str] = []
    lines.append(f"# Compare: {result.run_a.run_id} vs {result.run_b.run_id}")
    lines.append("")

    payload = result.to_json()
    if not payload["valid_as_model_quality"]:
        lines.append("> ⚠ This compare is **not** valid as model-quality data.")
        for w in result.warnings:
            lines.append(f">   - {w}")
        lines.append("")

    lines.append(f"- Target: `{result.run_a.target_repo}`")
    lines.append(
        f"- A: provider=**{result.run_a.provider}** "
        f"model=`{result.run_a.model}` sha=`{result.run_a.target_sha}`"
    )
    lines.append(
        f"- B: provider=**{result.run_b.provider}** "
        f"model=`{result.run_b.model}` sha=`{result.run_b.target_sha}`"
    )
    if result.overrides.to_list():
        lines.append(f"- Overrides: {', '.join(result.overrides.to_list())}")
    lines.append("")

    lines.append("## Per-suite buckets")
    lines.append("")
    lines.append("| Suite | A | B | both | partial | A-only | B-only | agreement |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    per_suite_meta = payload["per_suite"]
    assert isinstance(per_suite_meta, dict)
    for suite, d in sorted(result.per_suite.items()):
        meta = per_suite_meta.get(suite) or {}
        assert isinstance(meta, dict)
        agreement = float(meta.get("agreement_rate", 0.0))
        lines.append(
            f"| {suite} | {d.a_count} | {d.b_count} | {len(d.both)} | "
            f"{len(d.partial)} | {len(d.a_only)} | {len(d.b_only)} | "
            f"{agreement:.2f} |"
        )
    lines.append("")

    for suite, d in sorted(result.per_suite.items()):
        lines.append(f"## {suite}")
        lines.append("")
        lines.append(
            f"|A| = {d.a_count}, |B| = {d.b_count}; "
            f"both = {len(d.both)}, partial = {len(d.partial)}, "
            f"A_only = {len(d.a_only)}, B_only = {len(d.b_only)}"
        )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_compare_md(result: CompareResult, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "compare.md"
    path.write_text(render_compare(result))
    return path


def load_report_md(run_dir: Path) -> str:
    """Convenience for ``autosre eval show`` — read the rendered report."""
    path = run_dir / "report.md"
    if path.exists():
        return path.read_text()
    return f"(no report.md found in {run_dir})"


def load_run_manifest(run_dir: Path) -> dict[str, object]:
    """Convenience for ``autosre eval list`` — load one manifest."""
    path = run_dir / "manifest.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}
