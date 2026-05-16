# ruff: noqa: TC001, TC003
"""Render :class:`RunResult` + :class:`Violation` to stdout or markdown."""

from __future__ import annotations

from pathlib import Path

import click

from autosre.perf.baseline import Baseline, Violation
from autosre.perf.harness import PhaseResult, RunResult

_PHASE_ORDER = ["isolated", "contention", "saturation"]
_WORKLOAD_ORDER = ["translation", "coding"]


def _ordered_phases(run: RunResult) -> list[PhaseResult]:
    def _key(p: PhaseResult) -> tuple[int, int]:
        phase_idx = _PHASE_ORDER.index(p.phase) if p.phase in _PHASE_ORDER else len(_PHASE_ORDER)
        wl_idx = (
            _WORKLOAD_ORDER.index(p.workload)
            if p.workload in _WORKLOAD_ORDER
            else len(_WORKLOAD_ORDER)
        )
        return (phase_idx, wl_idx)

    return sorted(run.phases, key=_key)


def render_stdout(
    run: RunResult,
    baseline: Baseline | None,
    violations: list[Violation],
) -> None:
    """Print a compact click-colored report to stdout."""
    click.echo()
    click.secho("autosre perf — run summary", bold=True)
    click.echo(f"  timestamp:  {run.timestamp}")
    click.echo(f"  model:      {run.environment.get('model_id', '?')}")
    click.echo(f"  gpu:        {run.environment.get('gpu', '?')}")
    click.echo(f"  vllm args:  {run.environment.get('vllm_args', '?')[:100]}")
    click.echo(f"  autosre:    {run.environment.get('autosre_sha', '?')}")
    if baseline:
        click.echo(f"  baseline:   {baseline.name} ({baseline.timestamp})")
    else:
        click.secho("  baseline:   <none — not comparing>", fg="yellow")
    click.echo()

    header = "  Workload     │  Phase       │ Samples │ TTFT p50 │ TTFT p95 │  TPS p50 │ TPS agg │ Errors"
    rule = "  ─────────────┼──────────────┼─────────┼──────────┼──────────┼──────────┼─────────┼───────"
    click.echo(header)
    click.echo(rule)
    for phase in _ordered_phases(run):
        row = (
            f"  {phase.workload:12s} │ {phase.phase:12s} │ {phase.samples:7d} │ "
            f"{phase.ttft_p50_ms:7.0f}ms │ {phase.ttft_p95_ms:7.0f}ms │ "
            f"{phase.tps_p50:7.2f} │ {phase.tps_agg:6.2f} │ "
        )
        err_cell = f"{phase.errors:5d}"
        click.echo(row, nl=False)
        if phase.errors > 0:
            click.secho(err_cell, fg="red")
        else:
            click.echo(err_cell)

    click.echo()
    click.secho("scheduler counters (during contention)", bold=True)
    sch = run.scheduler
    click.echo(f"  preemptions_delta:       {sch.preemptions_delta}")
    click.echo(f"  requests_running_peak:   {sch.requests_running_peak}")
    click.echo(f"  requests_waiting_peak:   {sch.requests_waiting_peak}")
    click.echo(f"  kv_cache_pct_peak:       {sch.kv_cache_pct_peak:.3f}")
    click.echo(f"  queue_time_avg_ms:       {sch.queue_time_avg_ms}")
    click.echo(f"  prefix_cache_hit_pct:    {sch.prefix_cache_hit_pct}")

    if run.proxy_sanity is not None:
        click.echo()
        click.secho("proxy sanity (via :8011/v1/messages)", bold=True)
        ps = run.proxy_sanity
        color = "green" if ps.ok else "red"
        click.secho(f"  status: {'OK' if ps.ok else 'FAIL'}", fg=color)
        click.echo(
            f"  message_start={ps.message_start}  deltas={ps.content_delta_count}  message_stop={ps.message_stop}"
        )
        if ps.upstream_error:
            click.secho(f"  upstream_error: {ps.upstream_error}", fg="yellow")
        if ps.note:
            click.echo(f"  note: {ps.note}")
        click.echo(f"  elapsed: {ps.elapsed_ms:.0f}ms")

    if violations:
        click.echo()
        fails = [v for v in violations if v.severity == "fail"]
        warns = [v for v in violations if v.severity == "warn"]
        if fails:
            click.secho(f"violations: {len(fails)} FAIL, {len(warns)} warn", fg="red", bold=True)
        else:
            click.secho(f"violations: {len(warns)} warn", fg="yellow", bold=True)
        for v in violations:
            color = "red" if v.severity == "fail" else "yellow"
            click.secho(f"  {v.summary()}", fg=color)
    elif baseline is not None:
        click.echo()
        click.secho("violations: none — run is within tolerance", fg="green", bold=True)

    click.echo()


# ── Markdown writer ────────────────────────────────────────────


def _markdown_table(rows: list[list[str]], header: list[str]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def render_markdown(
    run: RunResult,
    baseline: Baseline | None,
    violations: list[Violation],
    *,
    title: str | None = None,
) -> str:
    parts: list[str] = []
    parts.append(f"# {title or 'autosre perf — run'} ({run.timestamp})")
    parts.append("")
    parts.append("## Environment")
    parts.append("")
    for k, v in run.environment.items():
        parts.append(f"- **{k}**: `{v}`")
    parts.append("")
    parts.append("## Config")
    parts.append("")
    parts.append("```json")
    parts.append("\n".join(f'  "{k}": {v!r},' for k, v in run.config.items()))
    parts.append("```")
    parts.append("")

    parts.append("## Per-workload results")
    parts.append("")
    header = [
        "Workload",
        "Phase",
        "Samples",
        "TTFT p50 (ms)",
        "TTFT p95 (ms)",
        "TTFT p99 (ms)",
        "TPS p50",
        "TPS agg",
        "Errors",
        "Wall (s)",
    ]
    rows: list[list[str]] = []
    for phase in _ordered_phases(run):
        rows.append(
            [
                phase.workload,
                phase.phase,
                str(phase.samples),
                f"{phase.ttft_p50_ms:.0f}",
                f"{phase.ttft_p95_ms:.0f}",
                f"{phase.ttft_p99_ms:.0f}",
                f"{phase.tps_p50:.2f}",
                f"{phase.tps_agg:.2f}",
                str(phase.errors),
                f"{phase.wall_seconds:.1f}",
            ]
        )
    parts.append(_markdown_table(rows, header))
    parts.append("")

    parts.append("## Scheduler counters (during contention)")
    parts.append("")
    sch = run.scheduler
    parts.append(f"- `preemptions_delta`: **{sch.preemptions_delta}**")
    parts.append(f"- `requests_running_peak`: {sch.requests_running_peak}")
    parts.append(f"- `requests_waiting_peak`: {sch.requests_waiting_peak}")
    parts.append(f"- `kv_cache_pct_peak`: {sch.kv_cache_pct_peak}")
    parts.append(f"- `queue_time_avg_ms`: {sch.queue_time_avg_ms}")
    parts.append(f"- `prefix_cache_hit_pct`: {sch.prefix_cache_hit_pct}")
    parts.append("")

    if run.proxy_sanity is not None:
        ps = run.proxy_sanity
        parts.append("## Proxy sanity (`:8011/v1/messages`)")
        parts.append("")
        parts.append(f"- status: **{'OK' if ps.ok else 'FAIL'}**")
        parts.append(f"- `message_start`: {ps.message_start}")
        parts.append(f"- `content_block_delta` count: {ps.content_delta_count}")
        parts.append(f"- `message_stop`: {ps.message_stop}")
        parts.append(f"- `upstream_error`: {ps.upstream_error}")
        parts.append(f"- elapsed: {ps.elapsed_ms} ms")
        if ps.note:
            parts.append(f"- note: {ps.note}")
        parts.append("")

    if baseline is not None:
        parts.append(f"## Comparison against baseline `{baseline.name}` ({baseline.timestamp})")
        parts.append("")
        if not violations:
            parts.append("Clean — all metrics within tolerance.")
        else:
            parts.append(
                f"**{sum(1 for v in violations if v.severity == 'fail')} fail / {sum(1 for v in violations if v.severity == 'warn')} warn**"
            )
            parts.append("")
            for v in violations:
                parts.append(f"- {v.summary()}")
        parts.append("")

    return "\n".join(parts) + "\n"


def write_markdown(
    run: RunResult,
    baseline: Baseline | None,
    violations: list[Violation],
    path: Path,
    *,
    title: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown(run, baseline, violations, title=title))
