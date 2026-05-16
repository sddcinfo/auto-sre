# ruff: noqa: RUF001
"""Boot-time benchmark for the meeting-scribe systemd service.

Measures cold-start time from ``systemctl --user stop`` to all backends
healthy. Isolates the measurement from the compose watchdog timer and
scopes journal parsing to the exact systemd invocation.

See ``autosre perf boot --help`` for usage.
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import click

logger = logging.getLogger(__name__)

_SCRIBE_SERVICE = "meeting-scribe.service"
_WATCHDOG_TIMER = "scribe-compose-watchdog.timer"

# Journal message patterns emitted by the service during startup.
_COMPOSE_DONE_MARKER = "compose up: done"
_PREFLIGHT_DONE_MARKER = "[PASS] phase2 backends_healthy"
_SERVER_READY_MARKER = "Started server process"

# How long to wait for the service to become active before giving up.
_MAX_WAIT_SECONDS = 900  # 15 min — generous to cover worst-case cold boot


@dataclass
class BootResult:
    """Timing result for a single boot benchmark run."""

    timestamp: str
    total_seconds: float
    compose_up_seconds: float | None = None
    preflight_seconds: float | None = None
    server_ready_seconds: float | None = None
    all_backends_healthy: bool = False
    service_final_state: str = ""
    invocation_id: str = ""
    kind: str = "boot"

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> BootResult:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class BootBaseline:
    """Committed boot-time baseline."""

    name: str
    timestamp: str
    result: BootResult
    tolerances: dict[str, float] = field(
        default_factory=lambda: {
            "total_warn_ratio": 1.30,
            "total_fail_ratio": 2.00,
        }
    )
    kind: str = "boot"

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "timestamp": self.timestamp,
            "kind": self.kind,
            "result": self.result.to_json(),
            "tolerances": self.tolerances,
        }


@dataclass
class BootViolation:
    metric: str
    observed: float
    baseline: float
    limit_ratio: float
    severity: str  # "warn" or "fail"

    def summary(self) -> str:
        return (
            f"[{self.severity.upper()}] {self.metric}: "
            f"observed={self.observed:.1f}s, baseline={self.baseline:.1f}s, "
            f"limit: ≤{self.limit_ratio:.2f}× baseline"
        )


# ── Timer state preservation ──────────────────────────────────


def _systemctl_user(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["systemctl", "--user", *args],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


@dataclass
class _TimerState:
    enabled: bool
    active: bool


def _save_timer_state() -> _TimerState:
    enabled = _systemctl_user("is-enabled", _WATCHDOG_TIMER).returncode == 0
    active = _systemctl_user("is-active", _WATCHDOG_TIMER).returncode == 0
    return _TimerState(enabled=enabled, active=active)


def _restore_timer_state(state: _TimerState) -> None:
    if state.enabled and state.active:
        _systemctl_user("enable", "--now", _WATCHDOG_TIMER)
    elif state.enabled:
        _systemctl_user("enable", _WATCHDOG_TIMER)
    else:
        _systemctl_user("disable", _WATCHDOG_TIMER)


def _stop_timer() -> None:
    _systemctl_user("stop", _WATCHDOG_TIMER)


# ── Service control ───────────────────────────────────────────


def _stop_stack() -> None:
    """Stop meeting-scribe service and compose stack."""
    click.echo("Stopping meeting-scribe service...")
    _systemctl_user("stop", _SCRIBE_SERVICE)

    # Also bring down compose containers
    try:
        subprocess.run(
            ["meeting-scribe", "gb10", "down"],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except FileNotFoundError:
        click.secho("meeting-scribe CLI not found — skipping gb10 down", fg="yellow")


def _start_service() -> None:
    """Start the meeting-scribe systemd service (non-blocking).

    ``Type=notify`` services block ``systemctl start`` until READY=1,
    which can take minutes for a cold boot. Use ``--no-block`` so we
    can poll ``ActiveState`` ourselves with timing.
    """
    _systemctl_user("start", "--no-block", _SCRIBE_SERVICE)


def _get_invocation_id() -> str:
    result = _systemctl_user("show", _SCRIBE_SERVICE, "-p", "InvocationID", "--value")
    return result.stdout.strip()


def _get_active_state() -> str:
    result = _systemctl_user("show", _SCRIBE_SERVICE, "-p", "ActiveState", "--value")
    return result.stdout.strip()


# ── Health polling ────────────────────────────────────────────


async def _check_backends_healthy() -> bool:
    """Check all 4 backends once (non-waiting)."""
    try:
        from meeting_scribe.infra.health import check_all_services

        results = await check_all_services("localhost", wait=False)
        return all(s.healthy for s in results.values())
    except ImportError:
        # Fallback: probe the known ports directly
        import httpx

        ports = {"translation": 8010, "diarization": 8001, "tts": 8002, "asr": 8003}
        async with httpx.AsyncClient(timeout=3) as client:
            for port in ports.values():
                try:
                    resp = await client.get(f"http://localhost:{port}/health")
                    if resp.status_code != 200:
                        return False
                except (httpx.ConnectError, httpx.ReadTimeout, httpx.ConnectTimeout, OSError):
                    return False
        return True


# ── Journal parsing ───────────────────────────────────────────


def _parse_sub_phases(
    start_time: datetime,
    invocation_id: str,
) -> dict[str, float | None]:
    """Parse journal for sub-phase timestamps, scoped to invocation."""
    since_str = start_time.strftime("%Y-%m-%d %H:%M:%S")
    result = subprocess.run(
        [
            "journalctl",
            "--user",
            "-u",
            _SCRIBE_SERVICE,
            "--since",
            since_str,
            "-o",
            "json",
            "--no-pager",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    compose_up_ts: float | None = None
    preflight_ts: float | None = None
    server_ready_ts: float | None = None
    start_ts: float | None = None

    for line in result.stdout.splitlines():
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue

        # Filter by invocation ID when available
        if invocation_id and entry.get("_SYSTEMD_INVOCATION_ID", "") != invocation_id:
            continue

        msg = entry.get("MESSAGE", "")
        # __REALTIME_TIMESTAMP is microseconds since epoch
        ts_us = int(entry.get("__REALTIME_TIMESTAMP", 0))
        ts = ts_us / 1_000_000.0

        if start_ts is None:
            start_ts = ts

        if _COMPOSE_DONE_MARKER in msg and compose_up_ts is None:
            compose_up_ts = ts
        if _PREFLIGHT_DONE_MARKER in msg and preflight_ts is None:
            preflight_ts = ts
        if _SERVER_READY_MARKER in msg and server_ready_ts is None:
            server_ready_ts = ts

    phases: dict[str, float | None] = {
        "compose_up_seconds": None,
        "preflight_seconds": None,
        "server_ready_seconds": None,
    }
    if start_ts is not None:
        if compose_up_ts is not None:
            phases["compose_up_seconds"] = compose_up_ts - start_ts
        if preflight_ts is not None:
            phases["preflight_seconds"] = preflight_ts - start_ts
        if server_ready_ts is not None:
            phases["server_ready_seconds"] = server_ready_ts - start_ts

    return phases


# ── Main benchmark ────────────────────────────────────────────


def run_boot_benchmark() -> BootResult:
    """Run a cold-boot benchmark of the meeting-scribe stack.

    Stops the service and compose stack, starts the service, and measures
    time to all backends healthy.
    """
    timer_state = _save_timer_state()
    _stop_timer()
    click.echo(
        f"  watchdog timer paused (was: enabled={timer_state.enabled}, active={timer_state.active})"
    )

    try:
        _stop_stack()
        click.echo("  stack stopped — starting cold boot measurement")

        # Record start
        start_mono = time.monotonic()
        start_wall = datetime.now(tz=UTC)

        _start_service()
        click.echo("  systemctl start issued")

        # Wait for service to become active
        invocation_id = ""
        while time.monotonic() - start_mono < _MAX_WAIT_SECONDS:
            state = _get_active_state()
            if state == "active":
                invocation_id = _get_invocation_id()
                break
            if state == "failed":
                elapsed = time.monotonic() - start_mono
                click.secho(f"  service FAILED after {elapsed:.1f}s", fg="red")
                return BootResult(
                    timestamp=start_wall.strftime("%Y%m%dT%H%M%S"),
                    total_seconds=elapsed,
                    all_backends_healthy=False,
                    service_final_state="failed",
                    invocation_id=invocation_id,
                )
            time.sleep(2)

        # Service is active — now confirm all backends are healthy
        click.echo("  service active — checking backends...")
        backends_healthy = False
        while time.monotonic() - start_mono < _MAX_WAIT_SECONDS:
            backends_healthy = asyncio.run(_check_backends_healthy())
            if backends_healthy:
                break
            time.sleep(3)

        elapsed = time.monotonic() - start_mono
        state_str = "healthy" if backends_healthy else "backends_unhealthy"
        color = "green" if backends_healthy else "red"
        click.secho(f"  boot complete: {elapsed:.1f}s ({state_str})", fg=color)

        # Parse sub-phases from journal
        sub_phases = _parse_sub_phases(start_wall, invocation_id)

        return BootResult(
            timestamp=start_wall.strftime("%Y%m%dT%H%M%S"),
            total_seconds=elapsed,
            compose_up_seconds=sub_phases.get("compose_up_seconds"),
            preflight_seconds=sub_phases.get("preflight_seconds"),
            server_ready_seconds=sub_phases.get("server_ready_seconds"),
            all_backends_healthy=backends_healthy,
            service_final_state="active" if backends_healthy else _get_active_state(),
            invocation_id=invocation_id,
        )
    finally:
        _restore_timer_state(timer_state)
        click.echo(
            f"  watchdog timer restored (enabled={timer_state.enabled}, active={timer_state.active})"
        )


# ── Baseline I/O ──────────────────────────────────────────────


def boot_baselines_dir() -> Path:
    """Locate the committed baselines directory."""
    here = Path(__file__).resolve()
    repo_root = here.parents[2]
    return repo_root / "benchmarks" / "baselines"


def load_boot_baseline(name: str) -> BootBaseline:
    path = boot_baselines_dir() / f"boot_{name}.json"
    if not path.exists():
        msg = f"Boot baseline not found: {path}"
        raise FileNotFoundError(msg)
    raw = json.loads(path.read_text())
    if raw.get("kind", "boot") != "boot":
        msg = f"Baseline {path} has kind={raw.get('kind')!r}, expected 'boot'"
        raise ValueError(msg)
    return BootBaseline(
        name=raw["name"],
        timestamp=raw["timestamp"],
        result=BootResult.from_json(raw["result"]),
        tolerances=raw.get("tolerances", {}),
    )


def save_boot_baseline(name: str, result: BootResult) -> Path:
    path = boot_baselines_dir() / f"boot_{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    baseline = BootBaseline(name=name, timestamp=result.timestamp, result=result)
    path.write_text(json.dumps(baseline.to_json(), indent=2, ensure_ascii=False) + "\n")
    return path


# ── Comparison ────────────────────────────────────────────────


def compare_boot(result: BootResult, baseline: BootBaseline) -> list[BootViolation]:
    violations: list[BootViolation] = []
    tol = baseline.tolerances

    if not result.all_backends_healthy:
        violations.append(
            BootViolation(
                metric="all_backends_healthy",
                observed=0,
                baseline=1,
                limit_ratio=1.0,
                severity="fail",
            )
        )
        return violations

    base_total = baseline.result.total_seconds
    if base_total > 0:
        ratio = result.total_seconds / base_total
        fail_limit = tol.get("total_fail_ratio", 2.00)
        warn_limit = tol.get("total_warn_ratio", 1.30)
        if ratio > fail_limit:
            violations.append(
                BootViolation(
                    metric="total_seconds",
                    observed=result.total_seconds,
                    baseline=base_total,
                    limit_ratio=fail_limit,
                    severity="fail",
                )
            )
        elif ratio > warn_limit:
            violations.append(
                BootViolation(
                    metric="total_seconds",
                    observed=result.total_seconds,
                    baseline=base_total,
                    limit_ratio=warn_limit,
                    severity="warn",
                )
            )

    return violations


# ── Rendering ─────────────────────────────────────────────────


def render_boot_stdout(
    result: BootResult,
    baseline: BootBaseline | None,
    violations: list[BootViolation],
) -> None:
    click.echo()
    click.secho("autosre perf boot — summary", bold=True)
    click.echo(f"  timestamp:      {result.timestamp}")
    click.echo(f"  total:          {result.total_seconds:.1f}s")
    if result.compose_up_seconds is not None:
        click.echo(f"  compose up:     {result.compose_up_seconds:.1f}s")
    if result.preflight_seconds is not None:
        click.echo(f"  preflight:      {result.preflight_seconds:.1f}s")
    if result.server_ready_seconds is not None:
        click.echo(f"  server ready:   {result.server_ready_seconds:.1f}s")
    status_color = "green" if result.all_backends_healthy else "red"
    click.secho(
        f"  backends:       {'all healthy' if result.all_backends_healthy else 'UNHEALTHY'}",
        fg=status_color,
    )

    if baseline:
        click.echo()
        click.echo(f"  baseline:       {baseline.name} ({baseline.result.total_seconds:.1f}s)")
        if baseline.result.total_seconds > 0:
            ratio = result.total_seconds / baseline.result.total_seconds
            click.echo(f"  ratio:          {ratio:.2f}×")

    if violations:
        click.echo()
        for v in violations:
            color = "red" if v.severity == "fail" else "yellow"
            click.secho(f"  {v.summary()}", fg=color)
    elif baseline:
        click.secho("  clean — within tolerance", fg="green")
    click.echo()


def render_boot_markdown(
    result: BootResult,
    baseline: BootBaseline | None,
    violations: list[BootViolation],
    title: str = "autosre perf boot",
) -> str:
    lines = [f"# {title} ({result.timestamp})", ""]
    lines.append(f"- **total**: {result.total_seconds:.1f}s")
    if result.compose_up_seconds is not None:
        lines.append(f"- **compose up**: {result.compose_up_seconds:.1f}s")
    if result.preflight_seconds is not None:
        lines.append(f"- **preflight**: {result.preflight_seconds:.1f}s")
    if result.server_ready_seconds is not None:
        lines.append(f"- **server ready**: {result.server_ready_seconds:.1f}s")
    lines.append(f"- **backends**: {'all healthy' if result.all_backends_healthy else 'UNHEALTHY'}")
    lines.append("")

    if baseline:
        lines.append(f"## Comparison vs `{baseline.name}`")
        lines.append("")
        lines.append(f"- baseline total: {baseline.result.total_seconds:.1f}s")
        if baseline.result.total_seconds > 0:
            ratio = result.total_seconds / baseline.result.total_seconds
            lines.append(f"- ratio: {ratio:.2f}x")
        lines.append("")

    if violations:
        lines.append("## Violations")
        lines.append("")
        for v in violations:
            lines.append(f"- {v.summary()}")
        lines.append("")

    return "\n".join(lines) + "\n"
