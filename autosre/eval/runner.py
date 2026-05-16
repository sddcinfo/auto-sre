"""EvalRunner — single-provider capture orchestration.

One invocation = one provider, N suites. The runner:

1. Creates a timestamped run directory.
2. Writes ``manifest.json``.
3. For each suite: materializes a read-only snapshot, launches Claude
   Code via ``SwarmLauncher`` in eval mode, normalizes telemetry into
   ``turns.jsonl``, extracts findings into ``findings.jsonl``, writes a
   ``parse_report.json``, and cleans up the snapshot.
4. Appends one row to ``runs.jsonl``.

Runs are independent. Running the other provider is a separate
invocation. Diffing them is a third separate invocation
(``autosre eval compare``).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol

from autosre.eval.capture import TurnRecord, normalize_run, write_turns
from autosre.eval.extract import (
    AgentExtraction,
    ExtractionReport,
    extract_agent_findings,
    write_findings_jsonl,
    write_parse_report,
)
from autosre.eval.snapshot import Snapshot, cleanup, materialize
from autosre.eval.suite import EvalSuite, load_all_suites
from autosre.swarm.launcher import (
    DEFAULT_ANTHROPIC_MODEL,
    CaptureResult,
    EvalLaunchSpec,
)

if TYPE_CHECKING:
    from autosre.eval.findings import Finding

RunProvider = Literal["local", "anthropic"]


EVAL_RUNS_ROOT = Path.home() / ".local" / "share" / "autosre" / "eval-runs"
DEFAULT_PROXY_LOG = Path.home() / ".local" / "share" / "autosre" / "proxy-requests.jsonl"


class LauncherFactory(Protocol):
    """Test seam: lets tests inject a fake launcher without touching files."""

    def __call__(
        self,
        provider: RunProvider,
        suite: EvalSuite,
    ) -> SwarmLauncherLike: ...


class SwarmLauncherLike(Protocol):
    """Subset of ``SwarmLauncher`` the runner needs."""

    def launch(
        self,
        model_key: str | None = None,
        *,
        eval_mode: EvalLaunchSpec | None = None,
    ) -> CaptureResult | dict[str, object]: ...


# ── Manifest + result objects ──────────────────────────────────────


@dataclass
class SuiteRunResult:
    suite: str
    findings: list[Finding]
    report: ExtractionReport
    turns: list[TurnRecord]
    capture: CaptureResult | None
    snapshot: Snapshot


@dataclass
class RunResult:
    run_id: str
    run_dir: Path
    provider: RunProvider
    target_repo: Path
    target_sha: str | None
    snapshot_digest: str | None
    suites: list[SuiteRunResult] = field(default_factory=list)
    manifest_path: Path | None = None


# ── Runner ─────────────────────────────────────────────────────────


class EvalRunner:
    """Orchestrates one single-provider capture run."""

    def __init__(
        self,
        *,
        runs_root: Path | None = None,
        proxy_log_path: Path | None = None,
        launcher_factory: LauncherFactory | None = None,
        clock: callable | None = None,  # type: ignore[valid-type]
    ) -> None:
        self.runs_root = runs_root or EVAL_RUNS_ROOT
        self.proxy_log_path = proxy_log_path or DEFAULT_PROXY_LOG
        self.launcher_factory = launcher_factory
        self.clock = clock or time.time

    # ── Public entry point ────────────────────────────────────────

    def run(
        self,
        *,
        provider: RunProvider,
        suites: list[str],
        target: Path,
        run_id: str | None = None,
        anthropic_model: str = DEFAULT_ANTHROPIC_MODEL,
        allow_dirty: bool = False,
        keep_worktrees: bool = False,
    ) -> RunResult:
        target = target.resolve()
        all_suites = load_all_suites()
        unknown = [s for s in suites if s not in all_suites]
        if unknown:
            raise ValueError(f"unknown suites: {unknown}")

        run_dir, final_run_id = self._make_run_dir(provider, run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "suites").mkdir()
        (run_dir / "worktrees").mkdir()

        # First suite materializes to infer SHA/digest for the manifest.
        suite_results: list[SuiteRunResult] = []
        run_target_sha: str | None = None
        run_snapshot_digest: str | None = None

        for suite_name in suites:
            suite = all_suites[suite_name]
            snapshot = materialize(
                target,
                run_dir / "worktrees" / suite_name,
                allow_dirty=allow_dirty,
            )
            if run_target_sha is None:
                run_target_sha = snapshot.source_sha
                run_snapshot_digest = snapshot.snapshot_digest

            suite_dir = run_dir / "suites" / suite_name
            suite_dir.mkdir(parents=True, exist_ok=True)
            agent_outputs = suite_dir / "agent-outputs"
            agent_outputs.mkdir(exist_ok=True)

            # The "primary" findings file receives the team-lead summary.
            # Per-sub-agent files live under ``agent-outputs/<role>.json``
            # and are extracted individually. The team lead is free to
            # also write ``findings.json`` which the extractor picks up as
            # one of the agents.
            findings_file = agent_outputs / "lead.json"

            # Run the swarm.
            capture = self._launch_suite(
                provider=provider,
                suite=suite,
                snapshot=snapshot,
                capture_dir=suite_dir,
                findings_file=findings_file,
                run_id=f"{final_run_id}::{suite_name}",
                anthropic_model=anthropic_model,
            )

            # Normalize turns regardless of provider.
            turns_path = suite_dir / "turns.jsonl"
            turns = normalize_run(
                provider=provider,
                run_id=f"{final_run_id}::{suite_name}",
                capture_dir=suite_dir,
                proxy_log_path=self.proxy_log_path,
                out_path=turns_path,
            )

            # Extract findings from agent-outputs/*.json.
            merged_findings, report = self._extract_suite_findings(
                suite=suite,
                agent_outputs=agent_outputs,
                turns_path=turns_path,
                provider=provider,
            )

            write_findings_jsonl(merged_findings, suite_dir / "findings.jsonl")
            write_parse_report(report, suite_dir / "parse_report.json")

            suite_results.append(
                SuiteRunResult(
                    suite=suite_name,
                    findings=merged_findings,
                    report=report,
                    turns=turns,
                    capture=capture,
                    snapshot=snapshot,
                )
            )

            if not keep_worktrees:
                cleanup(snapshot)

        result = RunResult(
            run_id=final_run_id,
            run_dir=run_dir,
            provider=provider,
            target_repo=target,
            target_sha=run_target_sha,
            snapshot_digest=run_snapshot_digest,
            suites=suite_results,
        )
        self._write_manifest(result, anthropic_model=anthropic_model)
        self._append_runs_index(result)
        return result

    # ── Internals ─────────────────────────────────────────────────

    def _make_run_dir(
        self,
        provider: RunProvider,
        run_id: str | None,
    ) -> tuple[Path, str]:
        ts = time.strftime("%Y-%m-%dT%H-%M-%S", time.gmtime(self.clock()))
        tag = run_id or "run"
        final = f"{ts}-{provider}-{tag}"
        return self.runs_root / final, final

    def _launch_suite(
        self,
        *,
        provider: RunProvider,
        suite: EvalSuite,
        snapshot: Snapshot,
        capture_dir: Path,
        findings_file: Path,
        run_id: str,
        anthropic_model: str,  # noqa: ARG002  (used by real factories)
    ) -> CaptureResult | None:
        if self.launcher_factory is None:
            # Default path: real SwarmLauncher with a stub backend is not
            # possible without a real backend. We require the caller to
            # pass a launcher_factory in all non-smoke scenarios.
            # Producing no launch is a valid outcome — the extractor will
            # simply see an empty agent-outputs directory and record
            # "failed" for every expected role.
            return None

        launcher = self.launcher_factory(provider, suite)
        spec = EvalLaunchSpec(
            capture_dir=capture_dir,
            worktree_path=snapshot.path,
            findings_file=findings_file,
            suite_name=suite.name,
            run_id=run_id,
        )
        out = launcher.launch(eval_mode=spec)
        return out if isinstance(out, CaptureResult) else None

    def _extract_suite_findings(
        self,
        *,
        suite: EvalSuite,
        agent_outputs: Path,
        turns_path: Path,
        provider: RunProvider,
    ) -> tuple[list[Finding], ExtractionReport]:
        """Iterate expected agent files + any extras and merge findings."""
        expected_roles = [f"agent-{i + 1}" for i in range(suite.num_agents)]
        # Also pick up any file the agents happened to write, even if
        # the name doesn't match an expected role — we still want those.
        discovered: dict[str, Path] = {}
        for p in sorted(agent_outputs.glob("*.json")):
            discovered[p.stem] = p
        for role in expected_roles:
            discovered.setdefault(role, agent_outputs / f"{role}.json")

        merged: list[Finding] = []
        agent_reports: list[AgentExtraction] = []
        for role in sorted(discovered.keys()):
            agent_file = discovered[role]
            findings, report = extract_agent_findings(
                agent_file,
                role=role,
                suite=suite.name,
                provider=provider,
                turns_path=turns_path,
            )
            merged.extend(findings)
            agent_reports.append(report)

        return merged, ExtractionReport(suite=suite.name, agents=agent_reports)

    def _write_manifest(
        self,
        result: RunResult,
        *,
        anthropic_model: str,
    ) -> None:
        data = {
            "run_id": result.run_id,
            "provider": result.provider,
            "target_repo": str(result.target_repo),
            "target_sha": result.target_sha,
            "snapshot_digest": result.snapshot_digest,
            "suites": [sr.suite for sr in result.suites],
            "model": (anthropic_model if result.provider == "anthropic" else "local-backend"),
            "created_at": time.time(),
            "harness_version": _harness_version(),
            "per_suite": {
                sr.suite: {
                    "finding_count": len(sr.findings),
                    "parse_failures": sum(1 for a in sr.report.agents if a.status == "failed"),
                    "snapshot_mode": sr.snapshot.mode,
                }
                for sr in result.suites
            },
        }
        path = result.run_dir / "manifest.json"
        path.write_text(json.dumps(data, indent=2))
        result.manifest_path = path

    def _append_runs_index(self, result: RunResult) -> None:
        index = self.runs_root / "runs.jsonl"
        index.parent.mkdir(parents=True, exist_ok=True)
        totals = sum(len(sr.findings) for sr in result.suites)
        parse_failures = sum(
            1 for sr in result.suites for a in sr.report.agents if a.status == "failed"
        )
        row = {
            "run_id": result.run_id,
            "ts": time.time(),
            "provider": result.provider,
            "target_repo": str(result.target_repo),
            "target_sha": result.target_sha,
            "snapshot_digest": result.snapshot_digest,
            "suites": [sr.suite for sr in result.suites],
            "totals": {
                "findings": totals,
                "by_suite": {sr.suite: len(sr.findings) for sr in result.suites},
                "parse_failures": parse_failures,
            },
        }
        with index.open("a") as f:
            f.write(json.dumps(row) + "\n")


def _harness_version() -> str:
    try:
        from autosre import __version__ as v

        return str(v)
    except Exception:  # pragma: no cover
        return "unknown"


__all__ = [
    "DEFAULT_PROXY_LOG",
    "EVAL_RUNS_ROOT",
    "EvalRunner",
    "RunProvider",
    "RunResult",
    "SuiteRunResult",
    "write_turns",
]
