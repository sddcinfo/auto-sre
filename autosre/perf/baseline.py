# ruff: noqa: RUF001, RUF002, RUF003
"""Load, save, and compare committed perf baselines.

A baseline is a pair of files under ``repos/auto-sre/benchmarks/baselines/``:

- ``<name>.json`` — machine-readable :class:`Baseline` snapshot (full
  ``RunResult`` plus a ``tolerances`` object). This is the comparison
  target; updating it requires a deliberate commit.
- ``<name>.md`` — human-readable sibling produced by
  :mod:`autosre.perf.report`. Reviewed in PRs when a baseline is
  intentionally updated.

:func:`compare` evaluates a freshly-captured :class:`RunResult` against
a loaded :class:`Baseline` and returns a list of :class:`Violation`
records. CLI callers translate violation severity into exit codes:
``warn`` ⇒ 1, ``fail`` ⇒ 2.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from autosre.perf.harness import PhaseResult, RunResult

# Minimum successful samples required for ratio-based comparisons.
# Below this threshold, percentile estimates are too noisy for the default
# tolerance bands — a single outlier swings p95 wildly. Phases with fewer
# ok samples emit a "warn" violation instead of ratio-based checks.
_MIN_SAMPLES_FOR_COMPARISON = 20

# Default tolerances — see plan §6. Ratios apply to (observed / baseline)
# for "upper bound" metrics (TTFT, errors) and (baseline / observed) for
# "lower bound" metrics (TPS), i.e. each rule is "observed must not
# exceed limit × baseline" or "observed must be ≥ limit × baseline".

_DEFAULT_TOLERANCES: dict[str, dict[str, dict[str, float]]] = {
    "isolated": {
        "translation": {
            "ttft_p50_max_ratio": 1.20,
            "ttft_p95_max_ratio": 1.30,
            "tps_p50_min_ratio": 0.85,
        },
        "coding": {
            "ttft_p50_max_ratio": 1.20,
            "ttft_p95_max_ratio": 1.30,
            "tps_p50_min_ratio": 0.85,
        },
    },
    "contention": {
        "translation": {
            "ttft_p50_max_ratio": 1.50,
            "ttft_p95_max_ratio": 2.00,
            "tps_p50_min_ratio": 0.70,
        },
        "coding": {
            "ttft_p50_max_ratio": 1.50,
            "ttft_p95_max_ratio": 2.00,
            "tps_p50_min_ratio": 0.40,
        },
    },
    "absolute": {
        "translation": {
            # Translation TTFT p95 under contention must not be more
            # than 2× its own isolated measurement — the "hard SLO"
            # that protects live translation from coding interference
            # regardless of how the baseline shifts.
            "contention_ttft_p95_vs_isolated_max_ratio": 2.0,
        },
        "coding": {
            "contention_tps_vs_isolated_min_ratio": 0.40,
        },
    },
}


@dataclass
class Baseline:
    name: str
    timestamp: str
    environment: dict[str, Any]
    phases: list[PhaseResult]
    tolerances: dict[str, Any] = field(default_factory=lambda: dict(_DEFAULT_TOLERANCES))

    def phase(self, workload: str, phase: str) -> PhaseResult | None:
        for p in self.phases:
            if p.workload == workload and p.phase == phase:
                return p
        return None

    kind: str = "vllm"

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "timestamp": self.timestamp,
            "kind": self.kind,
            "environment": self.environment,
            "phases": [asdict(p) for p in self.phases],
            "tolerances": self.tolerances,
        }


@dataclass
class Violation:
    metric: str
    workload: str
    phase: str
    observed: float
    baseline: float
    limit_ratio: float
    direction: str  # "max" or "min"
    severity: str  # "warn" or "fail"

    def summary(self) -> str:
        cmp = "≤" if self.direction == "max" else "≥"
        return (
            f"[{self.severity.upper()}] {self.workload}/{self.phase} {self.metric}: "
            f"observed={self.observed:.2f}, baseline={self.baseline:.2f}, "
            f"limit: observed {cmp} {self.limit_ratio:.2f}× baseline"
        )


def baselines_dir() -> Path:
    """Locate the committed baselines directory inside the auto-sre repo."""
    here = Path(__file__).resolve()
    # autosre/perf/baseline.py → repo root is 2 levels up from autosre/
    repo_root = here.parents[2]
    return repo_root / "benchmarks" / "baselines"


def load_baseline(name: str, *, expected_kind: str = "vllm") -> Baseline:
    path = baselines_dir() / f"{name}.json"
    if not path.exists():
        msg = f"Baseline not found: {path}"
        raise FileNotFoundError(msg)
    raw = json.loads(path.read_text())
    # Backward compat: baselines written before the kind field default to "vllm"
    kind = raw.get("kind", "vllm")
    if kind != expected_kind:
        msg = f"Baseline {name} has kind={kind!r}, expected {expected_kind!r}"
        raise ValueError(msg)
    phases = [PhaseResult(**p) for p in raw["phases"]]
    return Baseline(
        name=raw["name"],
        timestamp=raw["timestamp"],
        environment=raw.get("environment", {}),
        phases=phases,
        tolerances=raw.get("tolerances") or dict(_DEFAULT_TOLERANCES),
        kind=kind,
    )


def save_baseline(name: str, run: RunResult) -> Path:
    """Write a :class:`RunResult` as a committed baseline.

    Caller is responsible for ``git add`` + commit; this function only
    writes the file.
    """
    path = baselines_dir() / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    baseline = Baseline(
        name=name,
        timestamp=run.timestamp,
        environment=run.environment,
        phases=list(run.phases),
        tolerances=dict(_DEFAULT_TOLERANCES),
    )
    path.write_text(json.dumps(baseline.to_json(), indent=2, ensure_ascii=False) + "\n")
    return path


# ── Comparison ─────────────────────────────────────────────────


def _severity(observed: float, baseline: float, limit: float, direction: str) -> str | None:
    """Return None (clean), "warn" (1×–2× past limit) or "fail" (beyond)."""
    if baseline <= 0:
        return None
    ratio = observed / baseline
    if direction == "max":
        if ratio <= limit:
            return None
        # "warn" band: observed is past the limit but within (limit, 2*limit).
        # Example: limit=1.20, observed_ratio=1.40 → warn; 2.50 → fail.
        warn_cap = limit + (limit - 1.0) * 2.0 + 0.10
        return "warn" if ratio <= warn_cap else "fail"
    # Lower-bound (min) direction: observed must be at least limit × baseline.
    if ratio >= limit:
        return None
    # "warn" band: observed is below the limit but within (limit*0.5, limit).
    warn_floor = limit - (1.0 - limit) * 2.0 - 0.10
    return "warn" if ratio >= warn_floor else "fail"


def _check(
    violations: list[Violation],
    metric: str,
    workload: str,
    phase: str,
    observed: float,
    baseline: float,
    limit: float,
    direction: str,
) -> None:
    sev = _severity(observed, baseline, limit, direction)
    if sev is None:
        return
    violations.append(
        Violation(
            metric=metric,
            workload=workload,
            phase=phase,
            observed=observed,
            baseline=baseline,
            limit_ratio=limit,
            direction=direction,
            severity=sev,
        )
    )


def compare(run: RunResult, baseline: Baseline) -> list[Violation]:
    """Return a list of :class:`Violation` records.

    An empty list means the run is clean. A list containing any
    ``severity="fail"`` entry means the run is a hard regression.
    """
    violations: list[Violation] = []
    tol = baseline.tolerances

    for phase_name in ("isolated", "contention"):
        phase_tol = tol.get(phase_name, {})
        for workload in ("translation", "coding"):
            wl_tol = phase_tol.get(workload, {})
            observed = run.phase(workload, phase_name)
            base = baseline.phase(workload, phase_name)
            if observed is None or base is None:
                continue

            # Skip ratio-based checks when ok sample count is too low —
            # percentile estimates are too noisy for meaningful comparison.
            ok_samples = observed.samples - observed.errors
            if ok_samples < _MIN_SAMPLES_FOR_COMPARISON:
                violations.append(
                    Violation(
                        metric="insufficient_samples",
                        workload=workload,
                        phase=phase_name,
                        observed=float(ok_samples),
                        baseline=float(_MIN_SAMPLES_FOR_COMPARISON),
                        limit_ratio=0.0,
                        direction="min",
                        severity="warn",
                    )
                )
                continue

            _check(
                violations,
                "ttft_p50_ms",
                workload,
                phase_name,
                observed.ttft_p50_ms,
                base.ttft_p50_ms,
                wl_tol.get("ttft_p50_max_ratio", 1.20),
                "max",
            )
            _check(
                violations,
                "ttft_p95_ms",
                workload,
                phase_name,
                observed.ttft_p95_ms,
                base.ttft_p95_ms,
                wl_tol.get("ttft_p95_max_ratio", 1.30),
                "max",
            )
            _check(
                violations,
                "tps_p50",
                workload,
                phase_name,
                observed.tps_p50,
                base.tps_p50,
                wl_tol.get("tps_p50_min_ratio", 0.85),
                "min",
            )

    # Errors are absolute, not ratio-based — any error is a failure.
    for phase in run.phases:
        if phase.errors > 0:
            violations.append(
                Violation(
                    metric="errors",
                    workload=phase.workload,
                    phase=phase.phase,
                    observed=float(phase.errors),
                    baseline=0.0,
                    limit_ratio=0.0,
                    direction="max",
                    severity="fail",
                )
            )

    # Absolute SLOs: the hard coexistence contract, independent of baseline.
    abs_tol = tol.get("absolute", {})

    tr_iso = run.phase("translation", "isolated")
    tr_con = run.phase("translation", "contention")
    tr_limit = abs_tol.get("translation", {}).get("contention_ttft_p95_vs_isolated_max_ratio", 2.0)
    if tr_iso and tr_con and tr_iso.ttft_p95_ms > 0:
        _check(
            violations,
            "ttft_p95_contention_over_isolated",
            "translation",
            "absolute",
            tr_con.ttft_p95_ms,
            tr_iso.ttft_p95_ms,
            tr_limit,
            "max",
        )

    cd_iso = run.phase("coding", "isolated")
    cd_con = run.phase("coding", "contention")
    cd_limit = abs_tol.get("coding", {}).get("contention_tps_vs_isolated_min_ratio", 0.40)
    if cd_iso and cd_con and cd_iso.tps_p50 > 0:
        _check(
            violations,
            "tps_p50_contention_over_isolated",
            "coding",
            "absolute",
            cd_con.tps_p50,
            cd_iso.tps_p50,
            cd_limit,
            "min",
        )

    return violations
