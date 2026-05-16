"""Concurrent-workload regression harness for the shared vLLM server.

Drives translation (priority=-10) and coding (priority=10) workloads
simultaneously against :8010, records per-workload TTFT/TPS percentiles
plus vLLM scheduler counters, and compares against committed named
baselines under ``repos/auto-sre/benchmarks/baselines/``.

See ``autosre perf --help`` for usage and
``repos/auto-sre/benchmarks/README.md`` for the baseline workflow.
"""

from __future__ import annotations

from .baseline import Baseline, Violation, compare, load_baseline, save_baseline
from .boot import (
    BootBaseline,
    BootResult,
    BootViolation,
    compare_boot,
    load_boot_baseline,
    run_boot_benchmark,
    save_boot_baseline,
)
from .harness import PhaseResult, RunConfig, RunResult, run
from .report import render_markdown, render_stdout
from .smoke import SmokeResult, run_smoke
from .workloads import CODING_WORKLOAD, TRANSLATION_WORKLOAD, Workload

__all__ = [
    "CODING_WORKLOAD",
    "TRANSLATION_WORKLOAD",
    "Baseline",
    "BootBaseline",
    "BootResult",
    "BootViolation",
    "PhaseResult",
    "RunConfig",
    "RunResult",
    "SmokeResult",
    "Violation",
    "Workload",
    "compare",
    "compare_boot",
    "load_baseline",
    "load_boot_baseline",
    "render_markdown",
    "render_stdout",
    "run",
    "run_boot_benchmark",
    "run_smoke",
    "save_baseline",
    "save_boot_baseline",
]
