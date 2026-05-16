# ruff: noqa: RUF003, TC003
"""Unit tests for the autosre perf harness.

All tests are pure-Python / no network. They exercise:

- Percentile + summarize math
- Baseline load/save roundtrip
- Tolerance comparison (clean, warn, fail)
- Error-aware summary counting
- Markdown renderer stability (structural, not exact match)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from autosre.perf.baseline import Baseline, Violation, compare, load_baseline, save_baseline
from autosre.perf.harness import (
    PhaseResult,
    ProxySanity,
    RunResult,
    Sample,
    SchedulerCounters,
    percentile,
    summarize,
)
from autosre.perf.report import render_markdown
from autosre.perf.workloads import CODING_WORKLOAD, TRANSLATION_WORKLOAD

# ── percentile ──────────────────────────────────────────────────


class TestPercentile:
    def test_empty_returns_zero(self) -> None:
        assert percentile([], 0.5) == 0.0

    def test_single_value(self) -> None:
        assert percentile([42.0], 0.5) == 42.0
        assert percentile([42.0], 0.99) == 42.0

    def test_sorted_input(self) -> None:
        vals = [10.0, 20.0, 30.0, 40.0, 50.0]
        assert percentile(vals, 0.0) == 10.0
        assert percentile(vals, 1.0) == 50.0
        # Linear interpolation at p50 of 5 values → exactly middle
        assert percentile(vals, 0.5) == 30.0

    def test_unsorted_input_is_sorted(self) -> None:
        vals = [50.0, 10.0, 40.0, 20.0, 30.0]
        assert percentile(vals, 0.5) == 30.0


# ── summarize ───────────────────────────────────────────────────


def _sample(ttft: float, tokens: int, tps: float, err: str | None = None) -> Sample:
    return Sample(
        workload="coding",
        ttft_ms=ttft,
        gen_ms=tokens * 50.0,
        output_tokens=tokens,
        tps=tps,
        error=err,
    )


class TestSummarize:
    def test_clean_samples(self) -> None:
        samples = [
            _sample(100.0, 50, 30.0),
            _sample(200.0, 50, 25.0),
            _sample(300.0, 50, 20.0),
            _sample(400.0, 50, 15.0),
        ]
        r = summarize("coding", "isolated", samples, wall=10.0)
        assert r.samples == 4
        assert r.errors == 0
        assert r.ttft_p50_ms == 250.0
        assert r.tps_agg == 20.0  # 200 total tokens / 10s wall

    def test_errors_excluded_from_percentiles_but_counted(self) -> None:
        samples = [
            _sample(100.0, 50, 30.0),
            _sample(0.0, 0, 0.0, err="RemoteProtocolError"),
            _sample(200.0, 50, 20.0),
        ]
        r = summarize("coding", "contention", samples, wall=5.0)
        assert r.samples == 3
        assert r.errors == 1
        # Percentiles computed only from the 2 successful samples
        assert r.ttft_p50_ms == 150.0
        assert r.tps_agg == 20.0  # 100 tokens / 5s

    def test_zero_token_samples_excluded(self) -> None:
        samples = [
            _sample(100.0, 50, 30.0),
            _sample(0.0, 0, 0.0),  # no tokens, no error — e.g. empty stream
        ]
        r = summarize("coding", "isolated", samples, wall=2.0)
        assert r.samples == 2
        assert r.errors == 0
        assert r.ttft_p50_ms == 100.0


# ── Baseline roundtrip ─────────────────────────────────────────


def _phase(
    workload: str, phase: str, ttft_p50: float, ttft_p95: float, tps_p50: float
) -> PhaseResult:
    return PhaseResult(
        workload=workload,
        phase=phase,
        samples=50,
        ttft_p50_ms=ttft_p50,
        ttft_p95_ms=ttft_p95,
        ttft_p99_ms=ttft_p95 * 1.1,
        tps_p50=tps_p50,
        tps_p95=tps_p50 * 0.9,
        tps_agg=tps_p50,
        errors=0,
        wall_seconds=60.0,
    )


def _run(**overrides: Any) -> RunResult:
    phases = [
        _phase("translation", "isolated", 200.0, 300.0, 40.0),
        _phase("coding", "isolated", 800.0, 1500.0, 30.0),
        _phase("translation", "contention", 250.0, 420.0, 35.0),
        _phase("coding", "contention", 1100.0, 2500.0, 20.0),
    ]
    if "phases" in overrides:
        phases = overrides["phases"]
    return RunResult(
        timestamp="20260415T000000",
        config={"duration_seconds": 60},
        environment={"hostname": "test", "model_id": "test-model"},
        phases=phases,
        scheduler=SchedulerCounters(preemptions_delta=2, requests_running_peak=4.0),
        proxy_sanity=ProxySanity(
            ok=True, message_start=True, content_delta_count=3, message_stop=True
        ),
    )


class TestBaselineRoundtrip:
    def test_save_and_load(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("autosre.perf.baseline.baselines_dir", lambda: tmp_path)
        run = _run()
        path = save_baseline("test_baseline", run)
        assert path.exists()
        loaded = load_baseline("test_baseline")
        assert loaded.name == "test_baseline"
        assert loaded.timestamp == run.timestamp
        assert len(loaded.phases) == 4
        assert loaded.phase("translation", "isolated") == run.phases[0]

    def test_load_missing_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("autosre.perf.baseline.baselines_dir", lambda: tmp_path)
        with pytest.raises(FileNotFoundError):
            load_baseline("does_not_exist")

    def test_tolerances_persist_through_roundtrip(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("autosre.perf.baseline.baselines_dir", lambda: tmp_path)
        run = _run()
        save_baseline("test_baseline", run)
        # Hand-edit the JSON to tighten a tolerance and confirm it sticks
        path = tmp_path / "test_baseline.json"
        data = json.loads(path.read_text())
        data["tolerances"]["isolated"]["translation"]["ttft_p50_max_ratio"] = 1.05
        path.write_text(json.dumps(data))

        loaded = load_baseline("test_baseline")
        assert loaded.tolerances["isolated"]["translation"]["ttft_p50_max_ratio"] == 1.05


# ── compare ────────────────────────────────────────────────────


@pytest.fixture
def baseline_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Baseline:
    monkeypatch.setattr("autosre.perf.baseline.baselines_dir", lambda: tmp_path)
    save_baseline("fx", _run())
    return load_baseline("fx")


class TestCompare:
    def test_clean_run(self, baseline_fixture: Baseline) -> None:
        assert compare(_run(), baseline_fixture) == []

    def test_ttft_regression_fails(self, baseline_fixture: Baseline) -> None:
        # Triple the isolated coding TTFT — way past the 1.20× tolerance.
        phases = [
            _phase("translation", "isolated", 200.0, 300.0, 40.0),
            _phase("coding", "isolated", 2400.0, 4500.0, 30.0),
            _phase("translation", "contention", 250.0, 420.0, 35.0),
            _phase("coding", "contention", 1100.0, 2500.0, 20.0),
        ]
        run = _run(phases=phases)
        violations = compare(run, baseline_fixture)
        fails = [v for v in violations if v.severity == "fail" and v.metric == "ttft_p50_ms"]
        assert len(fails) >= 1
        assert any(v.workload == "coding" and v.phase == "isolated" for v in fails)

    def test_small_ttft_regression_warns(self, baseline_fixture: Baseline) -> None:
        # Bump isolated translation TTFT to 1.30× baseline — past 1.20 limit
        # but within the warn band (warn_cap = 1.20 + 0.40 + 0.10 = 1.70).
        phases = [
            _phase("translation", "isolated", 260.0, 300.0, 40.0),
            _phase("coding", "isolated", 800.0, 1500.0, 30.0),
            _phase("translation", "contention", 250.0, 420.0, 35.0),
            _phase("coding", "contention", 1100.0, 2500.0, 20.0),
        ]
        violations = compare(_run(phases=phases), baseline_fixture)
        warns = [v for v in violations if v.severity == "warn" and v.metric == "ttft_p50_ms"]
        assert len(warns) >= 1

    def test_tps_regression_fails(self, baseline_fixture: Baseline) -> None:
        # Cut isolated coding TPS to 15% of baseline — far below 0.85 floor.
        phases = [
            _phase("translation", "isolated", 200.0, 300.0, 40.0),
            _phase("coding", "isolated", 800.0, 1500.0, 4.5),
            _phase("translation", "contention", 250.0, 420.0, 35.0),
            _phase("coding", "contention", 1100.0, 2500.0, 20.0),
        ]
        violations = compare(_run(phases=phases), baseline_fixture)
        fails = [v for v in violations if v.severity == "fail" and v.metric == "tps_p50"]
        assert any(v.workload == "coding" and v.phase == "isolated" for v in fails)

    def test_any_error_is_fail(self, baseline_fixture: Baseline) -> None:
        phases = [
            _phase("translation", "isolated", 200.0, 300.0, 40.0),
            _phase("coding", "isolated", 800.0, 1500.0, 30.0),
            _phase("translation", "contention", 250.0, 420.0, 35.0),
            PhaseResult(
                workload="coding",
                phase="contention",
                samples=50,
                ttft_p50_ms=1100.0,
                ttft_p95_ms=2500.0,
                ttft_p99_ms=2700.0,
                tps_p50=20.0,
                tps_p95=18.0,
                tps_agg=20.0,
                errors=3,
                wall_seconds=60.0,
            ),
        ]
        violations = compare(_run(phases=phases), baseline_fixture)
        assert any(v.severity == "fail" and v.metric == "errors" for v in violations)

    def test_absolute_slo_contention_translation_ttft(self, baseline_fixture: Baseline) -> None:
        # Baseline-relative comparisons all clean, but contention
        # translation TTFT p95 is >2× its own isolated → absolute SLO fires.
        phases = [
            _phase("translation", "isolated", 200.0, 300.0, 40.0),
            _phase("coding", "isolated", 800.0, 1500.0, 30.0),
            _phase("translation", "contention", 250.0, 700.0, 35.0),  # 700 > 2*300
            _phase("coding", "contention", 1100.0, 2500.0, 20.0),
        ]
        violations = compare(_run(phases=phases), baseline_fixture)
        abs_hits = [v for v in violations if v.metric == "ttft_p95_contention_over_isolated"]
        assert len(abs_hits) == 1
        assert abs_hits[0].workload == "translation"


# ── Workload payloads ──────────────────────────────────────────


class TestWorkloads:
    def test_translation_has_priority_minus_10(self) -> None:
        payload, label = TRANSLATION_WORKLOAD.next_payload()
        assert payload["priority"] == -10
        assert payload["stream"] is True
        assert payload["stream_options"]["include_usage"] is True
        assert label.startswith(("ja_en:", "en_ja:"))
        # System message is the translator persona
        sys_msg = payload["messages"][0]
        assert sys_msg["role"] == "system"
        assert "translator" in sys_msg["content"]

    def test_coding_has_priority_10_and_tools(self) -> None:
        payload, label = CODING_WORKLOAD.next_payload()
        assert payload["priority"] == 10
        assert payload["stream"] is True
        assert len(payload["tools"]) == 20  # frozen tool count
        assert label.startswith("coding:")
        tool_names = {t["function"]["name"] for t in payload["tools"]}
        assert {"Read", "Write", "Edit", "Bash", "Grep", "Glob"} <= tool_names

    def test_translation_rotates(self) -> None:
        a, _ = TRANSLATION_WORKLOAD.next_payload()
        b, _ = TRANSLATION_WORKLOAD.next_payload()
        # JA→EN then EN→JA alternates the user content
        assert a["messages"][1]["content"] != b["messages"][1]["content"]


# ── Markdown ───────────────────────────────────────────────────


class TestMarkdown:
    def test_renders_all_sections(self) -> None:
        run = _run()
        md = render_markdown(run, None, [], title="test-run")
        assert "# test-run" in md
        assert "## Environment" in md
        assert "## Per-workload results" in md
        assert "## Scheduler counters" in md
        assert "## Proxy sanity" in md
        # All four phase rows present
        assert md.count("| translation |") + md.count("| coding |") == 4

    def test_renders_violations_when_present(self) -> None:
        run = _run()
        violations = [
            Violation(
                metric="ttft_p50_ms",
                workload="coding",
                phase="isolated",
                observed=2000.0,
                baseline=800.0,
                limit_ratio=1.20,
                direction="max",
                severity="fail",
            )
        ]
        baseline = Baseline(
            name="fx",
            timestamp="20260101T000000",
            environment={},
            phases=run.phases,
        )
        md = render_markdown(run, baseline, violations, title="test-run")
        assert "1 fail / 0 warn" in md
        assert "coding/isolated ttft_p50_ms" in md
