"""Tests for autosre.eval.report — markdown rendering."""

from __future__ import annotations

from pathlib import Path

from autosre.eval.differ import (
    CompareOverrides,
    CompareResult,
    DiffResult,
    MatchedPair,
    RunManifest,
)
from autosre.eval.extract import AgentExtraction, ExtractionReport
from autosre.eval.findings import Finding
from autosre.eval.report import (
    render_compare,
    render_single,
    write_compare_md,
    write_report_md,
)
from autosre.eval.runner import RunResult, SuiteRunResult
from autosre.eval.snapshot import Snapshot
from autosre.swarm.launcher import CaptureResult


def _snap(tmp_path: Path) -> Snapshot:
    return Snapshot(
        path=tmp_path / "snap",
        mode="git-worktree",
        source_repo=tmp_path,
        source_sha="aaaa",
        snapshot_digest="deadbeef",
        file_count=1,
        included_untracked=False,
    )


def _cap(tmp_path: Path) -> CaptureResult:
    return CaptureResult(
        exit_code=0,
        transcript_path=tmp_path / "t.jsonl",
        stderr_path=tmp_path / "e.log",
        duration_ms=1.0,
    )


def _finding(title: str) -> Finding:
    return Finding(
        suite="security",
        category="security",
        severity="high",
        file="src/x.py",
        line=10,
        title=title,
    ).with_id()


class TestRenderSingle:
    def test_basic_shape(self, tmp_path: Path) -> None:
        (tmp_path / "snap").mkdir()
        result = RunResult(
            run_id="r1",
            run_dir=tmp_path / "run",
            provider="local",
            target_repo=tmp_path / "target",
            target_sha="abcd",
            snapshot_digest="ffff",
            suites=[
                SuiteRunResult(
                    suite="security",
                    findings=[_finding("SQL injection")],
                    report=ExtractionReport(
                        suite="security",
                        agents=[
                            AgentExtraction(
                                role="injection",
                                status="ok",
                                source="/tmp/x",
                                finding_count=1,
                            )
                        ],
                    ),
                    turns=[],
                    capture=_cap(tmp_path),
                    snapshot=_snap(tmp_path),
                )
            ],
        )
        md = render_single(result)
        assert "# Eval run: r1" in md
        assert "Provider: **local**" in md
        assert "abcd" in md
        assert "SQL injection" in md
        assert "security" in md
        assert "Extraction status per agent" in md

    def test_zero_findings_shows_placeholder(self, tmp_path: Path) -> None:
        (tmp_path / "snap").mkdir()
        result = RunResult(
            run_id="r1",
            run_dir=tmp_path / "run",
            provider="local",
            target_repo=tmp_path / "target",
            target_sha="abcd",
            snapshot_digest="ffff",
            suites=[
                SuiteRunResult(
                    suite="security",
                    findings=[],
                    report=ExtractionReport(suite="security", agents=[]),
                    turns=[],
                    capture=None,
                    snapshot=_snap(tmp_path),
                )
            ],
        )
        md = render_single(result)
        assert "_No findings reported._" in md

    def test_write_report_md(self, tmp_path: Path) -> None:
        (tmp_path / "snap").mkdir()
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        result = RunResult(
            run_id="r1",
            run_dir=run_dir,
            provider="local",
            target_repo=tmp_path,
            target_sha="s",
            snapshot_digest="d",
            suites=[],
        )
        path = write_report_md(result)
        assert path.exists()
        assert "# Eval run: r1" in path.read_text()


class TestRenderCompare:
    def _manifest(self, **kw: object) -> RunManifest:
        defaults = {
            "run_id": "r",
            "provider": "local",
            "target_repo": "/repo",
            "target_sha": "s1",
            "snapshot_digest": "d1",
            "suites": ["security"],
            "run_dir": Path("/tmp"),
            "model": "m",
        }
        defaults.update(kw)
        return RunManifest(**defaults)  # type: ignore[arg-type]

    def test_basic_compare(self) -> None:
        d = DiffResult(
            suite="security",
            a_count=3,
            b_count=4,
            both=[MatchedPair(a_idx=0, b_idx=0, score=0.95)],
            partial=[MatchedPair(a_idx=1, b_idx=1, score=0.6)],
            a_only=[2],
            b_only=[2, 3],
        )
        result = CompareResult(
            run_a=self._manifest(run_id="ra", provider="local"),
            run_b=self._manifest(run_id="rb", provider="anthropic"),
            per_suite={"security": d},
            warnings=[],
            overrides=CompareOverrides(),
        )
        md = render_compare(result)
        assert "ra" in md and "rb" in md
        assert "security" in md
        # Conservation numbers appear.
        assert "|A| = 3" in md and "|B| = 4" in md

    def test_warning_banner_when_invalid(self, tmp_path: Path) -> None:
        d = DiffResult(suite="security", a_count=0, b_count=0)
        result = CompareResult(
            run_a=self._manifest(run_id="ra", provider="local"),
            run_b=self._manifest(run_id="rb", provider="anthropic", target_sha="s2"),
            per_suite={"security": d},
            warnings=["SHA mismatch: this compare is not valid as model-quality data"],
            overrides=CompareOverrides(allow_sha_mismatch=True),
        )
        md = render_compare(result)
        assert "not** valid as model-quality data" in md

    def test_write_compare_md(self, tmp_path: Path) -> None:
        d = DiffResult(suite="security", a_count=0, b_count=0)
        result = CompareResult(
            run_a=self._manifest(run_id="ra"),
            run_b=self._manifest(run_id="rb", provider="anthropic"),
            per_suite={"security": d},
            warnings=[],
            overrides=CompareOverrides(),
        )
        out = write_compare_md(result, tmp_path / "compare-out")
        assert out.exists()
        assert "ra" in out.read_text()
