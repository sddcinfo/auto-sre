"""Tests for autosre.eval.extract and autosre.eval.runner.

We exercise the full single-provider run path end-to-end against a
throwaway git repo, with the Claude Code subprocess stubbed out by a
fake launcher that drops a pre-baked findings file. That lets us
assert the runner's orchestration, snapshot lifecycle, extraction
pipeline, manifest, and runs-index row without spawning real LLMs.
"""

from __future__ import annotations

import json
import subprocess
from typing import TYPE_CHECKING, Any

import pytest

from autosre.eval.capture import TurnRecord, write_turns
from autosre.eval.extract import (
    AgentExtraction,
    ExtractionReport,
    _scrape_turns_for_findings,
    _slice_braces,
    extract_agent_findings,
    write_parse_report,
)
from autosre.eval.runner import EvalRunner
from autosre.swarm.launcher import CaptureResult, EvalLaunchSpec

if TYPE_CHECKING:
    from pathlib import Path

# ── helper: throwaway git repo + fake launcher ─────────────────────


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(cwd), *args],
        check=True,
        capture_output=True,
    )


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "target"
    r.mkdir()
    _git(r, "init", "-q", "-b", "main")
    _git(r, "config", "user.email", "t@t.t")
    _git(r, "config", "user.name", "t")
    (r / "code.py").write_text("def foo():\n    return 1\n")
    (r / "README.md").write_text("hi\n")
    _git(r, "add", ".")
    _git(r, "commit", "-q", "-m", "init")
    return r


class FakeLauncher:
    """Writes a pre-baked findings file when ``launch`` is called."""

    def __init__(self, payload: dict[str, Any], duration_ms: float = 12.0) -> None:
        self.payload = payload
        self.duration_ms = duration_ms
        self.invocations: list[EvalLaunchSpec] = []

    def launch(
        self,
        model_key: str | None = None,
        *,
        eval_mode: EvalLaunchSpec | None = None,
    ) -> CaptureResult:
        assert eval_mode is not None
        self.invocations.append(eval_mode)
        # Write the findings file just like a real agent would.
        eval_mode.findings_file.parent.mkdir(parents=True, exist_ok=True)
        eval_mode.findings_file.write_text(json.dumps(self.payload))
        # Drop an empty transcript so normalize_run has something to read.
        (eval_mode.capture_dir / "transcript.jsonl").write_text("")
        return CaptureResult(
            exit_code=0,
            transcript_path=eval_mode.capture_dir / "transcript.jsonl",
            stderr_path=eval_mode.capture_dir / "stderr.log",
            duration_ms=self.duration_ms,
        )


# ── extract.py unit tests ──────────────────────────────────────────


class TestSliceBraces:
    def test_balanced(self) -> None:
        assert _slice_braces('prefix {"a": 1} suffix') == '{"a": 1}'

    def test_nested(self) -> None:
        assert _slice_braces('{"a": {"b": 1}} tail') == '{"a": {"b": 1}}'

    def test_strings_with_braces(self) -> None:
        # A brace inside a JSON string must not throw off the scanner.
        src = '{"a": "not { a brace"}'
        assert _slice_braces(src) == '{"a": "not { a brace"}'

    def test_no_brace_returns_none(self) -> None:
        assert _slice_braces("plain text") is None


class TestExtractAgentFindings:
    def test_primary_ok(self, tmp_path: Path) -> None:
        f = tmp_path / "a.json"
        f.write_text(
            json.dumps(
                {
                    "suite": "security",
                    "agent": "injection",
                    "findings": [
                        {
                            "category": "security",
                            "file": "x.py",
                            "title": "SQL injection",
                            "severity": "high",
                        }
                    ],
                }
            )
        )
        findings, report = extract_agent_findings(
            f, role="injection", suite="security", provider="local"
        )
        assert report.status == "ok"
        assert len(findings) == 1
        assert findings[0].title == "SQL injection"

    def test_recovered_json_trailing_comma(self, tmp_path: Path) -> None:
        f = tmp_path / "a.json"
        f.write_text('{"findings": [{"category": "security", "file": "x.py", "title": "X",}]}')
        findings, report = extract_agent_findings(f, role="r", suite="security", provider="local")
        assert report.status == "recovered_json"
        assert len(findings) == 1

    def test_recovered_truncated(self, tmp_path: Path) -> None:
        f = tmp_path / "a.json"
        # Valid JSON object with trailing garbage after it. Tier-2 (lenient
        # loads) should fail because the extra text is not JSON at all,
        # and tier-3 should recover by brace-slicing.
        f.write_text(
            '{"findings": [{"category": "security", "file": "x.py", "title": "X"}]}\n'
            "stray tail garbage not valid json at all }}}\n"
        )
        findings, report = extract_agent_findings(f, role="r", suite="security", provider="local")
        # Either recovered_json or recovered_truncated is acceptable —
        # both mean the fallback chain did its job. The critical bit is
        # that we got the finding out.
        assert report.status in ("recovered_json", "recovered_truncated", "ok")
        assert len(findings) == 1

    def test_missing_file_falls_to_chat(self, tmp_path: Path) -> None:
        turns_path = tmp_path / "turns.jsonl"
        write_turns(
            [
                TurnRecord(
                    ts=1.0,
                    provider="local",
                    model="m",
                    response_prefix=(
                        '```json\n{"findings": [{"category": "security", '
                        '"file": "x.py", "title": "From chat"}]}\n```'
                    ),
                )
            ],
            turns_path,
        )
        missing = tmp_path / "nope.json"
        findings, report = extract_agent_findings(
            missing,
            role="r",
            suite="security",
            provider="local",
            turns_path=turns_path,
        )
        assert report.status == "recovered_chat"
        assert len(findings) == 1
        assert findings[0].title == "From chat"

    def test_failed_status_when_nothing_recoverable(self, tmp_path: Path) -> None:
        findings, report = extract_agent_findings(
            tmp_path / "nope.json",
            role="r",
            suite="security",
            provider="local",
            turns_path=None,
        )
        assert report.status == "failed"
        assert findings == []
        assert report.reason

    def test_scrape_ignores_unrelated_turns(self, tmp_path: Path) -> None:
        turns_path = tmp_path / "turns.jsonl"
        write_turns(
            [
                TurnRecord(
                    ts=1.0,
                    provider="local",
                    model="m",
                    response_prefix="Just some prose with no JSON.",
                ),
            ],
            turns_path,
        )
        data, src = _scrape_turns_for_findings(turns_path)
        assert data is None
        assert src == ""


class TestParseReport:
    def test_write_parse_report(self, tmp_path: Path) -> None:
        report = ExtractionReport(
            suite="security",
            agents=[
                AgentExtraction(
                    role="r",
                    status="ok",
                    source="/x",
                    finding_count=3,
                )
            ],
        )
        out = tmp_path / "parse.json"
        write_parse_report(report, out)
        data = json.loads(out.read_text())
        assert data["suite"] == "security"
        assert data["agents"][0]["status"] == "ok"


# ── runner.py integration tests ────────────────────────────────────


class TestEvalRunnerEndToEnd:
    def _bake_payload(self, title: str, file: str = "code.py") -> dict[str, Any]:
        return {
            "suite": "security",
            "agent": "lead",
            "findings": [
                {
                    "category": "security",
                    "file": file,
                    "line": 2,
                    "title": title,
                    "severity": "medium",
                    "description": "demo",
                    "confidence": 0.8,
                }
            ],
        }

    def test_full_run_produces_artifacts(
        self,
        repo: Path,
        tmp_path: Path,
    ) -> None:
        fake = FakeLauncher(self._bake_payload("SQL injection in foo"))

        def factory(provider, suite):  # type: ignore[no-untyped-def]
            return fake

        runner = EvalRunner(
            runs_root=tmp_path / "eval-runs",
            proxy_log_path=tmp_path / "never.jsonl",
            launcher_factory=factory,
            clock=lambda: 1_807_654_323.0,
        )
        result = runner.run(
            provider="local",
            suites=["security"],
            target=repo,
            run_id="test",
        )

        # Directory structure.
        assert result.run_dir.exists()
        assert (result.run_dir / "manifest.json").exists()
        assert (result.run_dir / "suites" / "security" / "findings.jsonl").exists()
        assert (result.run_dir / "suites" / "security" / "parse_report.json").exists()
        assert (result.run_dir / "suites" / "security" / "turns.jsonl").exists()

        # Manifest content.
        manifest = json.loads((result.run_dir / "manifest.json").read_text())
        assert manifest["provider"] == "local"
        assert manifest["suites"] == ["security"]
        assert manifest["target_sha"]
        assert manifest["snapshot_digest"]
        assert manifest["per_suite"]["security"]["finding_count"] >= 1

        # Runs index row appended.
        index = tmp_path / "eval-runs" / "runs.jsonl"
        assert index.exists()
        rows = [json.loads(line) for line in index.read_text().splitlines() if line.strip()]
        assert rows[-1]["run_id"] == result.run_id
        assert rows[-1]["totals"]["findings"] >= 1

        # Findings file contains exactly the baked finding.
        findings_path = result.run_dir / "suites" / "security" / "findings.jsonl"
        flines = findings_path.read_text().splitlines()
        assert len(flines) >= 1
        parsed = json.loads(flines[0])
        assert parsed["title"] == "SQL injection in foo"
        assert parsed["provider"] == "local"

    def test_snapshot_cleaned_up(self, repo: Path, tmp_path: Path) -> None:
        fake = FakeLauncher(self._bake_payload("x"))
        runner = EvalRunner(
            runs_root=tmp_path / "eval-runs",
            proxy_log_path=tmp_path / "p.jsonl",
            launcher_factory=lambda p, s: fake,
        )
        result = runner.run(
            provider="local",
            suites=["security"],
            target=repo,
            run_id="t",
        )
        worktree = result.run_dir / "worktrees" / "security"
        assert not worktree.exists()

    def test_keep_worktrees_preserves_snapshot(self, repo: Path, tmp_path: Path) -> None:
        fake = FakeLauncher(self._bake_payload("x"))
        runner = EvalRunner(
            runs_root=tmp_path / "eval-runs",
            proxy_log_path=tmp_path / "p.jsonl",
            launcher_factory=lambda p, s: fake,
        )
        result = runner.run(
            provider="local",
            suites=["security"],
            target=repo,
            run_id="t",
            keep_worktrees=True,
        )
        assert (result.run_dir / "worktrees" / "security").exists()

    def test_unknown_suite_rejected(self, repo: Path, tmp_path: Path) -> None:
        runner = EvalRunner(
            runs_root=tmp_path / "eval-runs",
            proxy_log_path=tmp_path / "p.jsonl",
        )
        with pytest.raises(ValueError):
            runner.run(
                provider="local",
                suites=["not-a-suite"],
                target=repo,
            )

    def test_multiple_suites_each_get_own_snapshot(self, repo: Path, tmp_path: Path) -> None:
        fake = FakeLauncher(self._bake_payload("x"))
        runner = EvalRunner(
            runs_root=tmp_path / "eval-runs",
            proxy_log_path=tmp_path / "p.jsonl",
            launcher_factory=lambda p, s: fake,
            clock=lambda: 1_807_654_323.0,
        )
        runner.run(
            provider="local",
            suites=["security", "a11y"],
            target=repo,
            run_id="multi",
        )
        assert len(fake.invocations) == 2
        # Each suite was given a distinct capture dir + findings file.
        dirs = {inv.capture_dir for inv in fake.invocations}
        assert len(dirs) == 2

    def test_provider_anthropic_still_runs(self, repo: Path, tmp_path: Path) -> None:
        fake = FakeLauncher(self._bake_payload("x"))
        runner = EvalRunner(
            runs_root=tmp_path / "eval-runs",
            proxy_log_path=tmp_path / "p.jsonl",
            launcher_factory=lambda p, s: fake,
        )
        result = runner.run(
            provider="anthropic",
            suites=["security"],
            target=repo,
            run_id="t",
        )
        manifest = json.loads((result.run_dir / "manifest.json").read_text())
        assert manifest["provider"] == "anthropic"
        assert manifest["model"].startswith("claude-opus-4-6")
