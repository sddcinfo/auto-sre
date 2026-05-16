"""Tests for the ``autosre eval`` CLI command group."""

from __future__ import annotations

import json
import subprocess
from typing import TYPE_CHECKING

import pytest
from click.testing import CliRunner

from autosre.cli import cli

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


class TestEvalSuitesCommand:
    def test_list_prints_all_ten(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["eval", "suites"])
        assert result.exit_code == 0
        for name in [
            "security",
            "leakage",
            "quality",
            "duplication",
            "dead-code",
            "tech-debt",
            "coverage",
            "ui-consistency",
            "a11y",
            "i18n",
        ]:
            assert name in result.output

    def test_show_one(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["eval", "suites", "--show", "security"])
        assert result.exit_code == 0
        assert "security" in result.output
        assert "agents" in result.output

    def test_show_unknown_suite_fails(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["eval", "suites", "--show", "nope"])
        assert result.exit_code != 0


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(cwd), *args],
        check=True,
        capture_output=True,
    )


@pytest.fixture()
def fake_run(tmp_path: Path) -> tuple[Path, Path]:
    """Create two pre-baked run directories so compare has real input."""
    root = tmp_path / "eval-runs"
    root.mkdir()

    for tag, provider in (("local-baseline", "local"), ("anthropic-baseline", "anthropic")):
        run_dir = root / tag
        run_dir.mkdir()
        manifest = {
            "run_id": tag,
            "provider": provider,
            "target_repo": "/fake/repo",
            "target_sha": "sha1",
            "snapshot_digest": "deadbeef",
            "suites": ["security"],
            "model": "m",
        }
        (run_dir / "manifest.json").write_text(json.dumps(manifest))
        suite_dir = run_dir / "suites" / "security"
        suite_dir.mkdir(parents=True)
        # One identical finding so `both` > 0.
        (suite_dir / "findings.jsonl").write_text(
            json.dumps(
                {
                    "id": "shared-id",
                    "suite": "security",
                    "category": "security",
                    "severity": "high",
                    "file": "src/app.py",
                    "line": 10,
                    "title": "SQL injection in login",
                    "description": "user input flows into a concatenated sql query",
                    "evidence": "",
                    "confidence": 0.9,
                    "agent": "injection",
                    "provider": provider,
                }
            )
            + "\n"
        )
        (suite_dir / "parse_report.json").write_text(
            json.dumps({"suite": "security", "agents": []})
        )
    return root / "local-baseline", root / "anthropic-baseline"


class TestEvalCompareCommand:
    def test_end_to_end(
        self,
        runner: CliRunner,
        fake_run: tuple[Path, Path],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        a, b = fake_run
        # Redirect EVAL_RUNS_ROOT to the fake root for resolution.
        monkeypatch.setattr("autosre.eval.runner.EVAL_RUNS_ROOT", a.parent, raising=False)

        result = runner.invoke(
            cli,
            [
                "eval",
                "compare",
                str(a),
                str(b),
                "--no-judge",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Compare written" in result.output
        assert "compare.md" in result.output
        assert "compare.json" in result.output

        compares = list((a.parent / "compares").glob("*"))
        assert compares
        compare_json = compares[0] / "compare.json"
        assert compare_json.exists()
        data = json.loads(compare_json.read_text())
        assert data["per_suite"]["security"]["both"] == 1

    def test_refuses_same_provider(
        self,
        runner: CliRunner,
        fake_run: tuple[Path, Path],
        tmp_path: Path,
    ) -> None:
        a, b = fake_run
        # Rewrite b's manifest to use the same provider as a.
        manifest = json.loads((b / "manifest.json").read_text())
        manifest["provider"] = "local"
        (b / "manifest.json").write_text(json.dumps(manifest))
        result = runner.invoke(
            cli,
            ["eval", "compare", str(a), str(b), "--no-judge"],
        )
        assert result.exit_code != 0
        assert "provider" in result.output.lower()


class TestEvalListShow:
    def test_list_with_no_runs(
        self,
        runner: CliRunner,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        empty = tmp_path / "empty"
        monkeypatch.setattr("autosre.eval.runner.EVAL_RUNS_ROOT", empty, raising=False)
        result = runner.invoke(cli, ["eval", "list"])
        assert result.exit_code == 0
        assert "no runs" in result.output.lower()

    def test_show_missing_run_fails(
        self,
        runner: CliRunner,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            "autosre.eval.runner.EVAL_RUNS_ROOT",
            tmp_path / "nope",
            raising=False,
        )
        result = runner.invoke(cli, ["eval", "show", "missing-run"])
        assert result.exit_code != 0
