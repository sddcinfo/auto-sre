"""Tests for the autosre claude_hooks scripts.

Each hook is invoked as a module with ``runpy`` (rather than a subprocess)
so the tests stay fast. Stdin is stubbed via ``monkeypatch.setattr`` and
stdout/stderr are captured.
"""

from __future__ import annotations

import io
import json
import subprocess
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from autosre.claude_hooks import (
    post_commit_scan_update,
    posttooluse_audit,
    precompact_context,
    pretooluse_bash_guard,
    pretooluse_plan_review,
    stop_session_check,
    subagent_plan_context,
    telemetry_async,
    user_prompt_submit_branch_check,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(autouse=True)
def _isolate_xdg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))


def _stub_stdin(monkeypatch: pytest.MonkeyPatch, payload: str) -> None:
    """Replace sys.stdin with an in-memory StringIO containing ``payload``."""
    monkeypatch.setattr("sys.stdin", io.StringIO(payload))


class TestTelemetryStub:
    def test_returns_continue(
        self,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _stub_stdin(monkeypatch, '{"tool_name": "Bash"}')
        rc = telemetry_async.main()
        assert rc == 0
        out = capsys.readouterr().out
        assert json.loads(out) == {"result": "continue"}


class TestPostCommitStub:
    def test_returns_continue(
        self,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _stub_stdin(monkeypatch, "{}")
        rc = post_commit_scan_update.main()
        assert rc == 0
        assert json.loads(capsys.readouterr().out) == {"result": "continue"}


class TestPosttooluseAudit:
    def test_writes_audit_log(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from autosre import paths

        _stub_stdin(
            monkeypatch,
            json.dumps({"tool_name": "Bash", "tool_input": {"command": "ls -la"}}),
        )
        rc = posttooluse_audit.main()
        assert rc == 0
        assert json.loads(capsys.readouterr().out) == {"result": "continue"}

        log = paths.hook_audit_log().read_text()
        assert "ls -la" in log

    def test_no_command_skips_log(
        self,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from autosre import paths

        _stub_stdin(monkeypatch, json.dumps({"tool_name": "Read", "tool_input": {}}))
        rc = posttooluse_audit.main()
        assert rc == 0
        assert json.loads(capsys.readouterr().out) == {"result": "continue"}
        assert not paths.hook_audit_log().exists()

    def test_truncates_long_commands(
        self,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from autosre import paths

        big_command = "echo " + "a" * 500
        _stub_stdin(
            monkeypatch,
            json.dumps({"tool_name": "Bash", "tool_input": {"command": big_command}}),
        )
        posttooluse_audit.main()
        capsys.readouterr()
        log = paths.hook_audit_log().read_text()
        # Command was truncated to 200 chars + "..."
        assert "..." in log


class TestUserPromptBranchCheck:
    @patch("autosre.claude_hooks.user_prompt_submit_branch_check._get_current_branch")
    def test_on_feature_branch_emits_message(
        self,
        mock_branch: MagicMock,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        mock_branch.return_value = "feature/xyz"
        _stub_stdin(monkeypatch, "{}")
        rc = user_prompt_submit_branch_check.main()
        assert rc == 0
        out = json.loads(capsys.readouterr().out)
        assert out["result"] == "continue"
        assert "feature branch" in out["message"]

    @patch("autosre.claude_hooks.user_prompt_submit_branch_check._get_current_branch")
    def test_on_dev_is_silent(
        self,
        mock_branch: MagicMock,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        mock_branch.return_value = "dev"
        _stub_stdin(monkeypatch, "{}")
        user_prompt_submit_branch_check.main()
        out = json.loads(capsys.readouterr().out)
        assert out == {"result": "continue"}

    @patch("autosre.claude_hooks.user_prompt_submit_branch_check._get_current_branch")
    def test_marker_cooldown(
        self,
        mock_branch: MagicMock,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """After a warning, a second call within an hour should stay silent."""
        mock_branch.return_value = "feature/abc"
        _stub_stdin(monkeypatch, "{}")
        user_prompt_submit_branch_check.main()
        first = capsys.readouterr().out
        assert "feature branch" in first

        _stub_stdin(monkeypatch, "{}")
        user_prompt_submit_branch_check.main()
        second = json.loads(capsys.readouterr().out)
        assert second == {"result": "continue"}


class TestStopSessionCheck:
    @patch("autosre.claude_hooks.stop_session_check.subprocess.run")
    def test_forwards_backend_stdout(
        self,
        mock_run: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=["autosre", "hooks-backend", "stop-check"],
            returncode=0,
            stdout='{"result": "continue", "message": "all clean"}',
            stderr="",
        )
        rc = stop_session_check.main()
        assert rc == 0
        out = capsys.readouterr().out
        assert '"result": "continue"' in out
        assert "all clean" in out

    @patch("autosre.claude_hooks.stop_session_check.subprocess.run")
    def test_timeout_fails_open(
        self,
        mock_run: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="autosre", timeout=10)
        stop_session_check.main()
        out = json.loads(capsys.readouterr().out)
        assert out["result"] == "continue"
        assert "timed out" in out.get("message", "")


class TestPreToolUseBashGuard:
    @patch("autosre.claude_hooks.pretooluse_bash_guard.subprocess.run")
    def test_forwards_deny_from_backend(
        self,
        mock_run: MagicMock,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        backend_output = json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": "nope",
                },
            },
        )
        mock_run.return_value = subprocess.CompletedProcess(
            args=["autosre", "hooks-backend", "guard"],
            returncode=0,
            stdout=backend_output,
            stderr="",
        )
        _stub_stdin(monkeypatch, '{"tool_name":"Bash","tool_input":{"command":"rm -rf /"}}')
        pretooluse_bash_guard.main()
        out = json.loads(capsys.readouterr().out)
        assert out["hookSpecificOutput"]["permissionDecision"] == "deny"
        assert out["hookSpecificOutput"]["permissionDecisionReason"] == "nope"

    @patch("autosre.claude_hooks.pretooluse_bash_guard.subprocess.run")
    def test_timeout_fails_closed(
        self,
        mock_run: MagicMock,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="autosre", timeout=10)
        _stub_stdin(monkeypatch, "{}")
        pretooluse_bash_guard.main()
        out = json.loads(capsys.readouterr().out)
        assert out["hookSpecificOutput"]["permissionDecision"] == "deny"
        assert "timed out" in out["hookSpecificOutput"]["permissionDecisionReason"]

    @patch(
        "autosre.claude_hooks.pretooluse_bash_guard.subprocess.run",
        side_effect=FileNotFoundError("no autosre"),
    )
    def test_missing_binary_fails_closed(
        self,
        _mock_run: MagicMock,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _stub_stdin(monkeypatch, "{}")
        pretooluse_bash_guard.main()
        out = json.loads(capsys.readouterr().out)
        assert out["hookSpecificOutput"]["permissionDecision"] == "deny"


class TestPreCompactContext:
    def test_injects_claude_md_and_git_state(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "CLAUDE.md").write_text(
            "# Project\n\nThis is my project.\n",
        )
        # Initialize git
        subprocess.run(
            ["git", "init", "-q", "-b", "dev"],
            cwd=repo,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.email", "t@t"],
            cwd=repo,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "t"],
            cwd=repo,
            check=True,
            capture_output=True,
        )

        monkeypatch.chdir(repo)
        _stub_stdin(monkeypatch, "{}")

        rc = precompact_context.main()
        assert rc == 0
        out = json.loads(capsys.readouterr().out)
        assert "systemMessage" in out
        assert "CLAUDE.md" in out["systemMessage"]
        assert "This is my project" in out["systemMessage"]
        assert "Branch: dev" in out["systemMessage"]

    def test_empty_project_returns_ok(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # No CLAUDE.md, no git — should return {"result": "ok"}
        monkeypatch.chdir(tmp_path)
        _stub_stdin(monkeypatch, "{}")
        precompact_context.main()
        out = json.loads(capsys.readouterr().out)
        assert out == {"result": "ok"}


class TestSubagentPlanContext:
    def test_emits_subagent_context(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        (tmp_path / "CLAUDE.md").write_text("# Root\n")
        (tmp_path / "src").mkdir()
        (tmp_path / "README.md").write_text("")
        monkeypatch.chdir(tmp_path)
        _stub_stdin(monkeypatch, "{}")

        rc = subagent_plan_context.main()
        assert rc == 0
        out = json.loads(capsys.readouterr().out)
        assert out["hookSpecificOutput"]["hookEventName"] == "SubagentStart"
        context = out["hookSpecificOutput"]["additionalContext"]
        assert "PROJECT CONTEXT" in context
        assert "CLAUDE.md" in context
        assert "Structure" in context


class TestPlanReviewHook:
    @patch("autosre.claude_hooks.pretooluse_plan_review.subprocess.run")
    def test_no_plan_path_allows(
        self,
        _mock_run: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _stub_stdin(
            monkeypatch,
            json.dumps({"session_id": "s1", "tool_input": {}}),
        )
        pretooluse_plan_review.main()
        out = json.loads(capsys.readouterr().out)
        assert out["hookSpecificOutput"]["permissionDecision"] == "allow"

    @patch("autosre.claude_hooks.pretooluse_plan_review.subprocess.run")
    def test_blocking_review_translates_to_deny(
        self,
        mock_run: MagicMock,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        plan = tmp_path / "plan.md"
        plan.write_text("# Plan\n\nSome content\n")

        review_output = json.dumps(
            {
                "findings": [
                    {
                        "severity": "P0",
                        "title": "SQL injection",
                        "description": "user input unsanitized",
                        "recommendation": "parameterized queries",
                    },
                ],
                "questions": [],
                "provider": "codex",
                "elapsed_seconds": 5.0,
                "iteration": 1,
                "blocking": True,
                "p2_only": False,
                "attempts": [],
            },
        )
        mock_run.return_value = subprocess.CompletedProcess(
            args=["autosre", "review", "plan"],
            returncode=1,  # 1 = blocking findings
            stdout=review_output,
            stderr="",
        )

        _stub_stdin(
            monkeypatch,
            json.dumps(
                {
                    "session_id": "s1",
                    "tool_input": {"planFilePath": str(plan)},
                },
            ),
        )
        pretooluse_plan_review.main()
        out = json.loads(capsys.readouterr().out)
        assert out["hookSpecificOutput"]["permissionDecision"] == "deny"
        assert "codex" in out["hookSpecificOutput"]["permissionDecisionReason"]
        assert "P0" in out["hookSpecificOutput"]["additionalContext"]
        assert "SQL injection" in out["hookSpecificOutput"]["additionalContext"]

    @patch("autosre.claude_hooks.pretooluse_plan_review.subprocess.run")
    def test_clean_review_translates_to_allow(
        self,
        mock_run: MagicMock,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        plan = tmp_path / "plan.md"
        plan.write_text("# Plan\n")
        review_output = json.dumps(
            {
                "findings": [],
                "questions": [],
                "provider": "codex",
                "elapsed_seconds": 5.0,
                "iteration": 1,
                "blocking": False,
                "p2_only": False,
                "attempts": [],
            },
        )
        mock_run.return_value = subprocess.CompletedProcess(
            args=["autosre", "review", "plan"],
            returncode=0,
            stdout=review_output,
            stderr="",
        )
        _stub_stdin(
            monkeypatch,
            json.dumps({"session_id": "s1", "tool_input": {"planFilePath": str(plan)}}),
        )
        pretooluse_plan_review.main()
        out = json.loads(capsys.readouterr().out)
        assert out["hookSpecificOutput"]["permissionDecision"] == "allow"
        assert "codex" in out["hookSpecificOutput"]["additionalContext"]

    @patch("autosre.claude_hooks.pretooluse_plan_review.subprocess.run")
    def test_p2_only_translates_to_allow_with_advisory(
        self,
        mock_run: MagicMock,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        plan = tmp_path / "plan.md"
        plan.write_text("# Plan\n")
        review_output = json.dumps(
            {
                "findings": [
                    {
                        "severity": "P2",
                        "title": "minor nit",
                        "description": "d",
                        "recommendation": "r",
                    },
                ],
                "questions": [],
                "provider": "codex",
                "elapsed_seconds": 5.0,
                "iteration": 1,
                "blocking": False,
                "p2_only": True,
                "attempts": [],
            },
        )
        mock_run.return_value = subprocess.CompletedProcess(
            args=["autosre", "review", "plan"],
            returncode=0,
            stdout=review_output,
            stderr="",
        )
        _stub_stdin(
            monkeypatch,
            json.dumps({"session_id": "s1", "tool_input": {"planFilePath": str(plan)}}),
        )
        pretooluse_plan_review.main()
        out = json.loads(capsys.readouterr().out)
        assert out["hookSpecificOutput"]["permissionDecision"] == "allow"
        assert "advisory" in out["hookSpecificOutput"]["additionalContext"]

    @patch("autosre.claude_hooks.pretooluse_plan_review.subprocess.run")
    def test_chain_failure_fails_open(
        self,
        mock_run: MagicMock,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        plan = tmp_path / "plan.md"
        plan.write_text("# Plan\n")
        mock_run.return_value = subprocess.CompletedProcess(
            args=["autosre", "review", "plan"],
            returncode=2,
            stdout="",
            stderr="all providers failed",
        )
        _stub_stdin(
            monkeypatch,
            json.dumps({"session_id": "s1", "tool_input": {"planFilePath": str(plan)}}),
        )
        pretooluse_plan_review.main()
        out = json.loads(capsys.readouterr().out)
        assert out["hookSpecificOutput"]["permissionDecision"] == "allow"
        assert "all providers failed" in out["hookSpecificOutput"]["additionalContext"]
