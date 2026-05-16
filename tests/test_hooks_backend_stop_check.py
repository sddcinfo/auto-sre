"""Tests for autosre.hooks_backend.stop_check."""

from __future__ import annotations

import json
import subprocess
from typing import TYPE_CHECKING

from autosre.hooks_backend import stop_check

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def _make_git_repo(path: Path) -> None:
    """Initialize a minimal git repo at ``path``."""
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "init", "-q", "-b", "main"],
        cwd=path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@test.local"],
        cwd=path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "test"],
        cwd=path,
        check=True,
        capture_output=True,
    )


class TestDetectRepoType:
    def test_cloudflare_worker(self, tmp_path: Path) -> None:
        (tmp_path / "wrangler.toml").write_text("")
        assert stop_check._detect_repo_type(tmp_path) == "cloudflare-worker"

    def test_python(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text("")
        assert stop_check._detect_repo_type(tmp_path) == "python"

    def test_node(self, tmp_path: Path) -> None:
        (tmp_path / "package.json").write_text("{}")
        assert stop_check._detect_repo_type(tmp_path) == "node"

    def test_unknown(self, tmp_path: Path) -> None:
        assert stop_check._detect_repo_type(tmp_path) == "unknown"

    def test_cloudflare_in_apps(self, tmp_path: Path) -> None:
        apps = tmp_path / "apps" / "worker"
        apps.mkdir(parents=True)
        (apps / "wrangler.toml").write_text("")
        assert stop_check._detect_repo_type(tmp_path) == "cloudflare-worker"


class TestDetectHints:
    def test_cloudflare_deploy_hint(self, tmp_path: Path) -> None:
        assert stop_check._detect_deploy_hint(tmp_path, "cloudflare-worker") == "wrangler deploy"

    def test_no_deploy_hint_for_python(self, tmp_path: Path) -> None:
        assert stop_check._detect_deploy_hint(tmp_path, "python") is None

    def test_python_pytest_hint(self, tmp_path: Path) -> None:
        (tmp_path / "tests").mkdir()
        assert stop_check._detect_test_hint(tmp_path, "python") == "pytest"

    def test_node_pnpm_test_hint(self, tmp_path: Path) -> None:
        (tmp_path / "package.json").write_text("{}")
        assert stop_check._detect_test_hint(tmp_path, "node") == "pnpm test"

    def test_playwright_takes_priority(self, tmp_path: Path) -> None:
        (tmp_path / "playwright.config.ts").write_text("")
        assert stop_check._detect_test_hint(tmp_path, "node") == "pnpm playwright test"


class TestHasRecentLocalPlan:
    def test_no_plans_dir(self, tmp_path: Path) -> None:
        assert stop_check._has_recent_local_plan(tmp_path) is None

    def test_plans_dir_empty(self, tmp_path: Path) -> None:
        (tmp_path / ".claude" / "plans").mkdir(parents=True)
        assert stop_check._has_recent_local_plan(tmp_path) is None

    def test_recent_plan_found(self, tmp_path: Path) -> None:
        plans = tmp_path / ".claude" / "plans"
        plans.mkdir(parents=True)
        plan = plans / "active-plan.md"
        plan.write_text("# Plan")
        # Freshly written file = mtime is now, definitely within 30 min
        result = stop_check._has_recent_local_plan(tmp_path)
        assert result is not None
        assert "active-plan.md" in result["text"]
        assert result["status"] == "info"

    def test_old_plan_ignored(self, tmp_path: Path) -> None:
        import os
        import time

        plans = tmp_path / ".claude" / "plans"
        plans.mkdir(parents=True)
        plan = plans / "old-plan.md"
        plan.write_text("# Plan")
        # Set mtime to 2 hours ago
        old = time.time() - 7200
        os.utime(plan, (old, old))
        assert stop_check._has_recent_local_plan(tmp_path) is None


class TestGitHelpers:
    def test_detect_repo_in_real_git_repo(self, tmp_path: Path) -> None:
        repo = tmp_path / "my-repo"
        _make_git_repo(repo)
        name, path = stop_check._detect_repo(str(repo))
        assert name == "my-repo"
        assert path == repo

    def test_detect_repo_outside_git(self, tmp_path: Path) -> None:
        name, path = stop_check._detect_repo(str(tmp_path))
        assert name is None
        assert path is None

    def test_uncommitted_count_clean(self, tmp_path: Path) -> None:
        repo = tmp_path / "clean"
        _make_git_repo(repo)
        assert stop_check._uncommitted_count(str(repo)) == 0

    def test_uncommitted_count_with_changes(self, tmp_path: Path) -> None:
        repo = tmp_path / "dirty"
        _make_git_repo(repo)
        (repo / "new-file.txt").write_text("hello")
        # new-file.txt is untracked — counts as 1
        assert stop_check._uncommitted_count(str(repo)) == 1


class TestBuildChecklist:
    def test_clean_repo_has_committed_item_done(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        _make_git_repo(repo)
        (repo / "pyproject.toml").write_text("[project]\nname='x'\n")
        subprocess.run(["git", "add", "-A"], cwd=repo, check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-q", "-m", "init"],
            cwd=repo,
            check=True,
            capture_output=True,
        )

        data = stop_check.build_checklist("repo", repo)
        assert data["repo_name"] == "repo"
        assert data["repo_type"] == "python"
        committed = [i for i in data["items"] if "committed" in i["text"]]
        assert any(i["status"] == "done" for i in committed)

    def test_dirty_repo_has_uncommitted_todo(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        _make_git_repo(repo)
        (repo / "new.txt").write_text("hi")

        data = stop_check.build_checklist("repo", repo)
        todos = [i for i in data["items"] if i["status"] == "todo"]
        assert any("uncommitted change" in i["text"] for i in todos)


class TestFormatChecklist:
    def test_format_all_done(self) -> None:
        data = {
            "repo_name": "myrepo",
            "repo_type": "python",
            "branch": "main",
            "items": [
                {"status": "done", "text": "All changes committed"},
                {"status": "done", "text": "Branch up to date"},
            ],
            "actions": [],
        }
        out = stop_check.format_checklist(data)
        assert "myrepo" in out
        assert "python" in out
        assert "[x]" in out
        assert "STOP" not in out  # no blocking items

    def test_format_with_blocking(self) -> None:
        data = {
            "repo_name": "myrepo",
            "repo_type": "python",
            "branch": "main",
            "items": [
                {"status": "todo", "text": "Commit first"},
            ],
            "actions": ["Commit outstanding changes"],
        }
        out = stop_check.format_checklist(data)
        assert "STOP" in out
        assert "REQUIRED" in out
        assert "Commit outstanding changes" in out


class TestStopCheckCmd:
    def test_non_git_dir_returns_continue(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.chdir(tmp_path)
        from click.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(stop_check.stop_check_cmd)
        assert result.exit_code == 0
        output = json.loads(result.stdout)
        assert output == {"result": "continue"}

    def test_clean_git_repo(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo = tmp_path / "repo"
        _make_git_repo(repo)
        (repo / "pyproject.toml").write_text("[project]\nname='x'\n")
        subprocess.run(["git", "add", "-A"], cwd=repo, check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-q", "-m", "init"],
            cwd=repo,
            check=True,
            capture_output=True,
        )
        monkeypatch.chdir(repo)

        from click.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(stop_check.stop_check_cmd)
        assert result.exit_code == 0
        output = json.loads(result.stdout)
        # Clean repo with no upstream → branch reports "no upstream" which is
        # still a "todo" item, so result is "block", not "continue".
        assert output["result"] in ("continue", "block")
        # Either way, the message/reason should mention the repo name.
        msg = output.get("message", output.get("reason", ""))
        assert "repo" in msg
