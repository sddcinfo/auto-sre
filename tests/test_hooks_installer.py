"""Tests for autosre.hooks_installer — the installer for bare-claude hooks.

Covers:

- Install into an empty ~/.claude/settings.json (fresh install).
- Install into a settings file with existing user hooks — user's entries
  must be preserved; our entries must be appended.
- Idempotent re-install — second call reports everything as skipped.
- Install with a different ``python`` — strip old tracked entries, add
  fresh ones.
- Uninstall — removes only our entries, leaves user's hooks intact.
- Status — reports installed + drift detection.
- The ``autosre hooks`` CLI group happy paths.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest
from click.testing import CliRunner

from autosre import cli as autosre_cli
from autosre import hooks_installer as installer

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def settings_path(tmp_path: Path) -> Path:
    return tmp_path / "settings.json"


@pytest.fixture
def sidecar_path(tmp_path: Path) -> Path:
    return tmp_path / ".autosre-hooks-installed.json"


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


class TestInstallFresh:
    def test_creates_settings_from_scratch(
        self,
        settings_path: Path,
        sidecar_path: Path,
    ) -> None:
        result = installer.install(
            settings_path=settings_path,
            sidecar_path=sidecar_path,
        )
        assert len(result["added"]) == 13
        assert len(result["skipped"]) == 0
        assert settings_path.exists()
        assert sidecar_path.exists()

        data = _load_json(settings_path)
        # Event keys live under a top-level "hooks" wrapper per the Claude
        # Code settings.json schema — entries at the root are silently ignored.
        assert "hooks" in data
        hooks_root = data["hooks"]
        assert "PreToolUse" in hooks_root
        assert "PostToolUse" in hooks_root
        assert "Stop" in hooks_root
        assert "UserPromptSubmit" in hooks_root
        assert "PreCompact" in hooks_root
        assert "SubagentStart" in hooks_root

    def test_sidecar_records_entries(
        self,
        settings_path: Path,
        sidecar_path: Path,
    ) -> None:
        installer.install(
            settings_path=settings_path,
            sidecar_path=sidecar_path,
        )
        sidecar = _load_json(sidecar_path)
        assert sidecar["version"] == 2
        assert len(sidecar["entries"]) == 13

    def test_hook_commands_use_autosre_cli(
        self,
        settings_path: Path,
        sidecar_path: Path,
    ) -> None:
        """Commands must invoke ``autosre hooks run <module>``, with no
        absolute Python path — the ``autosre`` entrypoint resolves via $PATH."""
        installer.install(settings_path=settings_path, sidecar_path=sidecar_path)
        data = _load_json(settings_path)
        for block in data["hooks"]["PreToolUse"]:
            for hook in block["hooks"]:
                assert hook["command"].startswith("autosre hooks run ")
                assert "python" not in hook["command"]

    def test_exit_plan_mode_matcher_wired_to_review_hook(
        self,
        settings_path: Path,
        sidecar_path: Path,
    ) -> None:
        installer.install(settings_path=settings_path, sidecar_path=sidecar_path)
        data = _load_json(settings_path)
        pre = data["hooks"]["PreToolUse"]
        exit_plan = next(b for b in pre if b.get("matcher") == "ExitPlanMode")
        cmd = exit_plan["hooks"][0]["command"]
        assert "pretooluse_plan_review" in cmd

    def test_bash_matcher_wired_to_guard(
        self,
        settings_path: Path,
        sidecar_path: Path,
    ) -> None:
        installer.install(settings_path=settings_path, sidecar_path=sidecar_path)
        data = _load_json(settings_path)
        pre = data["hooks"]["PreToolUse"]
        bash = next(b for b in pre if b.get("matcher") == "Bash")
        cmd = bash["hooks"][0]["command"]
        assert "pretooluse_bash_guard" in cmd


class TestInstallPreservesUserHooks:
    def test_existing_user_hook_preserved(
        self,
        settings_path: Path,
        sidecar_path: Path,
    ) -> None:
        _write_json(
            settings_path,
            {
                "permissions": {"allow": ["Read", "Glob"]},
                "enabledPlugins": {"my-plugin": True},
                "hooks": {
                    "PreToolUse": [
                        {
                            "matcher": "Write|Edit",
                            "hooks": [
                                {"type": "command", "command": "/user/my-hook.sh"},
                            ],
                        },
                    ],
                },
            },
        )

        installer.install(settings_path=settings_path, sidecar_path=sidecar_path)
        data = _load_json(settings_path)

        # User permissions block untouched
        assert data["permissions"] == {"allow": ["Read", "Glob"]}
        # User plugin block untouched
        assert data["enabledPlugins"] == {"my-plugin": True}
        # User's Write|Edit hook still present
        pre = data["hooks"]["PreToolUse"]
        write_edit = [b for b in pre if b.get("matcher") == "Write|Edit"]
        assert len(write_edit) == 1
        assert write_edit[0]["hooks"][0]["command"] == "/user/my-hook.sh"
        # Our Bash and ExitPlanMode matchers are also there
        matchers = {b.get("matcher") for b in pre}
        assert "Bash" in matchers
        assert "ExitPlanMode" in matchers
        assert "Write|Edit" in matchers

    def test_existing_bash_user_hook_is_appended_not_replaced(
        self,
        settings_path: Path,
        sidecar_path: Path,
    ) -> None:
        """If the user already has a Bash matcher block, our hook gets
        appended to it, not dropped or clobbered."""
        _write_json(
            settings_path,
            {
                "hooks": {
                    "PostToolUse": [
                        {
                            "matcher": "Bash",
                            "hooks": [
                                {"type": "command", "command": "/user/post-bash.sh"},
                            ],
                        },
                    ],
                },
            },
        )

        installer.install(settings_path=settings_path, sidecar_path=sidecar_path)
        data = _load_json(settings_path)

        post = data["hooks"]["PostToolUse"]
        bash_blocks = [b for b in post if b.get("matcher") == "Bash"]
        assert len(bash_blocks) == 1  # Merged into the same block
        commands = [h["command"] for h in bash_blocks[0]["hooks"]]
        assert "/user/post-bash.sh" in commands
        assert any("posttooluse_audit" in c for c in commands)
        assert any("telemetry_async" in c for c in commands)


class TestInstallIdempotent:
    def test_second_install_is_noop(
        self,
        settings_path: Path,
        sidecar_path: Path,
    ) -> None:
        installer.install(settings_path=settings_path, sidecar_path=sidecar_path)
        first_mtime = settings_path.stat().st_mtime_ns
        first_content = settings_path.read_text()

        # Sleep a hair to ensure mtime would differ if a write happened
        import time

        time.sleep(0.01)
        result = installer.install(settings_path=settings_path, sidecar_path=sidecar_path)
        assert len(result["added"]) == 0
        assert len(result["skipped"]) == 13
        assert settings_path.read_text() == first_content
        # settings.json was NOT rewritten (mtime unchanged)
        assert settings_path.stat().st_mtime_ns == first_mtime

    def test_install_recovers_drifted_entries(
        self,
        settings_path: Path,
        sidecar_path: Path,
    ) -> None:
        """If the user manually pruned one of our entries but left the
        sidecar intact, the next install should re-add the missing entry."""
        installer.install(settings_path=settings_path, sidecar_path=sidecar_path)
        data = _load_json(settings_path)
        # Manually remove the UserPromptSubmit block
        data["hooks"].pop("UserPromptSubmit", None)
        _write_json(settings_path, data)

        result = installer.install(settings_path=settings_path, sidecar_path=sidecar_path)
        assert len(result["added"]) == 1
        data2 = _load_json(settings_path)
        assert "UserPromptSubmit" in data2["hooks"]


def _all_hook_commands(settings: dict[str, Any]) -> list[str]:
    commands: list[str] = []
    hooks_root = settings.get("hooks", {})
    if not isinstance(hooks_root, dict):
        return commands
    for event_key in (
        "PreToolUse",
        "PostToolUse",
        "Stop",
        "UserPromptSubmit",
        "PreCompact",
        "SubagentStart",
    ):
        for block in hooks_root.get(event_key, []):
            for hook in block.get("hooks", []):
                if isinstance(hook, dict) and "command" in hook:
                    commands.append(hook["command"])
    return commands


class TestUninstall:
    def test_removes_all_tracked_entries(
        self,
        settings_path: Path,
        sidecar_path: Path,
    ) -> None:
        installer.install(settings_path=settings_path, sidecar_path=sidecar_path)
        result = installer.uninstall(settings_path=settings_path, sidecar_path=sidecar_path)
        assert len(result["removed"]) == 13

        data = _load_json(settings_path)
        commands = _all_hook_commands(data)
        # None of our commands remain
        assert not any("autosre hooks run" in c for c in commands)
        # Sidecar is gone
        assert not sidecar_path.exists()

    def test_preserves_user_hooks(
        self,
        settings_path: Path,
        sidecar_path: Path,
    ) -> None:
        _write_json(
            settings_path,
            {
                "permissions": {"allow": ["Read"]},
                "hooks": {
                    "PreToolUse": [
                        {
                            "matcher": "Write",
                            "hooks": [
                                {"type": "command", "command": "/user/hook.sh"},
                            ],
                        },
                    ],
                },
            },
        )
        installer.install(settings_path=settings_path, sidecar_path=sidecar_path)
        installer.uninstall(settings_path=settings_path, sidecar_path=sidecar_path)

        data = _load_json(settings_path)
        assert data["permissions"] == {"allow": ["Read"]}
        pre = data.get("hooks", {}).get("PreToolUse", [])
        assert any(
            b.get("matcher") == "Write"
            and any(h.get("command") == "/user/hook.sh" for h in b.get("hooks", []))
            for b in pre
        )

    def test_uninstall_user_and_ours_in_same_block(
        self,
        settings_path: Path,
        sidecar_path: Path,
    ) -> None:
        """If the user had a Bash PostToolUse hook and our install merged
        ours into the same block, uninstall must leave the user's hook."""
        _write_json(
            settings_path,
            {
                "hooks": {
                    "PostToolUse": [
                        {
                            "matcher": "Bash",
                            "hooks": [
                                {"type": "command", "command": "/user/audit.sh"},
                            ],
                        },
                    ],
                },
            },
        )
        installer.install(settings_path=settings_path, sidecar_path=sidecar_path)
        installer.uninstall(settings_path=settings_path, sidecar_path=sidecar_path)

        data = _load_json(settings_path)
        # The Bash block should still exist with the user's hook
        post = data.get("hooks", {}).get("PostToolUse", [])
        bash_blocks = [b for b in post if b.get("matcher") == "Bash"]
        assert len(bash_blocks) == 1
        commands = [h["command"] for h in bash_blocks[0]["hooks"]]
        assert commands == ["/user/audit.sh"]

    def test_uninstall_no_sidecar_is_noop(
        self,
        settings_path: Path,
        sidecar_path: Path,
    ) -> None:
        result = installer.uninstall(settings_path=settings_path, sidecar_path=sidecar_path)
        assert "note" in result


class TestStatus:
    def test_reports_not_installed_initially(
        self,
        settings_path: Path,
        sidecar_path: Path,
    ) -> None:
        result = installer.status(settings_path=settings_path, sidecar_path=sidecar_path)
        assert result["installed"] is False
        assert result["entries"] == []
        assert result["drift"] == []

    def test_reports_installed(
        self,
        settings_path: Path,
        sidecar_path: Path,
    ) -> None:
        installer.install(settings_path=settings_path, sidecar_path=sidecar_path)
        result = installer.status(settings_path=settings_path, sidecar_path=sidecar_path)
        assert result["installed"] is True
        assert len(result["entries"]) == 13
        assert result["drift"] == []

    def test_detects_drift(
        self,
        settings_path: Path,
        sidecar_path: Path,
    ) -> None:
        installer.install(settings_path=settings_path, sidecar_path=sidecar_path)
        data = _load_json(settings_path)
        data["hooks"].pop("UserPromptSubmit", None)  # user deleted this by hand
        _write_json(settings_path, data)

        result = installer.status(settings_path=settings_path, sidecar_path=sidecar_path)
        assert len(result["drift"]) == 1
        assert result["drift"][0]["event"] == "UserPromptSubmit"


class TestCli:
    def test_hooks_install_cli(
        self,
        settings_path: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        monkeypatch.setenv("HOME", str(fake_home))

        runner = CliRunner()
        result = runner.invoke(
            autosre_cli.cli,
            ["hooks", "install", "--settings", str(settings_path)],
        )
        assert result.exit_code == 0, result.output
        assert "Installed autosre hooks" in result.output
        assert settings_path.exists()
        data = _load_json(settings_path)
        assert "PreToolUse" in data["hooks"]

    def test_hooks_install_then_status_cli(
        self,
        settings_path: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        monkeypatch.setenv("HOME", str(fake_home))

        runner = CliRunner()
        r1 = runner.invoke(
            autosre_cli.cli,
            ["hooks", "install", "--settings", str(settings_path)],
        )
        assert r1.exit_code == 0

        r2 = runner.invoke(
            autosre_cli.cli,
            ["hooks", "status", "--settings", str(settings_path)],
        )
        assert r2.exit_code == 0
        assert "installed" in r2.output.lower()

    def test_hooks_install_then_uninstall_cli(
        self,
        settings_path: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        monkeypatch.setenv("HOME", str(fake_home))

        runner = CliRunner()
        runner.invoke(
            autosre_cli.cli,
            ["hooks", "install", "--settings", str(settings_path)],
        )
        r = runner.invoke(
            autosre_cli.cli,
            ["hooks", "uninstall", "--settings", str(settings_path)],
        )
        assert r.exit_code == 0
        assert "Uninstalled" in r.output

        # Settings.json should still exist but have no autosre commands
        data = _load_json(settings_path)
        commands = _all_hook_commands(data)
        assert not any("autosre hooks run" in c for c in commands)

    def test_hooks_status_not_installed_cli(
        self,
        settings_path: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        monkeypatch.setenv("HOME", str(fake_home))

        runner = CliRunner()
        result = runner.invoke(
            autosre_cli.cli,
            ["hooks", "status", "--settings", str(settings_path)],
        )
        assert result.exit_code == 0
        assert "not installed" in result.output
