"""Tests for the `autosre claude` command's hook + env injection.

Verifies that launching ``autosre claude``:

1. Writes a temp settings file with hook-event keys nested under a
   top-level ``"hooks"`` wrapper per the Claude Code settings schema.
2. Each hook entry invokes ``autosre hooks run <module>`` — no absolute
   Python path is embedded.
3. Sets ``AUTOSRE_REVIEW_MODEL`` and ``AUTOSRE_RUN_ID`` env vars before exec.
4. Uses unique temp filenames keyed on ``AUTOSRE_RUN_ID`` so concurrent
   launches don't collide.
5. Does **not** touch ``~/.claude/settings.json`` — the core isolation
   invariant.

We monkey-patch ``os.execvpe`` so the command returns instead of replacing
the process, and monkey-patch the vLLM backend so we don't need a live
server.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

from autosre import cli as autosre_cli
from autosre.backends.base import BackendType


@pytest.fixture
def _patched_claude(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    """Patch out everything ``autosre claude`` touches besides our asserts.

    Returns a dict the test can inspect after invocation, populated by the
    patched ``os.execvpe`` stub.
    """
    captured: dict[str, Any] = {}

    # Redirect XDG roots into tmp_path so any paths autosre writes land here.
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg-data"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg-config"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "xdg-state"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg-cache"))
    monkeypatch.delenv("AUTOSRE_RUN_ID", raising=False)

    # Redirect the tempdir so the temp settings/mcp files don't leak into /tmp.
    fake_tmp = tmp_path / "tmp"
    fake_tmp.mkdir()
    monkeypatch.setenv("TMPDIR", str(fake_tmp))
    monkeypatch.setenv("TEMP", str(fake_tmp))
    monkeypatch.setenv("TMP", str(fake_tmp))

    # Redirect HOME so ~/.claude/settings.json lands in our fake HOME.
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))

    # Create a stale "user" settings.json so we can verify mtime is unchanged.
    user_settings = fake_home / ".claude" / "settings.json"
    user_settings.parent.mkdir(parents=True)
    user_settings.write_text('{"permissions": {"allow": ["Read"]}, "user_key": "keep"}')
    captured["user_settings_path"] = user_settings
    captured["user_settings_mtime_before"] = user_settings.stat().st_mtime
    captured["user_settings_content_before"] = user_settings.read_text()

    # Build a mock backend that reports healthy and returns a stable model id.
    mock_backend = MagicMock()
    mock_backend.is_healthy.return_value = True
    mock_backend.get_claude_env.return_value = {
        "ANTHROPIC_BASE_URL": "http://localhost:8011",
    }
    mock_backend.get_claude_model_arg.return_value = "Qwen/Qwen3.6-35B-A3B-FP8"

    # start_proxy is called only for VllmBackend instances; our mock is not,
    # so the isinstance check will be False. To take that path, we patch the
    # check itself. Simpler: patch get_backend to return our mock and patch
    # the VllmBackend isinstance check via direct attribute access.
    monkeypatch.setattr(
        "autosre.cli.load_active_state",
        lambda: {
            "backend": "vllm",
            "model": "qwen3.6-fp8",
            "api_host": "localhost",
            "api_port": 8010,
            "proxy_port": 8011,
        },
    )
    monkeypatch.setattr(
        "autosre.cli.get_backend",
        lambda _bt, active_state=None: mock_backend,  # type: ignore[misc]
    )

    # detect_platform isn't reached because active state is set, but patch
    # it anyway for safety.
    monkeypatch.setattr("autosre.cli.detect_platform", lambda: BackendType.VLLM)

    # shutil.which("claude") must return truthy so the command continues past
    # the binary check.
    import shutil as _shutil

    monkeypatch.setattr(_shutil, "which", lambda _cmd: "/usr/bin/claude")

    # Skip VllmBackend proxy autostart path — the isinstance check fails for
    # the MagicMock, so start_proxy won't be called.
    # We do need to avoid the "Check health" step succeeding, which it does.

    # Capture execvpe instead of actually exec'ing.
    def _fake_execvpe(
        file: str,
        args: list[str],
        env_dict: dict[str, str],
    ) -> None:
        captured["file"] = file
        captured["args"] = args
        captured["env"] = env_dict
        # Return instead of exec'ing.

    monkeypatch.setattr(os, "execvpe", _fake_execvpe)

    # Also record user_settings mtime after to detect mutation.
    captured["user_settings_mtime_after_getter"] = lambda: (
        user_settings.stat().st_mtime if user_settings.exists() else None
    )
    captured["user_settings_content_after_getter"] = lambda: (
        user_settings.read_text() if user_settings.exists() else None
    )

    return captured


class TestClaudeCommandHookInjection:
    def test_happy_path_injects_hooks_and_env(
        self,
        _patched_claude: dict[str, Any],
    ) -> None:
        runner = CliRunner()
        result = runner.invoke(autosre_cli.cli, ["claude"])

        assert result.exit_code == 0, result.output

        # --- env vars ---
        env = _patched_claude["env"]
        assert env["AUTOSRE_REVIEW_CHAIN"] == "local", (
            "autosre claude must force local-only review (no codex fallback); "
            "codex is for the online chain used by bare `claude`"
        )
        assert env["AUTOSRE_REVIEW_MODEL"] == "Qwen/Qwen3.6-35B-A3B-FP8"
        assert "AUTOSRE_RUN_ID" in env
        run_id = env["AUTOSRE_RUN_ID"]
        assert len(run_id) > 10  # uuid4 string

        # --- temp settings file ---
        args = _patched_claude["args"]
        # Find the --settings=... argument
        settings_arg = next((a for a in args if a.startswith("--settings=")), None)
        assert settings_arg is not None, f"--settings= flag missing: {args}"
        settings_path = settings_arg.split("=", 1)[1]
        assert run_id in settings_path, (
            f"settings path must include run_id for concurrent-safety: {settings_path}"
        )

        with Path(settings_path).open() as f:
            settings = json.load(f)

        # Event keys live under a top-level "hooks" wrapper per the Claude
        # Code settings.json schema — entries at the root are silently ignored.
        assert "hooks" in settings
        hooks_root = settings["hooks"]
        assert "PreToolUse" in hooks_root
        assert "PostToolUse" in hooks_root
        assert "Stop" in hooks_root
        assert "UserPromptSubmit" in hooks_root
        assert "PreCompact" in hooks_root
        assert "SubagentStart" in hooks_root

        # PreToolUse should have matchers: Bash, Edit, Write, ExitPlanMode
        matchers = {entry["matcher"] for entry in hooks_root["PreToolUse"]}
        assert matchers == {"Bash", "Edit", "Write", "ExitPlanMode"}

        # Hook commands must invoke `autosre hooks run <module>`
        all_hook_cmds: list[str] = []
        for event_key in (
            "PreToolUse",
            "PostToolUse",
            "Stop",
            "UserPromptSubmit",
            "PreCompact",
            "SubagentStart",
        ):
            for entry in hooks_root[event_key]:
                for hook in entry["hooks"]:
                    all_hook_cmds.append(hook["command"])
        assert all(c.startswith("autosre hooks run ") for c in all_hook_cmds)

        assert any("pretooluse_bash_guard" in c for c in all_hook_cmds)
        assert any("pretooluse_plan_review" in c for c in all_hook_cmds)
        assert any("stop_session_check" in c for c in all_hook_cmds)
        assert any("posttooluse_audit" in c for c in all_hook_cmds)
        assert any("telemetry_async" in c for c in all_hook_cmds)
        assert any("post_commit_scan_update" in c for c in all_hook_cmds)
        assert any("user_prompt_submit_branch_check" in c for c in all_hook_cmds)
        assert any("precompact_context" in c for c in all_hook_cmds)
        assert any("subagent_plan_context" in c for c in all_hook_cmds)

        # Every hook command must invoke `autosre hooks run <module>` — no
        # absolute Python path embedded; `autosre` resolves from $PATH.
        for cmd in all_hook_cmds:
            assert cmd.startswith("autosre hooks run ")
            assert "python" not in cmd

    def test_mcp_config_uses_uniquified_path(
        self,
        _patched_claude: dict[str, Any],
    ) -> None:
        runner = CliRunner()
        result = runner.invoke(autosre_cli.cli, ["claude"])
        assert result.exit_code == 0, result.output

        args = _patched_claude["args"]
        mcp_arg = next((a for a in args if a.startswith("--mcp-config=")), None)
        assert mcp_arg is not None
        mcp_path = mcp_arg.split("=", 1)[1]
        run_id = _patched_claude["env"]["AUTOSRE_RUN_ID"]
        assert run_id in mcp_path

    def test_mcp_config_includes_capabilities_server(
        self,
        _patched_claude: dict[str, Any],
    ) -> None:
        runner = CliRunner()
        result = runner.invoke(autosre_cli.cli, ["claude"])
        assert result.exit_code == 0, result.output

        args = _patched_claude["args"]
        mcp_arg = next(a for a in args if a.startswith("--mcp-config="))
        mcp_path = mcp_arg.split("=", 1)[1]
        with Path(mcp_path).open() as f:
            mcp_config = json.load(f)

        servers = mcp_config["mcpServers"]
        assert "autosre-fetch" in servers
        assert "autosre-search" in servers
        assert "autosre-capabilities" in servers
        assert servers["autosre-capabilities"]["command"] == "autosre-mcp-capabilities"

        # And the permissions.allow list should include the new tool IDs
        settings_arg = next(a for a in args if a.startswith("--settings="))
        settings_path = settings_arg.split("=", 1)[1]
        with Path(settings_path).open() as f:
            settings = json.load(f)
        allow = settings["permissions"]["allow"]
        assert "mcp__autosre-capabilities__search_commands" in allow
        assert "mcp__autosre-capabilities__list_modules" in allow
        assert "mcp__autosre-capabilities__get_command" in allow

    def test_user_claude_settings_untouched(
        self,
        _patched_claude: dict[str, Any],
    ) -> None:
        runner = CliRunner()
        result = runner.invoke(autosre_cli.cli, ["claude"])
        assert result.exit_code == 0, result.output

        content_before = _patched_claude["user_settings_content_before"]
        content_after = _patched_claude["user_settings_content_after_getter"]()
        assert content_after == content_before, (
            "autosre claude must never mutate ~/.claude/settings.json"
        )

        mtime_before = _patched_claude["user_settings_mtime_before"]
        mtime_after = _patched_claude["user_settings_mtime_after_getter"]()
        assert mtime_after == mtime_before

    def test_concurrent_run_ids_produce_distinct_files(
        self,
        _patched_claude: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Two invocations with the same process state must produce
        different temp filenames (they use fresh UUID per launch)."""
        seen_settings_paths: set[str] = set()

        def _record(
            _file: str,
            args: list[str],
            _env_dict: dict[str, str],
        ) -> None:
            settings_arg = next(a for a in args if a.startswith("--settings="))
            seen_settings_paths.add(settings_arg.split("=", 1)[1])

        monkeypatch.setattr(os, "execvpe", _record)

        runner = CliRunner()
        r1 = runner.invoke(autosre_cli.cli, ["claude"])
        r2 = runner.invoke(autosre_cli.cli, ["claude"])
        assert r1.exit_code == 0, r1.output
        assert r2.exit_code == 0, r2.output

        assert len(seen_settings_paths) == 2, (
            f"concurrent runs must use distinct temp files: {seen_settings_paths}"
        )
