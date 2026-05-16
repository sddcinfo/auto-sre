"""Tests for ``autosre.claude_hooks._io`` — the shared hook-IO layer.

These cover the concerns that bit us in practice: plan-file discovery
when Claude Code stops populating ``tool_input.planFilePath``, drift
warnings when new top-level keys appear, strict-mode visibility of
fail-open paths, and the raw-input archive.
"""

from __future__ import annotations

import io
import json
import os
from typing import TYPE_CHECKING

import pytest

from autosre import paths
from autosre.claude_hooks import _io

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(autouse=True)
def _isolate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect XDG dirs and Claude Code plan dir under tmp_path per test."""
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude"))
    monkeypatch.delenv("CLAUDE_PLAN_FILE", raising=False)
    monkeypatch.delenv("AUTOSRE_HOOKS_STRICT", raising=False)
    monkeypatch.delenv("AUTOSRE_HOOKS_DUMP_DIR", raising=False)


def _stub_stdin(monkeypatch: pytest.MonkeyPatch, payload: str) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO(payload))


def _make_inv(
    monkeypatch: pytest.MonkeyPatch,
    raw: dict[str, object] | None = None,
    *,
    hook_script: str = "/tmp/pretooluse_plan_review.py",
) -> _io.HookInvocation:
    _stub_stdin(monkeypatch, json.dumps(raw or {}))
    return _io.parse_stdin(hook_script)


class TestParseStdin:
    def test_infers_event_from_script_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        inv = _make_inv(monkeypatch, {}, hook_script="/foo/pretooluse_bash_guard.py")
        assert inv.event == "PreToolUse"

    def test_hook_event_name_overrides_inference(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        inv = _make_inv(
            monkeypatch,
            {"hook_event_name": "PostToolUse"},
            hook_script="/foo/stop_session_check.py",
        )
        assert inv.event == "PostToolUse"

    def test_malformed_json_yields_empty_raw(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _stub_stdin(monkeypatch, "not json")
        inv = _io.parse_stdin("/tmp/posttooluse_audit.py")
        assert inv.raw == {}
        assert inv.event == "PostToolUse"


class TestResolvePlanFile:
    def test_prefers_tool_input_plan_file_path(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan = tmp_path / "a.md"
        plan.write_text("plan")
        inv = _make_inv(monkeypatch, {"tool_input": {"planFilePath": str(plan)}})
        assert _io.resolve_plan_file(inv) == plan

    def test_falls_back_to_top_level_key(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan = tmp_path / "b.md"
        plan.write_text("plan")
        inv = _make_inv(monkeypatch, {"planFilePath": str(plan)})
        assert _io.resolve_plan_file(inv) == plan

    def test_falls_back_to_env_var(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan = tmp_path / "c.md"
        plan.write_text("plan")
        monkeypatch.setenv("CLAUDE_PLAN_FILE", str(plan))
        inv = _make_inv(monkeypatch, {})
        assert _io.resolve_plan_file(inv) == plan

    def test_falls_back_to_plans_dir_newest(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import time as _time

        pdir = paths.claude_plans_dir()
        pdir.mkdir(parents=True, exist_ok=True)
        older = pdir / "older.md"
        older.write_text("old")
        newer = pdir / "newer.md"
        newer.write_text("new")
        now = _time.time()
        os.utime(older, (now - 120, now - 120))
        os.utime(newer, (now - 10, now - 10))

        inv = _make_inv(monkeypatch, {})
        assert _io.resolve_plan_file(inv) == newer

    def test_ignores_plans_older_than_max_age(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        pdir = paths.claude_plans_dir()
        pdir.mkdir(parents=True, exist_ok=True)
        stale = pdir / "stale.md"
        stale.write_text("old")
        os.utime(stale, (100.0, 100.0))  # 1970-ish — beyond any reasonable max_age
        monkeypatch.setenv("AUTOSRE_HOOKS_PLAN_MAX_AGE_SECONDS", "60")

        inv = _make_inv(monkeypatch, {})
        assert _io.resolve_plan_file(inv) is None

    def test_returns_none_when_nothing_found(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        inv = _make_inv(monkeypatch, {})
        assert _io.resolve_plan_file(inv) is None


class TestDriftWarning:
    def test_unknown_top_level_key_logged_as_warn(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _make_inv(monkeypatch, {"brand_new_field_42": True, "session_id": "s"})
        log = (paths.hooks_log_dir() / "PreToolUse.log").read_text()
        assert "unknown top-level keys" in log
        assert "brand_new_field_42" in log

    def test_known_keys_dont_warn(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _make_inv(monkeypatch, {"session_id": "s", "tool_input": {}})
        log_path = paths.hooks_log_dir() / "PreToolUse.log"
        if log_path.exists():
            assert "unknown top-level keys" not in log_path.read_text()


class TestRawArchive:
    def test_writes_jsonl_line_per_invocation(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _make_inv(monkeypatch, {"session_id": "s1", "tool_input": {"command": "ls"}})
        _make_inv(monkeypatch, {"session_id": "s2"})
        lines = paths.hooks_raw_jsonl().read_text().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["session_id"] == "s1"
        assert json.loads(lines[1])["session_id"] == "s2"

    def test_dump_dir_produces_per_invocation_file(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        dump = tmp_path / "dumps"
        monkeypatch.setenv("AUTOSRE_HOOKS_DUMP_DIR", str(dump))
        _make_inv(monkeypatch, {"session_id": "s", "tool_input": {}})
        files = list(dump.iterdir())
        assert len(files) == 1
        assert files[0].name.startswith("PreToolUse-")
        assert json.loads(files[0].read_text())["session_id"] == "s"


class TestStrictMode:
    def test_fail_open_pretooluse_adds_system_message_in_strict(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setenv("AUTOSRE_HOOKS_STRICT", "1")
        inv = _make_inv(monkeypatch, {"session_id": "s"})
        rc = _io.fail_open_pretooluse(inv, "because tests")
        assert rc == 0
        out = json.loads(capsys.readouterr().out)
        assert out["hookSpecificOutput"]["permissionDecision"] == "allow"
        assert "because tests" in out["hookSpecificOutput"]["additionalContext"]
        assert "because tests" in out["systemMessage"]

    def test_fail_open_pretooluse_silent_without_strict(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        inv = _make_inv(monkeypatch, {"session_id": "s"})
        _io.fail_open_pretooluse(inv, "routine fallback")
        out = json.loads(capsys.readouterr().out)
        assert "systemMessage" not in out
        assert "routine fallback" in out["hookSpecificOutput"]["additionalContext"]

    def test_fail_open_continue_adds_message_in_strict(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setenv("AUTOSRE_HOOKS_STRICT", "1")
        inv = _make_inv(monkeypatch, {"session_id": "s"}, hook_script="/x/stop_foo.py")
        _io.fail_open_continue(inv, "backend flake")
        out = json.loads(capsys.readouterr().out)
        assert out["result"] == "continue"
        assert "backend flake" in out["message"]
