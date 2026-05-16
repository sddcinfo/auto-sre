"""Tests for autosre.paths XDG helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from autosre import paths

if TYPE_CHECKING:
    from pathlib import Path


class TestXdgOverrides:
    def test_data_dir_respects_xdg_data_home(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
        d = paths.data_dir()
        assert d == tmp_path / "autosre"
        assert d.is_dir()

    def test_config_dir_respects_xdg_config_home(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        d = paths.config_dir()
        assert d == tmp_path / "autosre"
        assert d.is_dir()

    def test_state_dir_respects_xdg_state_home(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
        d = paths.state_dir()
        assert d == tmp_path / "autosre"
        assert d.is_dir()

    def test_cache_dir_respects_xdg_cache_home(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
        d = paths.cache_dir()
        assert d == tmp_path / "autosre"
        assert d.is_dir()

    def test_defaults_when_env_unset(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("XDG_DATA_HOME", raising=False)
        monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
        monkeypatch.delenv("XDG_STATE_HOME", raising=False)
        monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
        monkeypatch.setenv("HOME", str(tmp_path))

        assert paths.data_dir() == tmp_path / ".local/share/autosre"
        assert paths.config_dir() == tmp_path / ".config/autosre"
        assert paths.state_dir() == tmp_path / ".local/state/autosre"
        assert paths.cache_dir() == tmp_path / ".cache/autosre"


class TestSubsystemPaths:
    @pytest.fixture(autouse=True)
    def _isolate(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # Redirect all four XDG roots into tmp_path so every subsystem path is
        # isolated from the real filesystem.
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
        monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    def test_review_state_dir(self, tmp_path: Path) -> None:
        d = paths.review_state_dir()
        assert d == tmp_path / "data" / "autosre" / "review-state"
        assert d.is_dir()

    def test_review_log_dir(self, tmp_path: Path) -> None:
        d = paths.review_log_dir()
        assert d == tmp_path / "data" / "autosre" / "review-log"
        assert d.is_dir()

    def test_guard_rules_file(self, tmp_path: Path) -> None:
        f = paths.guard_rules_file()
        assert f == tmp_path / "config" / "autosre" / "guard-rules.yaml"
        # file itself is not auto-created, only its parent
        assert f.parent.is_dir()
        assert not f.exists()

    def test_guard_approvals_dir(self, tmp_path: Path) -> None:
        d = paths.guard_approvals_dir()
        assert d == tmp_path / "state" / "autosre" / "approvals"
        assert d.is_dir()

    def test_hook_logs_and_marker(self, tmp_path: Path) -> None:
        assert paths.hook_audit_log() == tmp_path / "state" / "autosre" / "hook-audit.log"
        assert paths.hook_blocked_log() == tmp_path / "state" / "autosre" / "hook-blocked.log"
        assert paths.hook_errors_log() == tmp_path / "state" / "autosre" / "hook-errors.log"
        assert paths.branch_warned_marker() == tmp_path / "state" / "autosre" / "branch-warned"
        # Parents exist, files do not (touched on first write by callers).
        assert paths.hook_audit_log().parent.is_dir()
        assert not paths.hook_audit_log().exists()

    def test_capabilities_index_file(self, tmp_path: Path) -> None:
        f = paths.capabilities_index_file()
        assert f == tmp_path / "data" / "autosre" / "capabilities-index.json"
        assert f.parent.is_dir()
