"""Tests for autosre.tmux — isolated from any real tmux server."""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING
from unittest.mock import patch

from autosre import tmux as tmux_mod

if TYPE_CHECKING:
    import pytest


class TestDetection:
    def test_in_tmux_true_when_env_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TMUX", "/tmp/tmux-xyz,1,0")
        assert tmux_mod.in_tmux() is True

    def test_in_tmux_false_when_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("TMUX", raising=False)
        assert tmux_mod.in_tmux() is False

    def test_splitting_disabled_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AUTOSRE_NO_SPLIT", "1")
        assert tmux_mod.is_splitting_disabled() is True

    def test_splitting_enabled_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("AUTOSRE_NO_SPLIT", raising=False)
        assert tmux_mod.is_splitting_disabled() is False


class TestSelfSplitOnce:
    def test_noop_outside_tmux(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("TMUX", raising=False)
        result = tmux_mod.self_split_once()
        assert result.performed is False
        assert "not in tmux" in result.reason

    def test_noop_when_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TMUX", "fake")
        monkeypatch.setenv("AUTOSRE_NO_SPLIT", "1")
        result = tmux_mod.self_split_once()
        assert result.performed is False
        assert "NO_SPLIT" in result.reason

    def test_noop_when_already_split(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TMUX", "fake")
        monkeypatch.delenv("AUTOSRE_NO_SPLIT", raising=False)
        with patch("autosre.tmux.shutil.which", return_value="/usr/bin/tmux"):
            call_count = 0

            def fake_run_tmux(*args: str) -> subprocess.CompletedProcess[str]:
                nonlocal call_count
                call_count += 1
                # show-options -w -v @autosre_split_done → "on"
                if "show-options" in args:
                    return subprocess.CompletedProcess(
                        args=list(args),
                        returncode=0,
                        stdout="on\n",
                        stderr="",
                    )
                return subprocess.CompletedProcess(
                    args=list(args), returncode=0, stdout="", stderr=""
                )

            with patch("autosre.tmux._run_tmux", side_effect=fake_run_tmux):
                result = tmux_mod.self_split_once()
                assert result.performed is False
                assert "already split" in result.reason

    def test_performs_split(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TMUX", "fake")
        monkeypatch.delenv("AUTOSRE_NO_SPLIT", raising=False)

        def fake_run_tmux(*args: str) -> subprocess.CompletedProcess[str]:
            if "show-options" in args:
                # Not yet split.
                return subprocess.CompletedProcess(
                    args=list(args), returncode=0, stdout="off\n", stderr=""
                )
            if "display-message" in args:
                # Return a fake pane id.
                return subprocess.CompletedProcess(
                    args=list(args), returncode=0, stdout="%5\n", stderr=""
                )
            if "split-window" in args:
                return subprocess.CompletedProcess(
                    args=list(args), returncode=0, stdout="%6\n", stderr=""
                )
            return subprocess.CompletedProcess(args=list(args), returncode=0, stdout="", stderr="")

        with (
            patch("autosre.tmux.shutil.which", return_value="/usr/bin/tmux"),
            patch("autosre.tmux._run_tmux", side_effect=fake_run_tmux),
        ):
            result = tmux_mod.self_split_once()
            assert result.performed is True
            assert result.top_pane == "%5"
            assert result.bottom_pane == "%6"

    def test_split_failure_returns_reason(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TMUX", "fake")
        monkeypatch.delenv("AUTOSRE_NO_SPLIT", raising=False)

        def fake_run_tmux(*args: str) -> subprocess.CompletedProcess[str]:
            if "show-options" in args:
                return subprocess.CompletedProcess(
                    args=list(args), returncode=0, stdout="off\n", stderr=""
                )
            if "display-message" in args:
                return subprocess.CompletedProcess(
                    args=list(args), returncode=0, stdout="%5\n", stderr=""
                )
            if "split-window" in args:
                return subprocess.CompletedProcess(
                    args=list(args), returncode=1, stdout="", stderr="boom"
                )
            return subprocess.CompletedProcess(args=list(args), returncode=0, stdout="", stderr="")

        with (
            patch("autosre.tmux.shutil.which", return_value="/usr/bin/tmux"),
            patch("autosre.tmux._run_tmux", side_effect=fake_run_tmux),
        ):
            result = tmux_mod.self_split_once()
            assert result.performed is False
            assert "boom" in result.reason


class TestRunInPane:
    def test_noop_outside_tmux(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("TMUX", raising=False)
        assert tmux_mod.run_in_pane("%1", "echo hi") is False

    def test_invokes_respawn(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TMUX", "fake")
        captured: list[tuple[str, ...]] = []

        def fake_run_tmux(*args: str) -> subprocess.CompletedProcess[str]:
            captured.append(args)
            return subprocess.CompletedProcess(args=list(args), returncode=0, stdout="", stderr="")

        with (
            patch("autosre.tmux.shutil.which", return_value="/usr/bin/tmux"),
            patch("autosre.tmux._run_tmux", side_effect=fake_run_tmux),
        ):
            ok = tmux_mod.run_in_pane("%1", "autosre claude")
            assert ok is True
        assert captured
        args = captured[0]
        assert args[0] == "respawn-pane"
        assert "-k" in args
        assert "-t" in args and "%1" in args
        assert "autosre claude" in args
