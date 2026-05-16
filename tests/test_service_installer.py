"""Tests for ``autosre install-service`` / ``uninstall-service``.

systemctl + loginctl are mocked so the suite runs anywhere — including
CI containers without a user-bus.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from autosre import service_installer
from autosre.cli import cli


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    """Redirect XDG + autosre-bin discovery so the real user's $HOME
    is untouched."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / ".config"))
    monkeypatch.setenv("USER", "tester")
    fake_bin = tmp_path / "bin" / "autosre"
    fake_bin.parent.mkdir(parents=True, exist_ok=True)
    fake_bin.write_text("#!/bin/sh\n")
    fake_bin.chmod(0o755)
    monkeypatch.setattr(service_installer, "_autosre_bin", lambda: fake_bin)
    yield tmp_path


# ── Unit rendering ──────────────────────────────────────────────────


def test_render_unit_has_simple_type(fake_home):
    body = service_installer._render_unit(Path(fake_home) / "bin" / "autosre")
    assert "Type=simple" in body


def test_render_unit_runs_no_scribe(fake_home):
    """The unit must use --no-scribe so it doesn't fight with
    meeting-scribe's own user service."""
    body = service_installer._render_unit(Path(fake_home) / "bin" / "autosre")
    assert "start --no-scribe" in body


def test_render_unit_includes_execstop(fake_home):
    """ExecStop= so ``systemctl --user stop autosre`` actually
    propagates to the running stack."""
    body = service_installer._render_unit(Path(fake_home) / "bin" / "autosre")
    assert "ExecStop=" in body and " stop" in body


def test_render_unit_targets_default_target(fake_home):
    body = service_installer._render_unit(Path(fake_home) / "bin" / "autosre")
    assert "WantedBy=default.target" in body


def test_render_unit_path_includes_autosre_bin(fake_home):
    body = service_installer._render_unit(Path(fake_home) / "bin" / "autosre")
    expected_dir = str((Path(fake_home) / "bin").resolve())
    assert expected_dir in body


# ── install / uninstall ─────────────────────────────────────────────


class _MockProc:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_install_service_writes_unit_and_enables(runner, fake_home):
    calls: list[list[str]] = []

    def fake_run(cmd, *args, **kwargs):
        calls.append(list(cmd))
        return _MockProc(returncode=0)

    with (
        patch("autosre.service_installer.shutil.which", return_value="/bin/systemctl"),
        patch("autosre.service_installer.subprocess.run", side_effect=fake_run),
    ):
        result = runner.invoke(cli, ["install-service"])

    assert result.exit_code == 0, result.output
    unit = fake_home / ".config" / "systemd" / "user" / "autosre.service"
    assert unit.exists()
    assert "Type=simple" in unit.read_text()

    cmds = [" ".join(c) for c in calls]
    assert any("systemctl --user daemon-reload" in c for c in cmds)
    assert any("systemctl --user enable --now autosre.service" in c for c in cmds)


def test_install_service_no_start_only_enables(runner, fake_home):
    calls: list[list[str]] = []

    def fake_run(cmd, *args, **kwargs):
        calls.append(list(cmd))
        return _MockProc(returncode=0)

    with (
        patch("autosre.service_installer.shutil.which", return_value="/bin/systemctl"),
        patch("autosre.service_installer.subprocess.run", side_effect=fake_run),
    ):
        result = runner.invoke(cli, ["install-service", "--no-start"])

    assert result.exit_code == 0, result.output
    cmds = [" ".join(c) for c in calls]
    assert any("systemctl --user enable autosre.service" in c for c in cmds)
    assert not any("enable --now" in c for c in cmds)


def test_install_service_no_enable_skips_systemctl_enable(runner, fake_home):
    calls: list[list[str]] = []

    def fake_run(cmd, *args, **kwargs):
        calls.append(list(cmd))
        return _MockProc(returncode=0)

    with (
        patch("autosre.service_installer.shutil.which", return_value="/bin/systemctl"),
        patch("autosre.service_installer.subprocess.run", side_effect=fake_run),
    ):
        result = runner.invoke(cli, ["install-service", "--no-enable"])

    assert result.exit_code == 0, result.output
    cmds = [" ".join(c) for c in calls]
    # daemon-reload runs so the file is recognized; enable does not
    assert any("daemon-reload" in c for c in cmds)
    assert not any("systemctl --user enable" in c for c in cmds)
    assert not any("loginctl" in c for c in cmds)


def test_install_service_idempotent_when_unchanged(runner, fake_home):
    def fake_run(cmd, *args, **kwargs):
        return _MockProc(returncode=0)

    with (
        patch("autosre.service_installer.shutil.which", return_value="/bin/systemctl"),
        patch("autosre.service_installer.subprocess.run", side_effect=fake_run),
    ):
        first = runner.invoke(cli, ["install-service", "--no-enable"])
        second = runner.invoke(cli, ["install-service", "--no-enable"])

    assert first.exit_code == 0 and second.exit_code == 0
    assert "wrote" in first.output
    assert "already up-to-date" in second.output


def test_install_service_fails_loudly_without_systemctl(runner, fake_home):
    with patch("autosre.service_installer.shutil.which", return_value=None):
        result = runner.invoke(cli, ["install-service"])
    assert result.exit_code == 1
    assert "systemctl not found" in result.output


def test_install_service_bails_when_autosre_bin_missing(runner, fake_home, monkeypatch):
    monkeypatch.setattr(service_installer, "_autosre_bin", lambda: None)
    with patch("autosre.service_installer.shutil.which", return_value="/bin/systemctl"):
        result = runner.invoke(cli, ["install-service"])
    assert result.exit_code == 1
    assert "Could not resolve" in result.output


def test_uninstall_service_removes_unit_file(runner, fake_home):
    unit = fake_home / ".config" / "systemd" / "user" / "autosre.service"
    unit.parent.mkdir(parents=True)
    unit.write_text("# stale unit\n")

    def fake_run(cmd, *args, **kwargs):
        return _MockProc(returncode=0)

    with (
        patch("autosre.service_installer.shutil.which", return_value="/bin/systemctl"),
        patch("autosre.service_installer.subprocess.run", side_effect=fake_run),
    ):
        result = runner.invoke(cli, ["uninstall-service"])

    assert result.exit_code == 0, result.output
    assert not unit.exists()


def test_uninstall_service_tolerates_unit_not_loaded(runner, fake_home):
    def fake_run(cmd, *args, **kwargs):
        if "stop" in cmd or "disable" in cmd:
            return _MockProc(returncode=1, stderr="Unit autosre.service not loaded.")
        return _MockProc(returncode=0)

    with (
        patch("autosre.service_installer.shutil.which", return_value="/bin/systemctl"),
        patch("autosre.service_installer.subprocess.run", side_effect=fake_run),
    ):
        result = runner.invoke(cli, ["uninstall-service"])

    assert result.exit_code == 0
    assert "[WARN]" not in result.output
