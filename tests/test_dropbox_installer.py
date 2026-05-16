"""Tests for autosre.dropbox.installer — template rendering + sidecar."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from autosre.dropbox.config import DropboxConfig
from autosre.dropbox.installer import (
    InitSystem,
    InstallContext,
    InstallMode,
    ResolvedUser,
    _sha256,
    build_install_context,
    detect_init_system,
    probe_systemd_user_operable,
    render_unit_files,
    resolve_user,
)


@pytest.fixture
def fake_config() -> DropboxConfig:
    return DropboxConfig.load(config_file=Path("/nonexistent"))


@pytest.fixture
def fake_user() -> ResolvedUser:
    return ResolvedUser(
        name="testuser",
        uid=1000,
        gid=1000,
        home=Path("/home/testuser"),
    )


class TestDetectInitSystem:
    def test_returns_systemd_when_systemctl_present(self) -> None:
        with patch("autosre.dropbox.installer.shutil.which") as which:
            which.side_effect = lambda name: "/usr/bin/systemctl" if name == "systemctl" else None
            assert detect_init_system() == InitSystem.SYSTEMD

    def test_returns_unknown_when_no_init(self) -> None:
        with (
            patch("autosre.dropbox.installer.shutil.which", return_value=None),
            patch("autosre.dropbox.installer.Path.exists", return_value=False),
        ):
            assert detect_init_system() == InitSystem.UNKNOWN


class TestProbeSystemdUserOperable:
    def test_fails_when_systemctl_missing(self) -> None:
        with patch("autosre.dropbox.installer.shutil.which", return_value=None):
            ok, detail = probe_systemd_user_operable()
            assert not ok
            assert "systemctl" in detail

    def test_fails_when_show_environment_fails(self) -> None:
        with (
            patch("autosre.dropbox.installer.shutil.which", return_value="/usr/bin/systemctl"),
            patch("autosre.dropbox.installer.subprocess.run") as run,
        ):
            run.return_value = MagicMock(returncode=1, stderr="Failed to connect to bus", stdout="")
            ok, detail = probe_systemd_user_operable()
            assert not ok
            assert "show-environment" in detail

    def test_ok_when_both_probes_pass(self) -> None:
        with (
            patch("autosre.dropbox.installer.shutil.which", return_value="/usr/bin/systemctl"),
            patch("autosre.dropbox.installer.subprocess.run") as run,
        ):
            run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            ok, detail = probe_systemd_user_operable()
            assert ok
            assert "reachable" in detail

    def test_does_not_call_is_system_running(self) -> None:
        """Regression: we explicitly avoid is-system-running which returns
        non-zero on degraded systems that are otherwise fully operable."""
        with (
            patch("autosre.dropbox.installer.shutil.which", return_value="/usr/bin/systemctl"),
            patch("autosre.dropbox.installer.subprocess.run") as run,
        ):
            run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            probe_systemd_user_operable()

            called_args = [call.args[0] for call in run.call_args_list]
            for args in called_args:
                assert "is-system-running" not in args


class TestResolveUser:
    def test_user_mode_uses_pwd_getpwuid(self) -> None:
        fake_record = MagicMock(pw_name="alice", pw_uid=1001, pw_gid=1001, pw_dir="/home/alice")
        with patch("autosre.dropbox.installer.pwd.getpwuid", return_value=fake_record) as getpwuid:
            user = resolve_user(None, InstallMode.USER)
            assert user.name == "alice"
            assert user.home == Path("/home/alice")
            getpwuid.assert_called_once()

    def test_user_mode_rejects_service_user(self) -> None:
        with pytest.raises(ValueError, match="only valid with --system"):
            resolve_user("bob", InstallMode.USER)

    def test_system_mode_requires_service_user(self) -> None:
        with pytest.raises(ValueError, match="requires --service-user"):
            resolve_user(None, InstallMode.SYSTEM)

    def test_system_mode_uses_pwd_getpwnam(self) -> None:
        fake_record = MagicMock(pw_name="svc", pw_uid=500, pw_gid=500, pw_dir="/var/lib/svc")
        with patch("autosre.dropbox.installer.pwd.getpwnam", return_value=fake_record) as getpwnam:
            user = resolve_user("svc", InstallMode.SYSTEM)
            assert user.name == "svc"
            getpwnam.assert_called_once_with("svc")

    def test_system_mode_missing_user_errors(self) -> None:
        with (
            patch("autosre.dropbox.installer.pwd.getpwnam", side_effect=KeyError("svc")),
            pytest.raises(ValueError, match="not found in passwd"),
        ):
            resolve_user("svc", InstallMode.SYSTEM)

    def test_never_reads_os_environ(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Regression: resolver must never trust $USER or $HOME."""
        monkeypatch.setenv("USER", "hostile")
        monkeypatch.setenv("HOME", "/tmp/evil")
        fake_record = MagicMock(pw_name="real", pw_uid=1000, pw_gid=1000, pw_dir="/home/real")
        with patch("autosre.dropbox.installer.pwd.getpwuid", return_value=fake_record):
            user = resolve_user(None, InstallMode.USER)
            assert user.name == "real"
            assert user.home == Path("/home/real")


class TestRenderUnitFiles:
    def test_user_mode_omits_user_directives(
        self, fake_config: DropboxConfig, fake_user: ResolvedUser
    ) -> None:
        ctx = InstallContext(
            config=fake_config,
            mode=InstallMode.USER,
            user=fake_user,
            python_bin=Path("/usr/bin/python3"),
            filebrowser_bin=Path("/usr/local/bin/filebrowser"),
            repo_dir=Path("/repo"),
            config_file=None,
        )
        units = render_unit_files(ctx)

        assert "dropbox.service" in units
        assert "dropbox-sniff.service" in units

        dropbox_unit = units["dropbox.service"]
        # User-mode services run as the invoking user — no User=/Group= allowed
        assert "User=" not in dropbox_unit
        assert "Group=" not in dropbox_unit
        assert "WorkingDirectory=/data/dropbox" in dropbox_unit
        assert "/usr/local/bin/filebrowser" in dropbox_unit

        sniff_unit = units["dropbox-sniff.service"]
        assert "User=" not in sniff_unit
        assert "WorkingDirectory=/repo" in sniff_unit
        assert "-m autosre.dropbox.proxy" in sniff_unit
        assert "Requires=dropbox.service" in sniff_unit

    def test_system_mode_has_user_and_sandbox_directives(
        self, fake_config: DropboxConfig, fake_user: ResolvedUser
    ) -> None:
        ctx = InstallContext(
            config=fake_config,
            mode=InstallMode.SYSTEM,
            user=fake_user,
            python_bin=Path("/usr/bin/python3"),
            filebrowser_bin=Path("/usr/local/bin/filebrowser"),
            repo_dir=Path("/repo"),
            config_file=None,
        )
        units = render_unit_files(ctx)

        dropbox_unit = units["dropbox.service"]
        assert "User=testuser" in dropbox_unit
        assert "Group=testuser" in dropbox_unit
        assert "ProtectSystem=full" in dropbox_unit
        assert "ProtectHome=read-only" in dropbox_unit
        assert f"ReadWritePaths={fake_config.data_dir}" in dropbox_unit

    def test_user_mode_uses_default_target(
        self, fake_config: DropboxConfig, fake_user: ResolvedUser
    ) -> None:
        ctx = InstallContext(
            config=fake_config,
            mode=InstallMode.USER,
            user=fake_user,
            python_bin=Path("/usr/bin/python3"),
            filebrowser_bin=Path("/usr/local/bin/filebrowser"),
            repo_dir=Path("/repo"),
            config_file=None,
        )
        units = render_unit_files(ctx)
        assert "WantedBy=default.target" in units["dropbox.service"]

    def test_system_mode_uses_multi_user_target(
        self, fake_config: DropboxConfig, fake_user: ResolvedUser
    ) -> None:
        ctx = InstallContext(
            config=fake_config,
            mode=InstallMode.SYSTEM,
            user=fake_user,
            python_bin=Path("/usr/bin/python3"),
            filebrowser_bin=Path("/usr/local/bin/filebrowser"),
            repo_dir=Path("/repo"),
            config_file=None,
        )
        units = render_unit_files(ctx)
        assert "WantedBy=multi-user.target" in units["dropbox.service"]

    def test_config_file_passed_via_env(
        self, fake_config: DropboxConfig, fake_user: ResolvedUser
    ) -> None:
        ctx = InstallContext(
            config=fake_config,
            mode=InstallMode.USER,
            user=fake_user,
            python_bin=Path("/usr/bin/python3"),
            filebrowser_bin=Path("/usr/local/bin/filebrowser"),
            repo_dir=Path("/repo"),
            config_file=Path("/opt/dropbox.toml"),
        )
        units = render_unit_files(ctx)

        assert "AUTOSRE_DROPBOX_CONFIG_FILE=/opt/dropbox.toml" in units["dropbox-sniff.service"]

    def test_ports_appear_in_filebrowser_unit(
        self, fake_config: DropboxConfig, fake_user: ResolvedUser
    ) -> None:
        ctx = InstallContext(
            config=fake_config,
            mode=InstallMode.USER,
            user=fake_user,
            python_bin=Path("/usr/bin/python3"),
            filebrowser_bin=Path("/usr/local/bin/filebrowser"),
            repo_dir=Path("/repo"),
            config_file=None,
        )
        units = render_unit_files(ctx)
        assert f"-p {fake_config.upstream_port}" in units["dropbox.service"]


class TestBuildInstallContext:
    def test_uses_defaults_when_unspecified(self, fake_config: DropboxConfig) -> None:
        fake_record = MagicMock(pw_name="me", pw_uid=1000, pw_gid=1000, pw_dir="/home/me")
        with patch("autosre.dropbox.installer.pwd.getpwuid", return_value=fake_record):
            ctx = build_install_context(
                config=fake_config,
                mode=InstallMode.USER,
                service_user=None,
                python_bin=None,
                filebrowser_bin=None,
                repo_dir=Path("/repo"),
                config_file=None,
            )

        assert ctx.user.name == "me"
        # filebrowser_bin falls back to the config's default name (PATH lookup)
        assert str(ctx.filebrowser_bin).endswith("filebrowser")
        assert ctx.python_bin.exists()  # should be the current interpreter


class TestSha256:
    def test_matches_hashlib(self) -> None:
        import hashlib as _h

        content = "hello world"
        assert _sha256(content) == _h.sha256(content.encode()).hexdigest()
