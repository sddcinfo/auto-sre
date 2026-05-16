"""Tests for autosre.dropbox.config.DropboxConfig.load precedence."""

from __future__ import annotations

from pathlib import Path

import pytest

from autosre.dropbox.config import (
    DEFAULT_DATA_DIR,
    DEFAULT_LISTEN_PORT,
    DEFAULT_UPSTREAM_PORT,
    DropboxConfig,
)


class TestDefaults:
    def test_defaults_with_no_file_or_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Clear any env vars that might leak in
        for k in [
            "AUTOSRE_DROPBOX_DATA_DIR",
            "AUTOSRE_DROPBOX_LISTEN_PORT",
            "AUTOSRE_DROPBOX_UPSTREAM_PORT",
            "AUTOSRE_DROPBOX_TLS_DIR",
            "AUTOSRE_DROPBOX_CONFIG_DIR",
            "AUTOSRE_DROPBOX_FILES_DIR",
        ]:
            monkeypatch.delenv(k, raising=False)

        cfg = DropboxConfig.load(config_file=Path("/definitely/nonexistent.toml"))

        assert cfg.data_dir == DEFAULT_DATA_DIR
        assert cfg.listen_port == DEFAULT_LISTEN_PORT
        assert cfg.upstream_port == DEFAULT_UPSTREAM_PORT
        assert cfg.cert_file == DEFAULT_DATA_DIR / "tls/cert.pem"
        assert cfg.key_file == DEFAULT_DATA_DIR / "tls/key.pem"
        assert cfg.password_file == DEFAULT_DATA_DIR / "config/admin-password"
        assert cfg.filebrowser_db == DEFAULT_DATA_DIR / "config/filebrowser.db"
        assert cfg.upstream == ("127.0.0.1", DEFAULT_UPSTREAM_PORT)


class TestTomlFile:
    def test_data_dir_propagates_to_subdirs(self, tmp_path: Path) -> None:
        toml = tmp_path / "dropbox.toml"
        toml.write_text('[dropbox]\ndata_dir = "/srv/dbx"\n')

        cfg = DropboxConfig.load(config_file=toml)

        assert cfg.data_dir == Path("/srv/dbx")
        assert cfg.files_dir == Path("/srv/dbx/files")
        assert cfg.tls_dir == Path("/srv/dbx/tls")
        assert cfg.config_dir == Path("/srv/dbx/config")

    def test_explicit_subdir_overrides_data_dir(self, tmp_path: Path) -> None:
        toml = tmp_path / "dropbox.toml"
        toml.write_text(
            '[dropbox]\ndata_dir = "/srv/dbx"\nconfig_dir = "/etc/dbx"\ntls_dir = "/var/ssl/dbx"\n'
        )

        cfg = DropboxConfig.load(config_file=toml)

        assert cfg.data_dir == Path("/srv/dbx")
        assert cfg.config_dir == Path("/etc/dbx")
        assert cfg.tls_dir == Path("/var/ssl/dbx")
        # files_dir was not overridden, so it still derives from data_dir
        assert cfg.files_dir == Path("/srv/dbx/files")

    def test_ports_and_bind_addr(self, tmp_path: Path) -> None:
        toml = tmp_path / "dropbox.toml"
        toml.write_text(
            "[dropbox]\n"
            "listen_port = 9443\n"
            "upstream_port = 19443\n"
            'bind_addr = "127.0.0.1"\n'
            "cookie_ttl_seconds = 3600\n"
        )

        cfg = DropboxConfig.load(config_file=toml)

        assert cfg.listen_port == 9443
        assert cfg.upstream_port == 19443
        assert cfg.bind_addr == "127.0.0.1"
        assert cfg.cookie_ttl_seconds == 3600
        assert cfg.upstream == ("127.0.0.1", 19443)

    def test_top_level_section_accepted(self, tmp_path: Path) -> None:
        """Schema tolerates both [dropbox] and top-level keys."""
        toml = tmp_path / "dropbox.toml"
        toml.write_text('data_dir = "/top/level"\nlisten_port = 7777\n')

        cfg = DropboxConfig.load(config_file=toml)

        assert cfg.data_dir == Path("/top/level")
        assert cfg.listen_port == 7777

    def test_missing_file_silently_ignored(self, tmp_path: Path) -> None:
        cfg = DropboxConfig.load(config_file=tmp_path / "nope.toml")

        assert cfg.data_dir == DEFAULT_DATA_DIR


class TestEnvOverrides:
    def test_env_beats_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AUTOSRE_DROPBOX_DATA_DIR", "/opt/dbx")
        monkeypatch.setenv("AUTOSRE_DROPBOX_LISTEN_PORT", "1234")

        cfg = DropboxConfig.load(config_file=Path("/nonexistent"))

        assert cfg.data_dir == Path("/opt/dbx")
        assert cfg.tls_dir == Path("/opt/dbx/tls")
        assert cfg.listen_port == 1234

    def test_env_beats_toml(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        toml = tmp_path / "dropbox.toml"
        toml.write_text('[dropbox]\nlisten_port = 1\ndata_dir = "/toml"\n')
        monkeypatch.setenv("AUTOSRE_DROPBOX_LISTEN_PORT", "9999")
        monkeypatch.setenv("AUTOSRE_DROPBOX_DATA_DIR", "/env")

        cfg = DropboxConfig.load(config_file=toml)

        assert cfg.listen_port == 9999
        assert cfg.data_dir == Path("/env")
        assert cfg.tls_dir == Path("/env/tls")

    def test_env_path_field_overrides(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AUTOSRE_DROPBOX_TLS_DIR", "/secure/tls")

        cfg = DropboxConfig.load(config_file=Path("/nonexistent"))

        assert cfg.tls_dir == Path("/secure/tls")


class TestFrozen:
    def test_config_is_frozen(self) -> None:
        cfg = DropboxConfig.load(config_file=Path("/nonexistent"))
        with pytest.raises((AttributeError, Exception)):
            cfg.listen_port = 1  # type: ignore[misc]
