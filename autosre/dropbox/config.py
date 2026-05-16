"""Dropbox subsystem configuration.

Single source of truth for every dropbox path, port, and tunable.
Loaded via ``DropboxConfig.load()`` and consumed by ``proxy.py``,
``installer.py``, ``filebrowser.py``, the CLI, and the tests.

Resolution order (low → high precedence):
1. Built-in defaults
2. TOML file at ``$XDG_CONFIG_HOME/autosre/dropbox.toml`` (or an explicit
   ``--config-file`` path); missing files are ignored silently
3. Environment variables (``AUTOSRE_DROPBOX_*``)

There is intentionally no ``from_env()`` shortcut or any other constructor.
Every entry point — proxy ``__main__``, installer, CLI commands, tests —
goes through ``DropboxConfig.load`` so the precedence rules apply uniformly.

Passwords never appear on this dataclass. Credential handling lives in
``autosre.dropbox.credentials``.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from autosre.paths import config_dir as autosre_config_dir

DEFAULT_DATA_DIR = Path("/data/dropbox")
DEFAULT_LISTEN_PORT = 8443
DEFAULT_UPSTREAM_PORT = 18443
DEFAULT_BIND_ADDR = "0.0.0.0"  # noqa: S104 — public bind is the entire point
DEFAULT_COOKIE_TTL_SECONDS = 7 * 24 * 3600
DEFAULT_FILEBROWSER_BIN = "filebrowser"


def _default_config_file() -> Path:
    """Default TOML location: ``<XDG config>/autosre/dropbox.toml``."""
    return autosre_config_dir() / "dropbox.toml"


@dataclass(frozen=True)
class DropboxConfig:
    """Resolved dropbox configuration.

    Construct via :meth:`DropboxConfig.load`; never instantiate directly
    outside of tests.
    """

    data_dir: Path = DEFAULT_DATA_DIR
    files_dir: Path = field(default=Path("/data/dropbox/files"))
    config_dir: Path = field(default=Path("/data/dropbox/config"))
    tls_dir: Path = field(default=Path("/data/dropbox/tls"))
    state_dir: Path = field(default=Path("/data/dropbox/state"))
    bind_addr: str = DEFAULT_BIND_ADDR
    listen_port: int = DEFAULT_LISTEN_PORT
    upstream_port: int = DEFAULT_UPSTREAM_PORT
    cookie_ttl_seconds: int = DEFAULT_COOKIE_TTL_SECONDS
    filebrowser_bin: str = DEFAULT_FILEBROWSER_BIN

    # ------------------------------------------------------------------
    # Derived path helpers — every consumer goes through these so the
    # ``data_dir`` override propagates everywhere automatically.
    # ------------------------------------------------------------------

    @property
    def cert_file(self) -> Path:
        return self.tls_dir / "cert.pem"

    @property
    def key_file(self) -> Path:
        return self.tls_dir / "key.pem"

    @property
    def secret_file(self) -> Path:
        return self.config_dir / "proxy-secret"

    @property
    def password_file(self) -> Path:
        return self.config_dir / "admin-password"

    @property
    def filebrowser_db(self) -> Path:
        return self.config_dir / "filebrowser.db"

    @property
    def upstream(self) -> tuple[str, int]:
        return ("127.0.0.1", self.upstream_port)

    # ------------------------------------------------------------------
    # Loader
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, config_file: Path | None = None) -> DropboxConfig:
        """Resolve a ``DropboxConfig`` from defaults → file → env.

        Args:
            config_file: explicit TOML path. If ``None``, we look at the
                ``AUTOSRE_DROPBOX_CONFIG_FILE`` env var (so systemd units
                can point at a runbook-installed TOML), and finally fall
                back to ``$XDG_CONFIG_HOME/autosre/dropbox.toml``. Missing
                files are silently ignored at any layer.
        """
        cfg = cls()  # built-in defaults
        cfg = _apply_data_dir(cfg, cfg.data_dir)  # propagate default subdirs

        if config_file is not None:
            path = config_file
        elif "AUTOSRE_DROPBOX_CONFIG_FILE" in os.environ:
            path = Path(os.environ["AUTOSRE_DROPBOX_CONFIG_FILE"])
        else:
            path = _default_config_file()

        cfg = _apply_toml(cfg, path)
        return _apply_env(cfg, os.environ)


# ----------------------------------------------------------------------
# Layer appliers — each one returns a new DropboxConfig (frozen).
# ----------------------------------------------------------------------


def _apply_data_dir(cfg: DropboxConfig, data_dir: Path) -> DropboxConfig:
    """Re-derive subdirs from a new ``data_dir`` (only when the caller
    has not overridden them explicitly).
    """
    return replace(
        cfg,
        data_dir=data_dir,
        files_dir=data_dir / "files",
        config_dir=data_dir / "config",
        tls_dir=data_dir / "tls",
        state_dir=data_dir / "state",
    )


def _apply_toml(cfg: DropboxConfig, path: Path) -> DropboxConfig:
    if not path.exists():
        return cfg
    with path.open("rb") as fp:
        raw: dict[str, Any] = tomllib.load(fp)
    section: dict[str, Any] = raw.get("dropbox", raw)  # accept top-level OR [dropbox]

    # data_dir first so subdir defaults follow before explicit overrides
    if "data_dir" in section:
        cfg = _apply_data_dir(cfg, Path(str(section["data_dir"])))

    overrides: dict[str, Any] = {}
    for path_field in ("files_dir", "config_dir", "tls_dir", "state_dir"):
        if path_field in section:
            overrides[path_field] = Path(str(section[path_field]))
    for str_field in ("bind_addr", "filebrowser_bin"):
        if str_field in section:
            overrides[str_field] = str(section[str_field])
    for int_field in ("listen_port", "upstream_port", "cookie_ttl_seconds"):
        if int_field in section:
            overrides[int_field] = int(section[int_field])

    return replace(cfg, **overrides) if overrides else cfg


_ENV_PREFIX = "AUTOSRE_DROPBOX_"
_ENV_PATH_FIELDS = ("DATA_DIR", "FILES_DIR", "CONFIG_DIR", "TLS_DIR", "STATE_DIR")
_ENV_STR_FIELDS = ("BIND_ADDR", "FILEBROWSER_BIN")
_ENV_INT_FIELDS = ("LISTEN_PORT", "UPSTREAM_PORT", "COOKIE_TTL_SECONDS")


def _apply_env(cfg: DropboxConfig, env: os._Environ[str] | dict[str, str]) -> DropboxConfig:
    if f"{_ENV_PREFIX}DATA_DIR" in env:
        cfg = _apply_data_dir(cfg, Path(env[f"{_ENV_PREFIX}DATA_DIR"]))

    overrides: dict[str, Any] = {}
    for name in _ENV_PATH_FIELDS:
        if name == "DATA_DIR":
            continue
        key = f"{_ENV_PREFIX}{name}"
        if key in env:
            overrides[name.lower()] = Path(env[key])
    for name in _ENV_STR_FIELDS:
        key = f"{_ENV_PREFIX}{name}"
        if key in env:
            overrides[name.lower()] = env[key]
    for name in _ENV_INT_FIELDS:
        key = f"{_ENV_PREFIX}{name}"
        if key in env:
            overrides[name.lower()] = int(env[key])

    return replace(cfg, **overrides) if overrides else cfg
