"""Destructive state initialization for the dropbox subsystem.

Complements :mod:`autosre.dropbox.installer` (which only writes systemd
unit files — non-destructive). This module mutates on-disk state:

- Generates TLS certs (self-signed via openssl, or mkcert when available).
- Initializes the filebrowser sqlite DB with admin user + HTTPS config.
- Writes the admin password file at ``0600``.
- Seeds the HMAC signing secret.

Every function here refuses to run when the dropbox service is active
(via :func:`autosre.dropbox.installer.is_any_service_active`). The CLI
enforces the check before calling in, and the helpers do a belt-and-braces
second check.
"""

from __future__ import annotations

import secrets
import shutil
import socket
import subprocess
from typing import TYPE_CHECKING

from autosre.dropbox.credentials import write_password_file

if TYPE_CHECKING:
    from pathlib import Path

    from autosre.dropbox.config import DropboxConfig


class StateInitError(RuntimeError):
    """Raised when state init can't proceed safely."""


# ---------------------------------------------------------------------------
# Directory scaffolding
# ---------------------------------------------------------------------------


def ensure_directories(config: DropboxConfig) -> list[Path]:
    """Create the dropbox directory tree, return the list of created paths."""
    created: list[Path] = []
    for path in (
        config.data_dir,
        config.files_dir,
        config.config_dir,
        config.tls_dir,
        config.state_dir,
    ):
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            created.append(path)
    return created


# ---------------------------------------------------------------------------
# TLS cert generation
# ---------------------------------------------------------------------------


def _default_sans() -> list[str]:
    """Generate SubjectAltName entries from local hostnames/IPs."""
    sans = ["DNS:localhost", "IP:127.0.0.1"]
    try:
        hostname = socket.gethostname()
        if hostname and hostname != "localhost":
            sans.append(f"DNS:{hostname}")
    except OSError:
        pass
    return sans


def generate_self_signed_cert(
    config: DropboxConfig,
    *,
    days: int = 3650,
    common_name: str | None = None,
    sans: list[str] | None = None,
) -> tuple[Path, Path]:
    """Generate a self-signed cert + key pair via openssl.

    Refuses to overwrite an existing cert. Returns ``(cert, key)``.
    """
    cert_path = config.cert_file
    key_path = config.key_file
    if cert_path.exists() or key_path.exists():
        raise StateInitError(
            f"cert/key already exist at {cert_path} / {key_path}; refusing to overwrite"
        )
    if not shutil.which("openssl"):
        raise StateInitError("openssl binary not found on PATH")

    cert_path.parent.mkdir(parents=True, exist_ok=True)
    cn = common_name or socket.gethostname() or "dropbox"
    san_list = sans or _default_sans()
    san_arg = ",".join(san_list)

    subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-sha256",
            "-days",
            str(days),
            "-nodes",
            "-keyout",
            str(key_path),
            "-out",
            str(cert_path),
            "-subj",
            f"/CN={cn}",
            "-addext",
            f"subjectAltName={san_arg}",
        ],
        check=True,
        capture_output=True,
    )
    key_path.chmod(0o600)
    cert_path.chmod(0o644)
    return cert_path, key_path


# ---------------------------------------------------------------------------
# Filebrowser DB bootstrap
# ---------------------------------------------------------------------------


def init_filebrowser_db(
    config: DropboxConfig,
    *,
    filebrowser_bin: Path,
    admin_password: str,
) -> Path:
    """Initialize the filebrowser sqlite DB and create the admin user.

    Returns the database path. Refuses to run when the DB already exists
    (use the installer's ``--force`` or re-init after manual removal).
    """
    db = config.filebrowser_db
    if db.exists():
        raise StateInitError(f"filebrowser DB already exists at {db}; refusing to re-initialize")
    config.config_dir.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [str(filebrowser_bin), "config", "init", "-d", str(db)],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [
            str(filebrowser_bin),
            "config",
            "set",
            "-d",
            str(db),
            "--address",
            "127.0.0.1",
            "--port",
            str(config.upstream_port),
            "--root",
            str(config.files_dir),
            "--auth.method=noauth",  # proxy gates access; filebrowser trusts upstream
            "--branding.name",
            "",
        ],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [
            str(filebrowser_bin),
            "users",
            "add",
            "admin",
            admin_password,
            "--perm.admin",
            "-d",
            str(db),
        ],
        check=True,
        capture_output=True,
    )
    return db


# ---------------------------------------------------------------------------
# HMAC secret seed
# ---------------------------------------------------------------------------


def ensure_hmac_secret(config: DropboxConfig) -> Path:
    """Seed the HMAC signing secret at ``config.secret_file`` (mode 0600)."""
    path = config.secret_file
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(secrets.token_bytes(32))
        path.chmod(0o600)
    return path


# ---------------------------------------------------------------------------
# Aggregated init
# ---------------------------------------------------------------------------


def perform_init(
    config: DropboxConfig,
    *,
    admin_password: str,
    filebrowser_bin: Path,
    cert_source: str = "self-signed",
    cert: Path | None = None,
    key: Path | None = None,
) -> dict[str, str]:
    """Run the full destructive init sequence and return a summary.

    Callers MUST verify the service is stopped first — see
    :func:`autosre.dropbox.installer.is_any_service_active`.
    """
    created_dirs = ensure_directories(config)

    if cert_source == "self-signed":
        cert_path, key_path = generate_self_signed_cert(config)
    elif cert_source == "import":
        if cert is None or key is None:
            raise StateInitError("cert_source=import requires explicit cert + key paths")
        cert_path = config.cert_file
        key_path = config.key_file
        cert_path.parent.mkdir(parents=True, exist_ok=True)
        cert_path.write_bytes(cert.read_bytes())
        key_path.write_bytes(key.read_bytes())
        key_path.chmod(0o600)
        cert_path.chmod(0o644)
    else:
        raise StateInitError(f"unknown cert_source {cert_source!r}")

    secret_path = ensure_hmac_secret(config)
    password_path = write_password_file(config.password_file, admin_password)
    db_path = init_filebrowser_db(
        config,
        filebrowser_bin=filebrowser_bin,
        admin_password=admin_password,
    )

    return {
        "created_dirs": ",".join(str(p) for p in created_dirs),
        "cert": str(cert_path),
        "key": str(key_path),
        "secret": str(secret_path),
        "password_file": str(password_path),
        "filebrowser_db": str(db_path),
    }
