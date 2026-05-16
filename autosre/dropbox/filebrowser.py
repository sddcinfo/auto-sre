"""filebrowser binary fetch + cache helper.

Downloads the upstream ``filebrowser/filebrowser`` release binary to a
location under ``$XDG_DATA_HOME/autosre/dropbox/bin/`` if one isn't already
present on PATH, verifies the SHA256 against a pinned table, and returns a
``Path`` the installer can wire into the systemd unit.

Design choices:

- **Pinned version by default.** We don't chase latest — reproducible
  installs matter more than a few weeks of features.
- **SHA256 verification.** Refuses to install a tarball whose checksum
  doesn't match the pinned table.
- **PATH fallback.** If ``shutil.which("filebrowser")`` already finds a
  binary, we use it. Users who manage their own package get to keep
  doing that.
- **Arch detection.** ``uname -m`` → ``amd64``/``arm64``. Anything else
  errors out cleanly.
"""

from __future__ import annotations

import hashlib
import platform
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

from autosre.paths import data_dir as autosre_data_dir

DEFAULT_VERSION = "v2.63.2"

# Pinned SHA256 of the tarball for each (version, arch).
# Source of truth: the upstream release assets on
# https://github.com/filebrowser/filebrowser/releases
# Update alongside version bumps.
_PINNED_SHA256: dict[tuple[str, str], str] = {
    ("v2.63.2", "amd64"): "5a6bb687af0a4cf6148a6e09b6fc45f60e8d4b159db37b7138f81fc97033a9bb",
    ("v2.63.2", "arm64"): "246938e22a1d44caae43f114eb087a8553f4fa008fb01155e1acd89a80d257f1",
}


class FilebrowserInstallError(RuntimeError):
    """Raised when we can't produce a usable filebrowser binary."""


def _detect_arch() -> str:
    machine = platform.machine().lower()
    if machine in ("x86_64", "amd64"):
        return "amd64"
    if machine in ("aarch64", "arm64"):
        return "arm64"
    raise FilebrowserInstallError(
        f"unsupported architecture {machine!r}; filebrowser releases only "
        "ship amd64 + arm64 linux tarballs"
    )


def _cache_dir() -> Path:
    directory = autosre_data_dir() / "dropbox" / "bin"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _cached_path(version: str) -> Path:
    return _cache_dir() / f"filebrowser-{version}"


def _download_url(version: str, arch: str) -> str:
    return (
        f"https://github.com/filebrowser/filebrowser/releases/download/"
        f"{version}/linux-{arch}-filebrowser.tar.gz"
    )


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def find_or_download(
    *,
    version: str = DEFAULT_VERSION,
    prefer_path: bool = True,
) -> Path:
    """Return a path to a usable filebrowser binary.

    1. If ``prefer_path`` and ``filebrowser`` is on ``PATH``, return that.
    2. If the pinned version is already cached under XDG data, return it.
    3. Otherwise download + verify + extract into the cache.

    Raises :class:`FilebrowserInstallError` if the SHA256 doesn't match or
    the download fails.
    """
    if prefer_path:
        on_path = shutil.which("filebrowser")
        if on_path:
            return Path(on_path)

    cached = _cached_path(version)
    if cached.exists() and cached.is_file():
        return cached

    arch = _detect_arch()
    key = (version, arch)
    if key not in _PINNED_SHA256:
        raise FilebrowserInstallError(
            f"no pinned sha256 for filebrowser {version} on {arch}; "
            "update _PINNED_SHA256 in autosre/dropbox/filebrowser.py"
        )
    expected_sha = _PINNED_SHA256[key]
    url = _download_url(version, arch)

    with tempfile.TemporaryDirectory() as tmp:
        tarball = Path(tmp) / f"filebrowser-{version}.tar.gz"
        try:
            with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310
                tarball_bytes = response.read()
        except OSError as exc:
            raise FilebrowserInstallError(f"download failed: {exc}") from exc

        actual_sha = _sha256(tarball_bytes)
        if actual_sha != expected_sha:
            raise FilebrowserInstallError(
                f"sha256 mismatch for filebrowser {version} {arch}: "
                f"expected {expected_sha}, got {actual_sha}"
            )
        tarball.write_bytes(tarball_bytes)

        extract_dir = Path(tmp) / "extracted"
        extract_dir.mkdir()
        with tarfile.open(tarball, "r:gz") as tf:
            tf.extractall(extract_dir, filter="data")

        extracted_bin = extract_dir / "filebrowser"
        if not extracted_bin.exists():
            raise FilebrowserInstallError(f"filebrowser binary not found in tarball {url}")

        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_bytes(extracted_bin.read_bytes())
        cached.chmod(0o755)

    return cached


def cache_dir() -> Path:
    """Public accessor — tests + CLI can call this to show where binaries live."""
    return _cache_dir()
