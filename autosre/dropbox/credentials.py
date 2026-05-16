"""Secure credential handling for the dropbox admin password.

Three supported input sources (no ``--password`` literal anywhere):

1. **Interactive TTY prompt** — :func:`prompt_interactive`. Used when
   stdin is a TTY. Hides input, asks twice, length-checks.
2. **stdin** — :func:`read_stdin`. Reads one line from stdin; for piping
   from a secret manager (``pass show ... | autosre dropbox init --password-stdin``).
3. **password file** — :func:`read_password_file`. Reads from a file we
   verify is mode ``0600`` and owned by the invoking user.

Storage on disk is always via :func:`write_password_file`, which uses
``os.open`` with ``O_WRONLY|O_CREAT|O_TRUNC`` and mode ``0o600`` so the
file is never briefly world-readable between creation and chmod.

Filebrowser enforces a 12-character minimum itself; we match that.
"""

from __future__ import annotations

import getpass
import os
import stat
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

MIN_PASSWORD_LENGTH = 12


class PasswordError(ValueError):
    """Raised when a password fails validation (length, mode, ownership)."""


def _validate(password: str) -> str:
    stripped = password.rstrip("\n")
    if len(stripped) < MIN_PASSWORD_LENGTH:
        raise PasswordError(
            f"password must be at least {MIN_PASSWORD_LENGTH} characters (filebrowser minimum)"
        )
    return stripped


def prompt_interactive(*, confirm: bool = True) -> str:
    """Prompt the user for a password on the TTY (hidden input)."""
    if not sys.stdin.isatty():
        raise PasswordError("stdin is not a TTY; use --password-stdin or --password-file")
    while True:
        first = getpass.getpass("Admin password: ")
        if not confirm:
            return _validate(first)
        second = getpass.getpass("Confirm password: ")
        if first != second:
            print("Passwords did not match; try again.", file=sys.stderr)
            continue
        return _validate(first)


def read_stdin() -> str:
    """Read a single line from stdin (for piping from a secret manager)."""
    line = sys.stdin.readline()
    if not line:
        raise PasswordError("--password-stdin received no input")
    return _validate(line)


def read_password_file(path: Path) -> str:
    """Read a password file, verifying ownership + mode before trusting it."""
    if not path.exists():
        raise PasswordError(f"password file not found: {path}")
    info = path.stat()
    if info.st_uid != os.getuid():
        raise PasswordError(
            f"password file {path} is not owned by the invoking user "
            f"(uid {info.st_uid} vs {os.getuid()})"
        )
    mode = stat.S_IMODE(info.st_mode)
    if mode & 0o077:
        raise PasswordError(f"password file {path} has permissive mode {oct(mode)}; expected 0600")
    content = path.read_text()
    # Accept either a bare password file or an ADMIN_PW=... line.
    for line in content.splitlines():
        if line.startswith("ADMIN_PW="):
            return _validate(line.split("=", 1)[1])
        stripped = line.strip()
        if stripped and not line.startswith("#"):
            return _validate(stripped)
    raise PasswordError(f"no password content in {path}")


def write_password_file(path: Path, password: str) -> Path:
    """Atomically write a password file with mode 0600.

    Uses ``os.open`` with the mode set at creation time so the file is
    never momentarily readable by other users.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, f"ADMIN_PW={password}\n".encode())
    finally:
        os.close(fd)
    path.chmod(0o600)  # belt-and-braces in case umask munged it
    return path


def verify_password_file_mode(path: Path) -> tuple[bool, str]:
    """Used by ``status`` to warn on drift.

    Returns ``(ok, detail)``. ``ok`` is False when the file is missing or
    has had its permissions relaxed since install.
    """
    if not path.exists():
        return False, f"password file missing: {path}"
    info = path.stat()
    mode = stat.S_IMODE(info.st_mode)
    if mode & 0o077:
        return False, f"password file mode {oct(mode)} (expected 0600)"
    return True, f"password file ok (mode {oct(mode)})"


def resolve_password(
    *,
    from_stdin: bool,
    from_file: Path | None,
) -> str:
    """Dispatch to one of the three supported input paths.

    Called by :func:`autosre dropbox init` and :func:`autosre dropbox passwd`.
    Exactly one of ``from_stdin`` / ``from_file`` may be set; if neither
    is, we fall through to :func:`prompt_interactive`.
    """
    if from_stdin and from_file is not None:
        raise PasswordError("pick one of --password-stdin or --password-file, not both")
    if from_stdin:
        return read_stdin()
    if from_file is not None:
        return read_password_file(from_file)
    return prompt_interactive()
