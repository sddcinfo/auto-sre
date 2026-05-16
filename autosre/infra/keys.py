"""SSH key management for operational tasks.

Generates and tracks SSH key pairs used by autosre to run
commands on remote hosts (GB10 nodes, lab gear, jump boxes, etc.)
without relying on interactive password prompts.

Also provides ssh-agent integration: detecting a usable agent socket,
loading keys into the agent, and installing a managed block in
``~/.ssh/config`` so ``ssh user@host`` transparently uses the agent
without per-host ``IdentityFile`` directives.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

DEFAULT_KEY_DIR = Path.home() / ".ssh"
DEFAULT_KEY_NAME = "autosre_ed25519"
VALID_KEY_TYPES = ("ed25519", "rsa", "ecdsa")

SSH_CONFIG_PATH = Path.home() / ".ssh" / "config"
MANAGED_BLOCK_BEGIN = "# BEGIN autosre ssh-agent (managed — do not edit)"
MANAGED_BLOCK_END = "# END autosre ssh-agent"


@dataclass(frozen=True)
class SSHKeyPair:
    """A private/public SSH key pair on disk."""

    name: str
    private_key: Path
    public_key: Path

    @property
    def exists(self) -> bool:
        return self.private_key.exists() and self.public_key.exists()

    def read_public_key(self) -> str:
        return self.public_key.read_text().strip()


class SSHKeyManager:
    """Manage SSH key pairs used by autosre for remote operations."""

    def __init__(self, key_dir: Path | None = None) -> None:
        self.key_dir = Path(key_dir) if key_dir else DEFAULT_KEY_DIR

    def _ensure_key_dir(self) -> None:
        self.key_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.key_dir.chmod(0o700)

    def path_for(self, name: str) -> SSHKeyPair:
        if "/" in name or name in {"", ".", ".."}:
            raise ValueError(f"Invalid key name: {name!r}")
        private = self.key_dir / name
        public = self.key_dir / f"{name}.pub"
        return SSHKeyPair(name=name, private_key=private, public_key=public)

    def list_keys(self) -> list[SSHKeyPair]:
        if not self.key_dir.exists():
            return []
        pairs: list[SSHKeyPair] = []
        for pub in sorted(self.key_dir.glob("*.pub")):
            priv = pub.with_suffix("")
            if priv.exists():
                pairs.append(SSHKeyPair(name=priv.name, private_key=priv, public_key=pub))
        return pairs

    def generate(
        self,
        name: str = DEFAULT_KEY_NAME,
        *,
        key_type: str = "ed25519",
        comment: str | None = None,
        force: bool = False,
    ) -> SSHKeyPair:
        """Generate a new passphrase-less SSH key pair.

        If a pair with this name already exists and ``force`` is False,
        the existing pair is returned unchanged (idempotent).
        """
        if key_type not in VALID_KEY_TYPES:
            raise ValueError(
                f"Unsupported key type: {key_type!r} (expected one of {VALID_KEY_TYPES})",
            )
        if shutil.which("ssh-keygen") is None:
            raise RuntimeError("ssh-keygen not found on PATH")

        self._ensure_key_dir()
        pair = self.path_for(name)

        if pair.exists:
            if not force:
                return pair
            pair.private_key.unlink(missing_ok=True)
            pair.public_key.unlink(missing_ok=True)

        cmd = [
            "ssh-keygen",
            "-t",
            key_type,
            "-f",
            str(pair.private_key),
            "-N",
            "",
            "-q",
        ]
        if comment:
            cmd.extend(["-C", comment])

        result = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=30)
        if result.returncode != 0:
            msg = result.stderr.strip() or result.stdout.strip() or "unknown error"
            raise RuntimeError(f"ssh-keygen failed: {msg}")

        pair.private_key.chmod(0o600)
        pair.public_key.chmod(0o644)
        return pair

    def copy_id_command(self, target: str, pair: SSHKeyPair) -> list[str]:
        """Build the ``ssh-copy-id`` command to install this key on ``target``.

        The caller runs this themselves so they can enter the remote
        password interactively (ssh-copy-id cannot be fed a password by
        autosre without weakening security).
        """
        if "@" not in target:
            raise ValueError(f"target must be 'user@host', got {target!r}")
        return ["ssh-copy-id", "-i", str(pair.public_key), target]


def detect_agent_socket() -> Path | None:
    """Return a usable ssh-agent socket path, or None.

    Priority:
      1. ``$XDG_RUNTIME_DIR/gcr/ssh`` (gnome-keyring GCR agent, socket-activated).
      2. ``$SSH_AUTH_SOCK`` if set and the path exists.
      3. ``None`` — no agent available.
    """
    xdg = os.environ.get("XDG_RUNTIME_DIR")
    if xdg:
        gcr = Path(xdg) / "gcr" / "ssh"
        if gcr.exists():
            return gcr
    env_sock = os.environ.get("SSH_AUTH_SOCK")
    if env_sock:
        sock = Path(env_sock)
        if sock.exists():
            return sock
    return None


def _agent_env(socket: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["SSH_AUTH_SOCK"] = str(socket)
    return env


def agent_keys(socket: Path) -> list[str]:
    """Return fingerprint lines currently loaded in the agent at ``socket``.

    Treats exit code 1 (agent has no identities) as an empty list.
    Exit code 2 (cannot contact agent) raises ``RuntimeError``.
    """
    if shutil.which("ssh-add") is None:
        raise RuntimeError("ssh-add not found on PATH")
    result = subprocess.run(
        ["ssh-add", "-l"],
        capture_output=True,
        text=True,
        check=False,
        env=_agent_env(socket),
        timeout=10,
    )
    if result.returncode == 0:
        return [line for line in result.stdout.splitlines() if line.strip()]
    if result.returncode == 1:
        return []
    msg = result.stderr.strip() or result.stdout.strip() or "unknown error"
    raise RuntimeError(f"ssh-add -l failed (rc={result.returncode}): {msg}")


def _public_key_fingerprint(pair: SSHKeyPair) -> str | None:
    """Compute the SHA256 fingerprint of a public key via ``ssh-keygen -lf``.

    Returns the raw fingerprint token (e.g. ``SHA256:...``) or None on failure.
    """
    if shutil.which("ssh-keygen") is None:
        return None
    if not pair.public_key.exists():
        return None
    result = subprocess.run(
        ["ssh-keygen", "-lf", str(pair.public_key)],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    if result.returncode != 0:
        return None
    parts = result.stdout.strip().split()
    for token in parts:
        if token.startswith(("SHA256:", "MD5:")):
            return token
    return None


def add_to_agent(pair: SSHKeyPair, socket: Path) -> bool:
    """Load ``pair`` into the ssh-agent at ``socket``. Idempotent.

    Returns True if the key was newly added, False if it was already present.
    Raises ``RuntimeError`` on failure to contact the agent or load the key.
    """
    if not pair.exists:
        raise RuntimeError(f"key pair does not exist: {pair.private_key}")
    if shutil.which("ssh-add") is None:
        raise RuntimeError("ssh-add not found on PATH")

    fingerprint = _public_key_fingerprint(pair)
    if fingerprint is not None:
        loaded = agent_keys(socket)
        if any(fingerprint in line for line in loaded):
            return False

    result = subprocess.run(
        ["ssh-add", str(pair.private_key)],
        capture_output=True,
        text=True,
        check=False,
        env=_agent_env(socket),
        timeout=15,
    )
    if result.returncode != 0:
        msg = result.stderr.strip() or result.stdout.strip() or "unknown error"
        raise RuntimeError(f"ssh-add failed: {msg}")
    return True


def _build_managed_block(socket: Path) -> str:
    return (
        f"{MANAGED_BLOCK_BEGIN}\n"
        "Host *\n"
        "    AddKeysToAgent yes\n"
        f"    IdentityAgent {socket}\n"
        f"{MANAGED_BLOCK_END}\n"
    )


def _strip_managed_block(content: str) -> str:
    """Remove any existing autosre managed block from ``content``."""
    lines = content.splitlines(keepends=True)
    out: list[str] = []
    in_block = False
    for line in lines:
        stripped = line.strip()
        if not in_block and stripped == MANAGED_BLOCK_BEGIN:
            in_block = True
            continue
        if in_block:
            if stripped == MANAGED_BLOCK_END:
                in_block = False
            continue
        out.append(line)
    return "".join(out)


def ensure_config_uses_agent(
    socket: Path,
    config_path: Path | None = None,
) -> bool:
    """Idempotently install a managed ``Host *`` block in ``~/.ssh/config``.

    The block sets ``AddKeysToAgent yes`` and ``IdentityAgent <socket>`` so
    the ssh client uses the specified agent regardless of environment vars.

    The block is placed at the top of the file (ssh config is
    first-match-wins; top placement is conventional for defaults).

    Returns True if the file was created or modified, False if it was
    already up-to-date.
    """
    path = config_path if config_path is not None else SSH_CONFIG_PATH
    desired_block = _build_managed_block(socket)

    if not path.exists():
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        path.write_text(desired_block)
        path.chmod(0o600)
        return True

    existing = path.read_text()
    remainder = _strip_managed_block(existing)

    # Normalize: managed block first, then remainder. Insert a blank line
    # between the block and remainder if remainder is non-empty and doesn't
    # already start with a newline.
    if remainder and not remainder.startswith("\n"):
        new_content = desired_block + "\n" + remainder
    else:
        new_content = desired_block + remainder

    if new_content == existing:
        # Already up-to-date; still ensure mode is 0600.
        path.chmod(0o600)
        return False

    path.write_text(new_content)
    path.chmod(0o600)
    return True
