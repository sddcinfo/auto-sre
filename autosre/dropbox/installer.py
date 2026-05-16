"""OS-aware dropbox service installer.

Installs/uninstalls/queries the dropbox systemd units. Models itself on
:mod:`autosre.hooks_installer`: idempotent install, sidecar tracking in
``$XDG_STATE_HOME/autosre/.dropbox-installed.json``, surgical uninstall.

Design invariants:

- **Init-system detection is operability-based**, not binary-presence-based.
  We never trust ``systemctl --user is-system-running`` because it returns
  non-zero on ``degraded`` systems that are otherwise fully operable. We
  probe ``systemctl --user show-environment`` (proves the user manager is
  reachable) and a harmless ``systemctl --user list-units`` (proves unit
  operations work).
- **User/home resolution goes through ``pwd``**, never raw environment
  variables. Under ``sudo`` ``$USER`` and ``$HOME`` are typically wrong;
  ``pwd.getpwnam(name)`` gives us the authoritative record.
- **System mode requires an explicit ``--service-user``**. We never guess
  which user a system service should run as.
- **Sidecar captures exactly what was installed**: unit file paths, content
  hashes, binary path, config file path, mode (user/system), service user.
  Uninstall uses the sidecar to remove only what we installed.
- **Init operation is separate**. This installer writes/removes unit files
  and controls services; it does not touch certs, databases, or passwords.
  Destructive state operations live in :mod:`autosre.dropbox.init`.
"""

from __future__ import annotations

import hashlib
import json
import os
import pwd
import shutil
import subprocess
import sys
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from autosre.paths import state_dir as autosre_state_dir

if TYPE_CHECKING:
    from autosre.dropbox.config import DropboxConfig


class InstallMode(StrEnum):
    """Where systemd units live."""

    USER = "user"
    SYSTEM = "system"


class InitSystem(StrEnum):
    """Detected init system (for diagnostic clarity)."""

    SYSTEMD = "systemd"
    OPENRC = "openrc"
    RUNIT = "runit"
    LAUNCHD = "launchd"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ResolvedUser:
    """A real account record from ``pwd``, not ``os.environ``."""

    name: str
    uid: int
    gid: int
    home: Path


@dataclass(frozen=True)
class InstallContext:
    """Everything the installer needs to render a unit file."""

    config: DropboxConfig
    mode: InstallMode
    user: ResolvedUser
    python_bin: Path
    filebrowser_bin: Path
    repo_dir: Path  # where autosre is importable from
    config_file: Path | None  # TOML config path (None = default XDG)


# ---------------------------------------------------------------------------
# Init-system detection
# ---------------------------------------------------------------------------


def detect_init_system() -> InitSystem:
    """Return the init system for diagnostic purposes.

    Does **not** guarantee the system is usable for installs — see
    :func:`probe_systemd_user_operable` and :func:`probe_systemd_system_operable`
    for the operability checks.
    """
    if shutil.which("systemctl"):
        return InitSystem.SYSTEMD
    if Path("/sbin/openrc-run").exists() or shutil.which("rc-service"):
        return InitSystem.OPENRC
    if shutil.which("sv") and Path("/etc/runit").exists():
        return InitSystem.RUNIT
    if shutil.which("launchctl"):
        return InitSystem.LAUNCHD
    return InitSystem.UNKNOWN


def probe_systemd_user_operable() -> tuple[bool, str]:
    """Return ``(ok, detail)`` for whether ``systemctl --user`` works.

    Uses two independent operability probes and does **not** look at
    ``is-system-running`` exit status (which fails on ``degraded`` systems
    that are otherwise fully functional):

    1. ``systemctl --user show-environment`` — fails with "Failed to
       connect to bus" when the user manager is unreachable.
    2. ``systemctl --user list-units --type=service --state=loaded
       --no-legend --no-pager`` — exercises the same code path install/start
       use.
    """
    if not shutil.which("systemctl"):
        return False, "systemctl binary not found on PATH"
    env_result = subprocess.run(
        ["systemctl", "--user", "show-environment"],
        capture_output=True,
        text=True,
        check=False,
    )
    if env_result.returncode != 0:
        return False, f"systemctl --user show-environment failed: {env_result.stderr.strip()}"
    list_result = subprocess.run(
        [
            "systemctl",
            "--user",
            "list-units",
            "--type=service",
            "--state=loaded",
            "--no-legend",
            "--no-pager",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if list_result.returncode != 0:
        return False, f"systemctl --user list-units failed: {list_result.stderr.strip()}"
    return True, "user bus reachable, list-units ok"


def probe_systemd_system_operable() -> tuple[bool, str]:
    """Return ``(ok, detail)`` for whether system-mode systemctl works.

    System mode ultimately needs ``sudo`` for install and start. Here we
    only verify that ``systemctl`` can query the system manager at all.
    """
    if not shutil.which("systemctl"):
        return False, "systemctl binary not found on PATH"
    result = subprocess.run(
        [
            "systemctl",
            "list-units",
            "--type=service",
            "--state=loaded",
            "--no-legend",
            "--no-pager",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return False, f"systemctl list-units failed: {result.stderr.strip()}"
    return True, "system manager reachable"


# ---------------------------------------------------------------------------
# User / home resolution
# ---------------------------------------------------------------------------


def resolve_user(service_user: str | None, mode: InstallMode) -> ResolvedUser:
    """Resolve the account a service will run as.

    - ``--user`` mode: always the invoking user (via ``pwd.getpwuid``),
      never environment variables. ``--service-user`` is rejected.
    - ``--system`` mode: requires ``--service-user``. Resolved via
      ``pwd.getpwnam``.
    """
    if mode is InstallMode.USER:
        if service_user is not None:
            raise ValueError(
                "--service-user is only valid with --system; user-mode services "
                "always run as the invoking user"
            )
        record = pwd.getpwuid(os.getuid())
    else:  # SYSTEM
        if service_user is None:
            raise ValueError(
                "--system mode requires --service-user <name>; refusing to guess the account"
            )
        try:
            record = pwd.getpwnam(service_user)
        except KeyError as exc:
            raise ValueError(f"service user {service_user!r} not found in passwd") from exc
    return ResolvedUser(
        name=record.pw_name,
        uid=record.pw_uid,
        gid=record.pw_gid,
        home=Path(record.pw_dir),
    )


# ---------------------------------------------------------------------------
# Unit file rendering
# ---------------------------------------------------------------------------


_FILEBROWSER_UNIT_TEMPLATE = """\
[Unit]
Description=Dropbox filebrowser backend
After=network-online.target
Wants=network-online.target

[Service]
Type=simple{user_directives}
WorkingDirectory={data_dir}
ExecStart={filebrowser_bin} -d {db_file} -a 127.0.0.1 -p {upstream_port} -r {files_dir}
Restart=on-failure
RestartSec=5
LimitNOFILE=65536{sandbox_directives}
NoNewPrivileges=true

[Install]
WantedBy={wanted_by}
"""

_SNIFF_UNIT_TEMPLATE = """\
[Unit]
Description=Dropbox TLS+password-gate proxy
After=network-online.target {filebrowser_unit}
Wants=network-online.target
Requires={filebrowser_unit}

[Service]
Type=simple{user_directives}
WorkingDirectory={repo_dir}
Environment=PYTHONUNBUFFERED=1{config_env}
ExecStart={python_bin} -m autosre.dropbox.proxy
Restart=on-failure
RestartSec=5
NoNewPrivileges=true{sandbox_directives}

[Install]
WantedBy={wanted_by}
"""


def render_unit_files(ctx: InstallContext) -> dict[str, str]:
    """Return ``{filename: content}`` for the dropbox units."""
    wanted_by = "default.target" if ctx.mode is InstallMode.USER else "multi-user.target"
    filebrowser_unit = "dropbox.service"
    sniff_unit = "dropbox-sniff.service"

    # User mode services run as the invoking user already; system mode needs
    # an explicit User=/Group= directive.
    if ctx.mode is InstallMode.USER:
        user_directives = ""
        # User-mode services can't use ProtectSystem=full + ReadWritePaths the
        # same way — the sandbox directives aren't available to user units in
        # the same way. Keep a minimal sandbox.
        sandbox_directives = ""
    else:
        user_directives = f"\nUser={ctx.user.name}\nGroup={ctx.user.name}"
        sandbox_directives = (
            f"\nProtectSystem=full\nProtectHome=read-only\nReadWritePaths={ctx.config.data_dir}"
        )

    config_env = ""
    if ctx.config_file is not None:
        config_env = f"\nEnvironment=AUTOSRE_DROPBOX_CONFIG_FILE={ctx.config_file}"

    filebrowser = _FILEBROWSER_UNIT_TEMPLATE.format(
        user_directives=user_directives,
        sandbox_directives=sandbox_directives,
        data_dir=ctx.config.data_dir,
        filebrowser_bin=ctx.filebrowser_bin,
        db_file=ctx.config.filebrowser_db,
        upstream_port=ctx.config.upstream_port,
        files_dir=ctx.config.files_dir,
        wanted_by=wanted_by,
    )
    sniff = _SNIFF_UNIT_TEMPLATE.format(
        user_directives=user_directives,
        sandbox_directives=sandbox_directives,
        repo_dir=ctx.repo_dir,
        python_bin=ctx.python_bin,
        filebrowser_unit=filebrowser_unit,
        config_env=config_env,
        wanted_by=wanted_by,
    )
    return {filebrowser_unit: filebrowser, sniff_unit: sniff}


def _unit_target_dir(mode: InstallMode, user: ResolvedUser) -> Path:
    if mode is InstallMode.USER:
        return user.home / ".config" / "systemd" / "user"
    return Path("/etc/systemd/system")


def _systemctl_cmd(mode: InstallMode, *args: str) -> list[str]:
    if mode is InstallMode.USER:
        return ["systemctl", "--user", *args]
    return ["sudo", "systemctl", *args]


def _sidecar_path() -> Path:
    return autosre_state_dir() / ".dropbox-installed.json"


# ---------------------------------------------------------------------------
# Sidecar
# ---------------------------------------------------------------------------


def _sha256(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()


def _load_sidecar(path: Path | None = None) -> dict[str, Any] | None:
    resolved = path or _sidecar_path()
    if not resolved.exists():
        return None
    try:
        data: dict[str, Any] = json.loads(resolved.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(data, dict):
        return None
    return data


def _save_sidecar(data: dict[str, Any], path: Path | None = None) -> Path:
    resolved = path or _sidecar_path()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(data, indent=2) + "\n")
    return resolved


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def install(ctx: InstallContext, *, force: bool = False) -> dict[str, Any]:
    """Write unit files + reload systemd.

    Non-destructive: never touches filebrowser DB, certs, or password files.
    Returns a dict describing what changed.
    """
    rendered = render_unit_files(ctx)
    target_dir = _unit_target_dir(ctx.mode, ctx.user)
    target_dir.mkdir(parents=True, exist_ok=True)

    written: list[str] = []
    skipped: list[str] = []
    diffs: list[str] = []
    for filename, content in rendered.items():
        dest = target_dir / filename
        if dest.exists():
            existing = dest.read_text()
            if existing == content:
                skipped.append(str(dest))
                continue
            if not force:
                diffs.append(str(dest))
                continue
        if ctx.mode is InstallMode.SYSTEM:
            # Use sudo tee so we can write to /etc without losing our uid.
            subprocess.run(
                ["sudo", "tee", str(dest)],
                input=content,
                text=True,
                check=True,
                stdout=subprocess.DEVNULL,
            )
            subprocess.run(
                ["sudo", "chmod", "0644", str(dest)],
                check=True,
            )
        else:
            dest.write_text(content)
            dest.chmod(0o644)
        written.append(str(dest))

    if diffs and not force:
        raise RuntimeError(
            "existing unit files differ from planned content; re-run with --force to overwrite: "
            + ", ".join(diffs)
        )

    # daemon-reload so systemd picks up the new unit contents.
    subprocess.run(
        _systemctl_cmd(ctx.mode, "daemon-reload"),
        check=True,
    )

    sidecar_data: dict[str, Any] = {
        "version": 1,
        "mode": ctx.mode.value,
        "service_user": ctx.user.name,
        "python_bin": str(ctx.python_bin),
        "filebrowser_bin": str(ctx.filebrowser_bin),
        "repo_dir": str(ctx.repo_dir),
        "config_file": str(ctx.config_file) if ctx.config_file else None,
        "config_data_dir": str(ctx.config.data_dir),
        "config_listen_port": ctx.config.listen_port,
        "config_upstream_port": ctx.config.upstream_port,
        "units": [
            {
                "path": str(target_dir / name),
                "sha256": _sha256(content),
            }
            for name, content in rendered.items()
        ],
    }
    sidecar_file = _save_sidecar(sidecar_data)

    return {
        "written": written,
        "skipped": skipped,
        "sidecar": str(sidecar_file),
        "mode": ctx.mode.value,
        "service_user": ctx.user.name,
        "target_dir": str(target_dir),
    }


def uninstall(*, purge: bool = False) -> dict[str, Any]:
    """Stop services, disable them, remove unit files, drop sidecar.

    When ``purge=True``, also removes the data directory recorded in the
    sidecar. Never removes ``files_dir`` contents by accident: purge only
    happens if the data directory is exactly the sidecar-recorded value.
    """
    sidecar = _load_sidecar()
    if sidecar is None:
        return {"removed": [], "note": "no sidecar found — nothing to uninstall"}

    mode = InstallMode(sidecar["mode"])
    removed: list[str] = []

    # Best-effort stop + disable (don't fail if already stopped)
    for unit_info in sidecar.get("units", []):
        unit_name = Path(unit_info["path"]).name
        subprocess.run(
            _systemctl_cmd(mode, "stop", unit_name),
            check=False,
            capture_output=True,
        )
        subprocess.run(
            _systemctl_cmd(mode, "disable", unit_name),
            check=False,
            capture_output=True,
        )

    # Remove the unit files
    for unit_info in sidecar.get("units", []):
        path = Path(unit_info["path"])
        if not path.exists():
            continue
        if mode is InstallMode.SYSTEM:
            subprocess.run(["sudo", "rm", "-f", str(path)], check=True)
        else:
            path.unlink()
        removed.append(str(path))

    subprocess.run(
        _systemctl_cmd(mode, "daemon-reload"),
        check=False,
    )

    _sidecar_path().unlink(missing_ok=True)

    purged: list[str] = []
    if purge:
        data_dir_str = sidecar.get("config_data_dir")
        if data_dir_str:
            data_dir = Path(data_dir_str)
            if data_dir.exists():
                if mode is InstallMode.SYSTEM:
                    subprocess.run(["sudo", "rm", "-rf", str(data_dir)], check=True)
                else:
                    shutil.rmtree(data_dir)
                purged.append(str(data_dir))

    return {
        "removed": removed,
        "purged": purged,
        "mode": mode.value,
    }


def status() -> dict[str, Any]:
    """Report installer state and drift.

    Drift = sidecar claims a unit file exists but the current content on
    disk differs from the recorded sha256 (user hand-edited the unit file,
    re-render with ``install --force``).
    """
    sidecar = _load_sidecar()
    if sidecar is None:
        return {
            "installed": False,
            "drift": [],
        }

    mode = InstallMode(sidecar["mode"])
    drift: list[dict[str, str]] = []
    active: list[dict[str, str]] = []
    for unit_info in sidecar.get("units", []):
        path = Path(unit_info["path"])
        entry: dict[str, str] = {
            "unit": path.name,
            "path": str(path),
        }
        if not path.exists():
            entry["status"] = "missing"
            drift.append(entry)
            continue
        actual_sha = _sha256(path.read_text())
        if actual_sha != unit_info["sha256"]:
            entry["status"] = "modified"
            drift.append(entry)
            continue
        is_active = subprocess.run(
            _systemctl_cmd(mode, "is-active", path.name),
            capture_output=True,
            text=True,
            check=False,
        )
        entry["status"] = "active" if is_active.returncode == 0 else "inactive"
        active.append(entry)

    return {
        "installed": True,
        "mode": mode.value,
        "service_user": sidecar.get("service_user"),
        "config_file": sidecar.get("config_file"),
        "config_data_dir": sidecar.get("config_data_dir"),
        "config_listen_port": sidecar.get("config_listen_port"),
        "units": active,
        "drift": drift,
    }


# ---------------------------------------------------------------------------
# Service control
# ---------------------------------------------------------------------------


def _unit_names_from_sidecar() -> list[str]:
    sidecar = _load_sidecar()
    if sidecar is None:
        raise RuntimeError("dropbox not installed — run `autosre dropbox install` first")
    return [Path(u["path"]).name for u in sidecar.get("units", [])]


def _mode_from_sidecar() -> InstallMode:
    sidecar = _load_sidecar()
    if sidecar is None:
        raise RuntimeError("dropbox not installed")
    return InstallMode(sidecar["mode"])


def systemctl_action(action: str) -> dict[str, Any]:
    """Run ``systemctl <action>`` on the recorded units."""
    mode = _mode_from_sidecar()
    units = _unit_names_from_sidecar()
    results: list[dict[str, Any]] = []
    for unit in units:
        result = subprocess.run(
            _systemctl_cmd(mode, action, unit),
            capture_output=True,
            text=True,
            check=False,
        )
        results.append(
            {
                "unit": unit,
                "returncode": result.returncode,
                "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip(),
            }
        )
    return {"action": action, "mode": mode.value, "results": results}


def enable_and_start() -> dict[str, Any]:
    """Enable both units at boot and start them now."""
    mode = _mode_from_sidecar()
    units = _unit_names_from_sidecar()
    subprocess.run(
        _systemctl_cmd(mode, "enable", *units),
        check=True,
    )
    subprocess.run(
        _systemctl_cmd(mode, "start", *units),
        check=True,
    )
    return {"enabled": units, "started": units, "mode": mode.value}


def is_any_service_active() -> bool:
    """Return True if any of the sidecar-recorded units is currently running.

    Used by :mod:`autosre.dropbox.state_init` to refuse destructive state
    operations while the service is live.
    """
    sidecar = _load_sidecar()
    if sidecar is None:
        return False
    mode = InstallMode(sidecar["mode"])
    for unit_info in sidecar.get("units", []):
        unit_name = Path(unit_info["path"]).name
        result = subprocess.run(
            _systemctl_cmd(mode, "is-active", unit_name),
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip() == "active":
            return True
    return False


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------


def build_install_context(
    *,
    config: DropboxConfig,
    mode: InstallMode,
    service_user: str | None,
    python_bin: Path | None,
    filebrowser_bin: Path | None,
    repo_dir: Path | None,
    config_file: Path | None,
) -> InstallContext:
    """Resolve everything the installer needs from partial CLI inputs."""
    user = resolve_user(service_user, mode)
    resolved_python = python_bin or Path(sys.executable)
    resolved_filebrowser = filebrowser_bin or Path(config.filebrowser_bin)
    resolved_repo = repo_dir or _detect_repo_dir()
    return InstallContext(
        config=config,
        mode=mode,
        user=user,
        python_bin=resolved_python,
        filebrowser_bin=resolved_filebrowser,
        repo_dir=resolved_repo,
        config_file=config_file,
    )


def _detect_repo_dir() -> Path:
    """Locate the directory from which ``autosre`` is importable.

    Used as systemd ``WorkingDirectory`` so ``python -m autosre.dropbox.proxy``
    finds the package. Falls back to the parent of this file for editable
    installs.
    """
    import autosre

    pkg_file = autosre.__file__
    if pkg_file is None:
        return Path.cwd()
    return Path(pkg_file).resolve().parent.parent
