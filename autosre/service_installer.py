"""``autosre install-service`` / ``uninstall-service`` — boot autostart.

A fresh customer GB10 expects to power-cycle and have the translate
backend (vLLM @ :8010) come back up alongside meeting-scribe. The
existing ``deploy/example/systemd/autosre.service.template`` requires
a manual ``envsubst`` + ``sudo`` dance — fine for ops engineers, hostile
for a customer install.

This module installs a *user-level* systemd unit with no sudo for
day-to-day ``start``/``stop``, and runs ``loginctl enable-linger`` so
the user manager runs at boot before login.

The unit's ``ExecStart`` runs ``autosre start --no-scribe`` because:

* The customer flow has meeting-scribe owning its own user service
  (``meeting-scribe.service``) — that's the one the user actually
  visits at ``https://<host>:8080``.
* This unit is responsible solely for the translate backend
  (vLLM @ :8010). Decoupling the two means a vLLM model swap doesn't
  cycle scribe, and a scribe redeploy doesn't bounce the model.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import click

SERVICE_NAME = "autosre.service"


def _user_unit_dir() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "systemd" / "user"


def _autosre_bin() -> Path | None:
    """Resolve the ``autosre`` entry point currently in use.

    ``shutil.which`` would just hit PATH; we want the *exact* binary
    that the running interpreter resolves to so the unit file isn't
    pointing at a stray copy from another env.
    """
    found = shutil.which("autosre")
    if found:
        return Path(found).resolve()
    # Fall back to <sys.executable's bin>/autosre — covers the
    # ``python -m autosre`` invocation path.
    candidate = Path(sys.executable).parent / "autosre"
    return candidate.resolve() if candidate.exists() else None


def _render_unit(autosre_bin: Path) -> str:
    """Render the user unit body. Path is interpolated, not env-var'd,
    so the unit doesn't drift when the user changes shells.
    """
    return f"""[Unit]
Description=Auto-SRE — local vLLM translate backend (port 8010)
Documentation=https://github.com/sddcinfo/auto-sre
# docker.service is a system-level unit not directly visible to the
# user manager; the After= still drains the right event because the
# user manager runs late in boot.
After=network-online.target docker.service
Wants=network-online.target

[Service]
# autosre's start command does a long synchronous wait for vLLM to
# become healthy (cold-load takes 3-7 minutes for Qwen3.6-35B-FP8).
# Type=simple lets systemd flag ``active`` once the process is up,
# while the user-facing ``autosre status`` already exposes the real
# vLLM health for anyone who needs it.
Type=simple
WorkingDirectory={Path.home()}
Environment=HOME={Path.home()}
Environment=PATH={autosre_bin.parent}:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
# --no-scribe: meeting-scribe runs as its own user unit
# (meeting-scribe.service). This unit owns translate vLLM only.
ExecStart={autosre_bin} start --no-scribe
ExecStop={autosre_bin} stop
Restart=on-failure
RestartSec=15
TimeoutStartSec=600
TimeoutStopSec=120

[Install]
WantedBy=default.target
"""


def _systemctl_user(*args: str) -> tuple[int, str, str]:
    proc = subprocess.run(
        ["systemctl", "--user", *args],
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()


def _enable_linger(quiet: bool) -> None:
    """Enable lingering for the invoking user.

    Without this, --user services only run while the user has an
    active login session. A fresh GB10 boots without a console
    session, so we'd lose autostart entirely.
    """
    user = os.environ.get("USER") or Path.home().name
    proc = subprocess.run(
        ["loginctl", "enable-linger", user],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode == 0:
        if not quiet:
            click.secho(f"[OK] linger enabled for user '{user}'", fg="green")
        return
    sudo_proc = subprocess.run(
        ["sudo", "-n", "loginctl", "enable-linger", user],
        capture_output=True,
        text=True,
        check=False,
    )
    if sudo_proc.returncode == 0:
        if not quiet:
            click.secho(f"[OK] linger enabled (via sudo) for user '{user}'", fg="green")
        return
    click.secho(
        f"[WARN] could not enable linger for '{user}'. Service will install "
        "but won't autostart at boot until you run:",
        fg="yellow",
    )
    click.echo(f"  sudo loginctl enable-linger {user}")


def install(
    *,
    no_enable: bool = False,
    no_start: bool = False,
    quiet: bool = False,
) -> int:
    """Install the user systemd unit. Returns process exit code."""
    if shutil.which("systemctl") is None:
        click.secho("systemctl not found — this OS doesn't use systemd.", fg="red")
        return 1

    autosre_bin = _autosre_bin()
    if autosre_bin is None or not autosre_bin.exists():
        click.secho(
            "Could not resolve the `autosre` entry point. Run "
            "`pip install -e .` inside your venv first.",
            fg="red",
        )
        return 1

    unit_dir = _user_unit_dir()
    unit_path = unit_dir / SERVICE_NAME
    unit_dir.mkdir(parents=True, exist_ok=True, mode=0o755)

    body = _render_unit(autosre_bin)
    existing = unit_path.read_text() if unit_path.exists() else None
    if existing != body:
        tmp = unit_path.with_suffix(unit_path.suffix + f".tmp.{os.getpid()}")
        tmp.write_text(body)
        tmp.replace(unit_path)
        if not quiet:
            click.secho(f"[OK] wrote {unit_path}", fg="green")
    elif not quiet:
        click.secho(f"[OK] {unit_path} already up-to-date", fg="green")

    rc, _, err = _systemctl_user("daemon-reload")
    if rc != 0:
        click.secho(f"systemctl --user daemon-reload failed: {err}", fg="red")
        return rc or 1
    if not quiet:
        click.secho("[OK] systemd user manager reloaded", fg="green")

    if no_enable:
        click.secho(
            "Skipping enable/start (--no-enable). To activate later:\n"
            f"  systemctl --user enable --now {SERVICE_NAME}",
            fg="cyan",
        )
        return 0

    _enable_linger(quiet)

    enable_args = ("enable",) if no_start else ("enable", "--now")
    rc, _, err = _systemctl_user(*enable_args, SERVICE_NAME)
    if rc != 0:
        click.secho(
            f"systemctl --user {' '.join(enable_args)} {SERVICE_NAME} failed: {err}",
            fg="red",
        )
        return rc or 1

    if no_start:
        click.secho(
            f"[OK] {SERVICE_NAME} enabled for boot (not started yet — run "
            f"`systemctl --user start {SERVICE_NAME}` when you're ready for "
            "the 3-7 minute vLLM cold-load)",
            fg="green",
        )
    else:
        click.secho(
            f"[OK] {SERVICE_NAME} enabled + started — vLLM will autostart on boot",
            fg="green",
        )
        click.echo(
            "  Verify:  systemctl --user status autosre.service\n"
            "  Logs:    journalctl --user -u autosre.service -f"
        )
    return 0


def uninstall(*, quiet: bool = False) -> int:
    """Stop, disable, and remove the user systemd unit."""
    if shutil.which("systemctl") is None:
        if not quiet:
            click.secho("systemctl not found — nothing to uninstall.", fg="yellow")
        return 0

    for action in ("stop", "disable"):
        rc, _, err = _systemctl_user(action, SERVICE_NAME)
        if rc != 0 and "not loaded" not in err.lower() and "no such file" not in err.lower():
            click.secho(
                f"[WARN] systemctl --user {action} {SERVICE_NAME}: {err}",
                fg="yellow",
            )

    unit_path = _user_unit_dir() / SERVICE_NAME
    if unit_path.exists():
        unit_path.unlink()
        if not quiet:
            click.secho(f"[OK] removed {unit_path}", fg="green")

    rc, _, err = _systemctl_user("daemon-reload")
    if rc != 0 and not quiet:
        click.secho(f"daemon-reload after removal: {err}", fg="yellow")

    if not quiet:
        click.secho(f"[OK] {SERVICE_NAME} uninstalled", fg="green")
    return 0
