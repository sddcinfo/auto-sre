"""SSH transport via subprocess. No paramiko dependency."""

from __future__ import annotations

import shlex
import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autosre.infra.types import GB10Node


class SSHRunner:
    """Execute commands on a remote GB10 node via SSH.

    Uses subprocess.run(["ssh", ...]) for zero compiled dependencies.
    Assumes key-based auth is configured (password auth not supported).
    """

    def __init__(self, node: GB10Node) -> None:
        self.node = node

    def _build_ssh_cmd(self, cmd: list[str]) -> list[str]:
        """Build the full SSH command with options.

        SSH concatenates every argv token after ``user@host`` with a
        single space and feeds the result to the remote ``$SHELL -c``,
        which re-parses redirects / quotes / globs. To preserve argv
        boundaries we shell-quote each element with :func:`shlex.quote`
        before joining — so ``["bash", "-c", "echo x > file"]`` arrives
        as ``bash -c 'echo x > file'`` on the remote instead of being
        mangled by the outer shell.
        """
        ssh_cmd = [
            "ssh",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            "ConnectTimeout=5",
            "-o",
            "BatchMode=yes",
        ]
        if self.node.ssh_key:
            ssh_cmd.extend(["-i", self.node.ssh_key])
        ssh_cmd.append(self.node.ssh_target)
        # shlex.join preserves argv boundaries through SSH's re-parse on the
        # remote side. Without this, embedded spaces, quotes, and
        # redirects leak into the outer shell.
        ssh_cmd.append(shlex.join(cmd))
        return ssh_cmd

    def run(
        self,
        cmd: list[str],
        timeout: int = 30,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        """Run a command on the remote node.

        Args:
            cmd: Command and arguments to execute remotely.
            timeout: Timeout in seconds.
            check: If True, raise CalledProcessError on non-zero exit.

        Returns:
            CompletedProcess with stdout/stderr captured as text.
        """
        ssh_cmd = self._build_ssh_cmd(cmd)
        return subprocess.run(
            ssh_cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=check,
        )

    def run_bg(self, cmd: list[str]) -> str:
        """Run a command in the background on the remote node.

        Returns:
            The remote PID as a string.
        """
        # Wrap command to run in background and echo PID
        bg_cmd = ["nohup", *cmd, ">/dev/null", "2>&1", "&", "echo", "$!"]
        result = self.run(bg_cmd, timeout=10, check=False)
        return result.stdout.strip()

    def is_reachable(self, timeout: int = 5) -> bool:
        """Check if the node is reachable via SSH.

        Returns:
            True if SSH connection succeeds.
        """
        try:
            result = self.run(["echo", "ok"], timeout=timeout, check=False)
            return result.returncode == 0 and "ok" in result.stdout
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            return False

    def rsync(
        self,
        src: str,
        dest: str,
        exclude: list[str] | None = None,
        timeout: int = 3600,
    ) -> bool:
        """Rsync files to/from the remote node.

        Args:
            src: Source path (local or remote with node prefix).
            dest: Destination path (local or remote with node prefix).
            exclude: Patterns to exclude.
            timeout: Timeout in seconds (default 1 hour for large model transfers).

        Returns:
            True if rsync succeeded.
        """
        rsync_cmd = [
            "rsync",
            "-az",
            "--progress",
            "-e",
            self._build_ssh_arg(),
        ]
        if exclude:
            for pattern in exclude:
                rsync_cmd.extend(["--exclude", pattern])
        rsync_cmd.extend([src, dest])

        result = subprocess.run(
            rsync_cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return result.returncode == 0

    def _build_ssh_arg(self) -> str:
        """Build the -e argument for rsync."""
        parts = ["ssh", "-o", "StrictHostKeyChecking=accept-new", "-o", "BatchMode=yes"]
        if self.node.ssh_key:
            parts.extend(["-i", self.node.ssh_key])
        return " ".join(parts)

    def docker_run(
        self,
        image: str,
        cmd: list[str] | None = None,
        *,
        name: str | None = None,
        detach: bool = True,
        remove: bool = False,
        gpus: str = "all",
        network: str = "host",
        volumes: list[str] | None = None,
        env: dict[str, str] | None = None,
        shm_size: str = "16g",
        extra_args: list[str] | None = None,
    ) -> str:
        """Run a Docker container on the remote node.

        ``remove`` defaults to ``False`` so crashed containers stick
        around and ``docker logs`` still works after a failed startup —
        critical for diagnosing slow vLLM cold-load failures (the wait
        loop in :meth:`VllmBackend._wait_for_vllm` can return False
        before the inner process has a chance to log a clean error,
        and ``--rm`` would erase the evidence). The launch path is
        responsible for ``docker rm -f`` of any previous corpse before
        starting fresh — see :meth:`VllmBackend._start_local` for the
        local-mode reference implementation.

        Returns:
            Container ID if detached, or stdout if not.
        """
        # If a named slot is requested and an old container is sitting
        # there (stopped or running, possibly the corpse of a previous
        # crashed attempt now that ``remove=False`` is the default),
        # clear it first. ``docker run --name X`` would otherwise fail
        # with "container name already in use". `check=False` because
        # "no such container" is the expected outcome on a clean host.
        if name:
            self.run(["docker", "rm", "-f", name], timeout=10, check=False)

        docker_cmd = ["docker", "run"]
        if detach:
            docker_cmd.append("-d")
        if remove:
            docker_cmd.append("--rm")
        if name:
            docker_cmd.extend(["--name", name])
        if gpus:
            docker_cmd.extend(["--gpus", gpus])
        if network:
            docker_cmd.extend(["--network", network])
        if shm_size:
            docker_cmd.extend(["--shm-size", shm_size])
        docker_cmd.extend(["--ulimit", "memlock=-1"])

        if volumes:
            for vol in volumes:
                docker_cmd.extend(["-v", vol])
        if env:
            for key, value in env.items():
                docker_cmd.extend(["-e", f"{key}={value}"])
        if extra_args:
            docker_cmd.extend(extra_args)

        docker_cmd.append(image)
        if cmd:
            docker_cmd.extend(cmd)

        result = self.run(docker_cmd, timeout=60, check=True)
        return result.stdout.strip()

    def docker_stop(self, container_id: str, timeout: int = 30) -> bool:
        """Stop a Docker container on the remote node."""
        result = self.run(
            ["docker", "stop", "-t", str(timeout), container_id],
            timeout=timeout + 10,
            check=False,
        )
        return result.returncode == 0

    def docker_ps(self, name_filter: str | None = None) -> str:
        """List running Docker containers on the remote node."""
        cmd = ["docker", "ps", "--format", "{{.ID}}\t{{.Names}}\t{{.Status}}"]
        if name_filter:
            cmd.extend(["--filter", f"name={name_filter}"])
        result = self.run(cmd, timeout=10, check=False)
        return result.stdout.strip()
