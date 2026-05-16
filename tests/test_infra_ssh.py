"""Tests for autosre.infra.ssh module."""

from unittest.mock import patch

import pytest

from autosre.infra.ssh import SSHRunner
from autosre.infra.types import GB10Node, NodeRole


@pytest.fixture
def head_node() -> GB10Node:
    return GB10Node(hostname="gb10-1", ip="192.168.1.101", role=NodeRole.HEAD)


@pytest.fixture
def node_with_key() -> GB10Node:
    return GB10Node(
        hostname="gb10-2",
        ip="192.168.1.102",
        ssh_key="/home/user/.ssh/custom_key",
    )


@pytest.fixture
def runner(head_node: GB10Node) -> SSHRunner:
    return SSHRunner(head_node)


@pytest.fixture
def runner_with_key(node_with_key: GB10Node) -> SSHRunner:
    return SSHRunner(node_with_key)


class TestSSHCommandBuilding:
    def test_basic_command(self, runner: SSHRunner) -> None:
        cmd = runner._build_ssh_cmd(["echo", "hello"])
        assert cmd[0] == "ssh"
        assert "-o" in cmd
        assert "StrictHostKeyChecking=accept-new" in cmd
        assert "BatchMode=yes" in cmd
        assert "root@192.168.1.101" in cmd
        # Command is shlex-joined into a single remote-side string so argv
        # boundaries survive SSH's re-parse on the target. For a token list
        # without shell metacharacters shlex.join preserves the plain form.
        assert cmd[-1] == "echo hello"

    def test_command_with_ssh_key(self, runner_with_key: SSHRunner) -> None:
        cmd = runner_with_key._build_ssh_cmd(["nvidia-smi"])
        assert "-i" in cmd
        key_idx = cmd.index("-i")
        assert cmd[key_idx + 1] == "/home/user/.ssh/custom_key"

    def test_command_without_ssh_key(self, runner: SSHRunner) -> None:
        cmd = runner._build_ssh_cmd(["nvidia-smi"])
        assert "-i" not in cmd

    def test_command_with_shell_redirect_is_quoted(self, runner: SSHRunner) -> None:
        """Shell metacharacters must not leak into SSH's outer shell — they
        need to survive into the remote ``bash -c`` body intact."""
        cmd = runner._build_ssh_cmd(
            ["bash", "-c", "echo '{\"a\":1}' > /etc/foo.json"],
        )
        remote = cmd[-1]
        # The redirect operator stays inside the quoted bash argument.
        assert remote.startswith("bash -c ")
        assert "/etc/foo.json" in remote
        # The JSON content is preserved (quoted, not re-parsed).
        assert "'{\"a\":1}'" in remote or '{"a":1}' in remote


class TestSSHRun:
    @patch("autosre.infra.ssh.subprocess.run")
    def test_run_success(self, mock_run, runner: SSHRunner) -> None:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "OK\n"
        mock_run.return_value.stderr = ""

        runner.run(["echo", "OK"])

        assert mock_run.called
        call_args = mock_run.call_args
        assert call_args.kwargs["capture_output"] is True
        assert call_args.kwargs["text"] is True
        assert call_args.kwargs["timeout"] == 30
        assert call_args.kwargs["check"] is True

    @patch("autosre.infra.ssh.subprocess.run")
    def test_run_custom_timeout(self, mock_run, runner: SSHRunner) -> None:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = ""
        runner.run(["ls"], timeout=60)
        assert mock_run.call_args.kwargs["timeout"] == 60

    @patch("autosre.infra.ssh.subprocess.run")
    def test_run_no_check(self, mock_run, runner: SSHRunner) -> None:
        mock_run.return_value.returncode = 1
        mock_run.return_value.stdout = ""
        runner.run(["false"], check=False)
        assert mock_run.call_args.kwargs["check"] is False


class TestSSHReachable:
    @patch("autosre.infra.ssh.subprocess.run")
    def test_reachable(self, mock_run, runner: SSHRunner) -> None:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "ok\n"
        assert runner.is_reachable() is True

    @patch("autosre.infra.ssh.subprocess.run")
    def test_unreachable_exit_code(self, mock_run, runner: SSHRunner) -> None:
        mock_run.return_value.returncode = 255
        mock_run.return_value.stdout = ""
        assert runner.is_reachable() is False

    @patch("autosre.infra.ssh.subprocess.run")
    def test_unreachable_timeout(self, mock_run, runner: SSHRunner) -> None:
        import subprocess

        mock_run.side_effect = subprocess.TimeoutExpired(cmd="ssh", timeout=5)
        assert runner.is_reachable() is False

    @patch("autosre.infra.ssh.subprocess.run")
    def test_unreachable_no_ssh(self, mock_run, runner: SSHRunner) -> None:
        mock_run.side_effect = FileNotFoundError("ssh not found")
        assert runner.is_reachable() is False


class TestDockerRun:
    @patch("autosre.infra.ssh.subprocess.run")
    def test_docker_run_basic(self, mock_run, runner: SSHRunner) -> None:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "abc123\n"

        container_id = runner.docker_run("vllm:latest")

        assert container_id == "abc123"
        # argv is shlex-joined for SSH transport, so inspect the
        # combined remote-side command string.
        remote = mock_run.call_args[0][0][-1]
        # Default behavior: containers persist after exit so a crashed
        # vLLM cold-load is recoverable via ``docker logs``. ``--rm``
        # is opt-in via ``remove=True``.
        assert "--rm" not in remote, f"unexpected --rm in {remote!r}"
        for token in (
            "docker",
            "run",
            "-d",
            "--gpus",
            "all",
            "--network",
            "host",
            "--shm-size",
            "16g",
            "vllm:latest",
        ):
            assert token in remote, f"missing {token!r} in {remote!r}"

    @patch("autosre.infra.ssh.subprocess.run")
    def test_docker_run_remove_flag_opt_in(self, mock_run, runner: SSHRunner) -> None:
        """Callers that genuinely want auto-cleanup pass ``remove=True``."""
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "abc123\n"

        runner.docker_run("vllm:latest", remove=True)

        remote = mock_run.call_args[0][0][-1]
        assert "--rm" in remote, f"missing --rm in {remote!r}"

    @patch("autosre.infra.ssh.subprocess.run")
    def test_docker_run_with_name_clears_old_container(self, mock_run, runner: SSHRunner) -> None:
        """When a name slot is requested, any existing container with
        that name is force-removed first so ``docker run`` doesn't
        bail with "container name already in use".
        """
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "abc123\n"

        runner.docker_run("vllm:latest", name="autosre-vllm")

        # Two subprocess.run calls: the rm -f sweep, then the run.
        assert mock_run.call_count == 2
        sweep_remote = mock_run.call_args_list[0][0][0][-1]
        assert "docker rm -f autosre-vllm" in sweep_remote
        run_remote = mock_run.call_args_list[1][0][0][-1]
        assert "--name autosre-vllm" in run_remote

    @patch("autosre.infra.ssh.subprocess.run")
    def test_docker_run_with_name_and_env(self, mock_run, runner: SSHRunner) -> None:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "xyz789\n"

        runner.docker_run(
            "vllm:latest",
            name="autosre-vllm",
            env={"VLLM_HOST_IP": "192.168.1.101"},
            volumes=["/data/hf:/root/.cache/huggingface"],
        )

        # The final call is the run; the prior call is the rm sweep.
        remote = mock_run.call_args_list[-1][0][0][-1]
        assert "--name autosre-vllm" in remote
        assert "VLLM_HOST_IP=192.168.1.101" in remote
        assert "/data/hf:/root/.cache/huggingface" in remote


class TestDockerStop:
    @patch("autosre.infra.ssh.subprocess.run")
    def test_docker_stop_success(self, mock_run, runner: SSHRunner) -> None:
        mock_run.return_value.returncode = 0
        assert runner.docker_stop("abc123") is True

    @patch("autosre.infra.ssh.subprocess.run")
    def test_docker_stop_failure(self, mock_run, runner: SSHRunner) -> None:
        mock_run.return_value.returncode = 1
        assert runner.docker_stop("nonexistent") is False


class TestDockerPs:
    @patch("autosre.infra.ssh.subprocess.run")
    def test_docker_ps_with_filter(self, mock_run, runner: SSHRunner) -> None:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "abc123\tautosre-vllm\tUp 5 minutes"

        output = runner.docker_ps(name_filter="autosre")

        remote = mock_run.call_args[0][0][-1]
        assert "--filter" in remote
        assert "name=autosre" in remote
        assert "abc123" in output


class TestRsync:
    @patch("autosre.infra.ssh.subprocess.run")
    def test_rsync_success(self, mock_run, runner: SSHRunner) -> None:
        mock_run.return_value.returncode = 0
        result = runner.rsync("/data/models/", "root@192.168.1.102:/data/models/")
        assert result is True

    @patch("autosre.infra.ssh.subprocess.run")
    def test_rsync_with_exclude(self, mock_run, runner: SSHRunner) -> None:
        mock_run.return_value.returncode = 0
        runner.rsync(
            "/data/",
            "root@192.168.1.102:/data/",
            exclude=["*.tmp", "cache/"],
        )
        cmd = mock_run.call_args[0][0]
        assert "--exclude" in cmd
        assert "*.tmp" in cmd
        assert "cache/" in cmd

    @patch("autosre.infra.ssh.subprocess.run")
    def test_rsync_failure(self, mock_run, runner: SSHRunner) -> None:
        mock_run.return_value.returncode = 1
        result = runner.rsync("/src/", "/dest/")
        assert result is False
