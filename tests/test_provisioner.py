"""Tests for autosre.provision.provisioner module."""

from unittest.mock import MagicMock, patch

import pytest

from autosre.infra.types import GB10Node, NodeRole
from autosre.provision.provisioner import DATA_DIRS, DATA_MOUNT, Provisioner


@pytest.fixture
def node() -> GB10Node:
    return GB10Node(hostname="gb10-1", ip="192.168.1.101", role=NodeRole.HEAD)


@pytest.fixture
def provisioner(node: GB10Node) -> Provisioner:
    return Provisioner(node)


class TestProvisionerInit:
    def test_stores_node(self, provisioner: Provisioner) -> None:
        assert provisioner.node.ip == "192.168.1.101"

    def test_stores_hf_token(self, node: GB10Node) -> None:
        p = Provisioner(node, hf_token="hf_test123")
        assert p.hf_token == "hf_test123"


class TestVerifyConnectivity:
    @patch("autosre.provision.provisioner.SSHRunner")
    def test_all_ok(self, mock_ssh_cls: MagicMock, node: GB10Node) -> None:
        mock_runner = MagicMock()
        mock_runner.is_reachable.return_value = True
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "NVIDIA GB10 GPU, 128 GB"
        mock_runner.run.return_value = mock_result
        mock_ssh_cls.return_value = mock_runner

        p = Provisioner(node)
        p.ssh = mock_runner
        assert p.verify_connectivity() is True

    @patch("autosre.provision.provisioner.SSHRunner")
    def test_ssh_unreachable(self, mock_ssh_cls: MagicMock, node: GB10Node) -> None:
        mock_runner = MagicMock()
        mock_runner.is_reachable.return_value = False
        mock_ssh_cls.return_value = mock_runner

        p = Provisioner(node)
        p.ssh = mock_runner
        assert p.verify_connectivity() is False


class TestSetupDataPartition:
    def _mock_runner_with(self, root_source: str) -> MagicMock:
        mock_runner = MagicMock()
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = root_source + "\n"
        mock_runner.run.return_value = mock_result
        return mock_runner

    @patch("autosre.provision.provisioner.SSHRunner")
    def test_already_mounted(self, mock_ssh_cls: MagicMock, node: GB10Node) -> None:
        # Separate-partition layout: root is a different device than
        # DATA_PARTITION, and mountpoint -q returns 0 (already mounted).
        mock_runner = self._mock_runner_with("/dev/nvme1n1p1")
        mock_ssh_cls.return_value = mock_runner

        p = Provisioner(node)
        p.ssh = mock_runner
        assert p.setup_data_partition() is True

        calls = [str(c) for c in mock_runner.run.call_args_list]
        assert any("mkdir" in c for c in calls)

    @patch("autosre.provision.provisioner.SSHRunner")
    def test_data_partition_is_root(
        self,
        mock_ssh_cls: MagicMock,
        node: GB10Node,
    ) -> None:
        """Single-partition installs (GB10 DGX OS default): root == DATA_PARTITION.

        Provisioner must NOT try to mount root over /data — just create
        the directory on the root filesystem.
        """
        from autosre.provision.provisioner import DATA_PARTITION

        mock_runner = self._mock_runner_with(DATA_PARTITION)
        mock_ssh_cls.return_value = mock_runner

        p = Provisioner(node)
        p.ssh = mock_runner
        assert p.setup_data_partition() is True

        # No mount of DATA_PARTITION as a separate device should happen.
        commands = [c.args[0] if c.args else [] for c in mock_runner.run.call_args_list]
        for cmd in commands:
            joined = " ".join(cmd) if isinstance(cmd, list) else str(cmd)
            assert not (joined.startswith("mount ") and DATA_PARTITION in joined), (
                f"single-partition layout should not issue a mount: {joined!r}"
            )


class TestConfigureDocker:
    @patch("autosre.provision.provisioner.SSHRunner")
    def test_already_configured(self, mock_ssh_cls: MagicMock, node: GB10Node) -> None:
        mock_runner = MagicMock()
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = '{"default-runtime": "nvidia", "runtimes": {"nvidia": {}}}'
        mock_runner.run.return_value = mock_result
        mock_ssh_cls.return_value = mock_runner

        p = Provisioner(node)
        p.ssh = mock_runner
        assert p.configure_docker() is True


class TestValidate:
    @patch("autosre.provision.provisioner.SSHRunner")
    def test_all_pass(self, mock_ssh_cls: MagicMock, node: GB10Node) -> None:
        mock_runner = MagicMock()
        mock_runner.is_reachable.return_value = True
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "nvidia\nGPU OK"
        mock_runner.run.return_value = mock_result
        mock_ssh_cls.return_value = mock_runner

        p = Provisioner(node)
        p.ssh = mock_runner
        ok, issues = p.validate()
        assert ok is True
        assert issues == []

    @patch("autosre.provision.provisioner.SSHRunner")
    def test_ssh_unreachable(self, mock_ssh_cls: MagicMock, node: GB10Node) -> None:
        mock_runner = MagicMock()
        mock_runner.is_reachable.return_value = False
        mock_ssh_cls.return_value = mock_runner

        p = Provisioner(node)
        p.ssh = mock_runner
        ok, issues = p.validate()
        assert ok is False
        assert any("SSH" in i for i in issues)


class TestPreWipeBackup:
    @patch("autosre.provision.provisioner.SSHRunner")
    def test_backup_runs_commands(self, mock_ssh_cls: MagicMock, node: GB10Node) -> None:
        mock_runner = MagicMock()
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        mock_runner.run.return_value = mock_result
        mock_ssh_cls.return_value = mock_runner

        p = Provisioner(node)
        p.ssh = mock_runner
        assert p.pre_wipe_backup() is True

        # Should have run mkdir, cp, docker images
        assert mock_runner.run.call_count >= 3


class TestProvisionPipeline:
    """``Provisioner.provision()`` is the public entry point — covers
    only the generic prep steps (no operator-specific dotfiles /
    repo / hooks). The ``OperatorProvisioner`` subclass in private
    sddc-cli adds those parity steps; their unit tests live there."""

    @patch("autosre.provision.provisioner.SSHRunner")
    def test_provision_pipeline_runs_only_generic_steps(
        self,
        mock_ssh_cls: MagicMock,
        node: GB10Node,
    ) -> None:
        mock_runner = MagicMock()
        mock_ssh_cls.return_value = mock_runner

        p = Provisioner(node)
        p.ssh = mock_runner
        # Stub every step to True so the pipeline runs end-to-end.
        for name in (
            "verify_connectivity",
            "setup_ssh_keys",
            "configure_network",
            "setup_data_partition",
            "configure_docker",
            "setup_hf_cache",
            "set_performance_mode",
            "configure_firewall",
        ):
            setattr(p, name, MagicMock(return_value=True))

        assert p.provision() is True

        for name in (
            "verify_connectivity",
            "setup_ssh_keys",
            "configure_network",
            "setup_data_partition",
            "configure_docker",
            "setup_hf_cache",
            "set_performance_mode",
            "configure_firewall",
        ):
            getattr(p, name).assert_called_once()

        # Operator-specific methods must NOT exist on the public class.
        for name in (
            "sync_tmux_config",
            "_local_sddcinfo_root",
            "_rsync_tree_to_target",
            "clone_sddcinfo_repo",
            "push_age_key",
            "run_bootstrap",
            "install_claude_hooks",
        ):
            assert not hasattr(p, name), (
                f"{name!r} leaked into the public Provisioner — it must "
                f"only live on OperatorProvisioner in private sddc-cli."
            )


class TestConstants:
    def test_data_mount(self) -> None:
        assert DATA_MOUNT == "/data"

    def test_data_dirs(self) -> None:
        assert "huggingface" in DATA_DIRS
        assert "docker-images" in DATA_DIRS
        assert "ssh-keys" in DATA_DIRS
        assert "configs" in DATA_DIRS
        assert "backups" in DATA_DIRS
