"""Tests for autosre.cluster module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from autosre.backends.vllm_config import VllmConfig
from autosre.cluster.manager import K3S_VERSION, ClusterManager
from autosre.cluster.status import ClusterStatus, NodeStatus
from autosre.infra.types import GB10Node, NodeRole


@pytest.fixture
def config(tmp_path: Path) -> VllmConfig:
    c = VllmConfig(
        nodes=[
            GB10Node(hostname="gb10-1", ip="192.168.1.101", role=NodeRole.HEAD),
            GB10Node(hostname="gb10-2", ip="192.168.1.102", role=NodeRole.WORKER),
        ],
    )
    c.save(tmp_path / "vllm.yaml")
    return c


@pytest.fixture
def manager(config: VllmConfig) -> ClusterManager:
    return ClusterManager(config)


class TestClusterStatus:
    def test_node_status_defaults(self) -> None:
        ns = NodeStatus(hostname="gb10-1", ip="192.168.1.101", ready=True)
        assert ns.ready is True
        assert ns.roles == []
        assert ns.k3s_version is None
        assert ns.gpu_detected is False

    def test_cluster_status_defaults(self) -> None:
        cs = ClusterStatus(cluster_ready=False, k3s_server_running=False)
        assert cs.nodes == []
        assert cs.gpu_operator_ready is False
        assert cs.error is None


class TestClusterManagerInit:
    def test_stores_config(self, manager: ClusterManager) -> None:
        assert manager.config.head_node.ip == "192.168.1.101"


class TestInstallK3sServer:
    @patch("autosre.cluster.manager.SSHRunner")
    def test_install_success(self, mock_ssh_cls: MagicMock, config: VllmConfig) -> None:
        mock_runner = MagicMock()
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_runner.run.return_value = mock_result
        mock_ssh_cls.return_value = mock_runner

        mgr = ClusterManager(config)
        mgr._head_ssh = mock_runner
        assert mgr.install_k3s_server() is True

        # Verify k3s install command was run
        call_args = mock_runner.run.call_args[0][0]
        assert "get.k3s.io" in " ".join(call_args)
        assert "--docker" in " ".join(call_args)

    @patch("autosre.cluster.manager.SSHRunner")
    def test_install_failure(self, mock_ssh_cls: MagicMock, config: VllmConfig) -> None:
        mock_runner = MagicMock()
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "install error"
        mock_runner.run.return_value = mock_result
        mock_ssh_cls.return_value = mock_runner

        mgr = ClusterManager(config)
        mgr._head_ssh = mock_runner
        assert mgr.install_k3s_server() is False


class TestGetJoinToken:
    @patch("autosre.cluster.manager.SSHRunner")
    def test_get_token(self, mock_ssh_cls: MagicMock, config: VllmConfig) -> None:
        mock_runner = MagicMock()
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "K10abc123token456::server:xyz"
        mock_runner.run.return_value = mock_result
        mock_ssh_cls.return_value = mock_runner

        mgr = ClusterManager(config)
        mgr._head_ssh = mock_runner
        token = mgr.get_join_token()
        assert token == "K10abc123token456::server:xyz"


class TestTeardown:
    @patch("autosre.cluster.manager.SSHRunner")
    def test_teardown(self, mock_ssh_cls: MagicMock, config: VllmConfig) -> None:
        mock_runner = MagicMock()
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_runner.run.return_value = mock_result
        mock_ssh_cls.return_value = mock_runner

        mgr = ClusterManager(config)
        mgr._head_ssh = mock_runner
        assert mgr.teardown() is True

        # Should run uninstall on both nodes
        assert mock_runner.run.call_count >= 2


class TestValidateNccl:
    @patch("autosre.cluster.manager.SSHRunner")
    def test_nccl_success(self, mock_ssh_cls: MagicMock, config: VllmConfig) -> None:
        mock_runner = MagicMock()
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "# Out of bounds values: ...\nAvg bus bandwidth: 12.5 GB/s"
        mock_runner.run.return_value = mock_result
        mock_ssh_cls.return_value = mock_runner

        mgr = ClusterManager(config)
        mgr._head_ssh = mock_runner
        assert mgr.validate_nccl() is True

    @patch("autosre.cluster.manager.SSHRunner")
    def test_nccl_failure(self, mock_ssh_cls: MagicMock, config: VllmConfig) -> None:
        mock_runner = MagicMock()
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "nccl error"
        mock_runner.run.return_value = mock_result
        mock_ssh_cls.return_value = mock_runner

        mgr = ClusterManager(config)
        mgr._head_ssh = mock_runner
        assert mgr.validate_nccl() is False


class TestValidateRdma:
    @patch("autosre.cluster.manager.SSHRunner")
    def test_rdma_available(self, mock_ssh_cls: MagicMock, config: VllmConfig) -> None:
        mock_runner = MagicMock()
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "link mlx5_0/1 state ACTIVE\nlink mlx5_1/1 state ACTIVE"
        mock_runner.run.return_value = mock_result
        mock_ssh_cls.return_value = mock_runner

        mgr = ClusterManager(config)
        mgr._head_ssh = mock_runner
        assert mgr.validate_rdma() is True


class TestConstants:
    def test_k3s_version(self) -> None:
        assert K3S_VERSION.startswith("v1.")
