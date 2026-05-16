"""Tests for autosre.provision.lifecycle module."""

from unittest.mock import MagicMock, patch

import pytest

from autosre.infra.types import GB10Node, NodeRole
from autosre.provision.lifecycle import NodeLifecycle


@pytest.fixture
def two_nodes() -> list[GB10Node]:
    return [
        GB10Node(hostname="gb10-1", ip="192.168.1.101", role=NodeRole.HEAD),
        GB10Node(hostname="gb10-2", ip="192.168.1.102", role=NodeRole.WORKER),
    ]


@pytest.fixture
def lifecycle(two_nodes: list[GB10Node]) -> NodeLifecycle:
    return NodeLifecycle(two_nodes)


class TestNodeLifecycleProperties:
    def test_head_node(self, lifecycle: NodeLifecycle) -> None:
        assert lifecycle.head_node.ip == "192.168.1.101"
        assert lifecycle.head_node.role is NodeRole.HEAD

    def test_worker_nodes(self, lifecycle: NodeLifecycle) -> None:
        workers = lifecycle.worker_nodes
        assert len(workers) == 1
        assert workers[0].ip == "192.168.1.102"


class TestSyncModels:
    @patch("autosre.provision.lifecycle.SSHRunner")
    def test_sync_models_success(self, mock_ssh_cls: MagicMock, lifecycle: NodeLifecycle) -> None:
        mock_runner = MagicMock()
        mock_runner.rsync.return_value = True
        mock_ssh_cls.return_value = mock_runner

        source = lifecycle.nodes[0]
        dest = lifecycle.nodes[1]
        result = lifecycle.sync_models(source, dest)

        assert result is True
        mock_runner.rsync.assert_called_once()
        call_args = mock_runner.rsync.call_args
        assert "/data/huggingface/" in call_args[0][0]

    @patch("autosre.provision.lifecycle.SSHRunner")
    def test_sync_models_failure(self, mock_ssh_cls: MagicMock, lifecycle: NodeLifecycle) -> None:
        mock_runner = MagicMock()
        mock_runner.rsync.return_value = False
        mock_ssh_cls.return_value = mock_runner

        result = lifecycle.sync_models(lifecycle.nodes[0], lifecycle.nodes[1])
        assert result is False


class TestSyncDockerImages:
    @patch("autosre.provision.lifecycle.SSHRunner")
    def test_sync_images(self, mock_ssh_cls: MagicMock, lifecycle: NodeLifecycle) -> None:
        mock_runner = MagicMock()
        mock_runner.rsync.return_value = True
        mock_ssh_cls.return_value = mock_runner

        result = lifecycle.sync_docker_images(lifecycle.nodes[0], lifecycle.nodes[1])
        assert result is True


class TestFullWipeBoth:
    @patch("autosre.provision.lifecycle.Provisioner")
    @patch("autosre.provision.lifecycle.click.confirm", return_value=True)
    def test_full_wipe_confirms_and_backs_up(
        self, mock_confirm: MagicMock, mock_prov_cls: MagicMock, lifecycle: NodeLifecycle
    ) -> None:
        mock_prov = MagicMock()
        mock_prov.pre_wipe_backup.return_value = True
        mock_prov.save_docker_images.return_value = True
        mock_prov_cls.return_value = mock_prov

        result = lifecycle.full_wipe_both()

        assert result is True
        assert mock_prov.pre_wipe_backup.call_count == 2  # both nodes
        assert mock_prov.save_docker_images.call_count == 2

    @patch("autosre.provision.lifecycle.click.confirm", return_value=False)
    def test_full_wipe_aborted(self, mock_confirm: MagicMock, lifecycle: NodeLifecycle) -> None:
        result = lifecycle.full_wipe_both()
        assert result is False
