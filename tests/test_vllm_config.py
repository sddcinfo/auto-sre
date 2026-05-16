"""Tests for autosre.backends.vllm_config module."""

from pathlib import Path

import pytest

from autosre.backends.vllm_config import VllmConfig
from autosre.infra.types import GB10Node, NodeRole


@pytest.fixture
def two_node_config() -> VllmConfig:
    return VllmConfig(
        nodes=[
            GB10Node(hostname="gb10-1", ip="192.168.1.101", role=NodeRole.HEAD),
            GB10Node(hostname="gb10-2", ip="192.168.1.102", role=NodeRole.WORKER),
        ],
    )


@pytest.fixture
def single_node_config() -> VllmConfig:
    return VllmConfig(
        nodes=[
            GB10Node(hostname="gb10-1", ip="192.168.1.101", role=NodeRole.HEAD),
        ],
    )


class TestVllmConfigProperties:
    def test_head_node_explicit(self, two_node_config: VllmConfig) -> None:
        assert two_node_config.head_node.ip == "192.168.1.101"
        assert two_node_config.head_node.role is NodeRole.HEAD

    def test_head_node_fallback_to_first(self) -> None:
        config = VllmConfig(
            nodes=[
                GB10Node(hostname="gb10-1", ip="10.0.0.1"),
                GB10Node(hostname="gb10-2", ip="10.0.0.2"),
            ],
        )
        # No explicit HEAD role, falls back to first node
        assert config.head_node.ip == "10.0.0.1"

    def test_worker_nodes(self, two_node_config: VllmConfig) -> None:
        workers = two_node_config.worker_nodes
        assert len(workers) == 1
        assert workers[0].ip == "192.168.1.102"

    def test_worker_nodes_single(self, single_node_config: VllmConfig) -> None:
        assert single_node_config.worker_nodes == []

    def test_all_ips(self, two_node_config: VllmConfig) -> None:
        assert two_node_config.all_ips == ["192.168.1.101", "192.168.1.102"]

    def test_is_cluster_true(self, two_node_config: VllmConfig) -> None:
        assert two_node_config.is_cluster is True

    def test_is_cluster_false(self, single_node_config: VllmConfig) -> None:
        assert single_node_config.is_cluster is False

    def test_defaults(self) -> None:
        config = VllmConfig(nodes=[GB10Node(hostname="x", ip="1.2.3.4")])
        assert config.docker_image == "bjk110/spark-vllm:turboquant"
        assert config.docker_image_fallback == "eugr/spark-vllm:latest"
        assert config.hf_cache_dir == "/data/huggingface"
        assert config.nccl_socket_ifname == "enp1s0f0np0"


class TestVllmConfigPersistence:
    def test_save_and_load(self, two_node_config: VllmConfig, tmp_path: Path) -> None:
        path = tmp_path / "vllm.yaml"
        two_node_config.save(path)
        loaded = VllmConfig.load(path)

        assert loaded.head_node.ip == "192.168.1.101"
        assert loaded.head_node.role is NodeRole.HEAD
        assert len(loaded.worker_nodes) == 1
        assert loaded.worker_nodes[0].ip == "192.168.1.102"
        assert loaded.docker_image == two_node_config.docker_image

    def test_load_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="vLLM config not found"):
            VllmConfig.load(tmp_path / "nonexistent.yaml")

    def test_load_empty_nodes(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.yaml"
        path.write_text("docker_image: test\n")
        with pytest.raises(ValueError, match="no nodes"):
            VllmConfig.load(path)

    def test_roundtrip_custom_values(self, tmp_path: Path) -> None:
        config = VllmConfig(
            nodes=[GB10Node(hostname="h", ip="10.0.0.1", ssh_user="dgx", role=NodeRole.HEAD)],
            docker_image="custom:latest",
            hf_cache_dir="/custom/cache",
            nccl_socket_ifname="eth0",
        )
        path = tmp_path / "custom.yaml"
        config.save(path)
        loaded = VllmConfig.load(path)

        assert loaded.docker_image == "custom:latest"
        assert loaded.hf_cache_dir == "/custom/cache"
        assert loaded.nccl_socket_ifname == "eth0"
        assert loaded.head_node.ssh_user == "dgx"
