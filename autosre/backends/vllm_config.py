"""Configuration for the vLLM backend on GB10 nodes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path  # noqa: TC003 — used at runtime in classmethods

from autosre.infra.config import DATA_DIR, load_yaml, save_yaml
from autosre.infra.types import GB10Node, NodeRole


@dataclass
class VllmConfig:
    """Configuration for vLLM deployment on GB10 nodes.

    Persisted at ~/.local/share/autosre/vllm.yaml.
    Created by `autosre configure vllm --node <ip> [--node <ip>]`.
    """

    nodes: list[GB10Node]
    docker_image: str = "bjk110/spark-vllm:turboquant"
    docker_image_fallback: str = "eugr/spark-vllm:latest"
    hf_cache_dir: str = "/data/huggingface"
    nccl_socket_ifname: str = "enp1s0f0np0"

    @classmethod
    def default_path(cls) -> Path:
        """Default config file path."""
        return DATA_DIR / "vllm.yaml"

    @classmethod
    def load(cls, path: Path | None = None) -> VllmConfig:
        """Load config from YAML file.

        Args:
            path: Path to config file. None = default path.

        Raises:
            FileNotFoundError: If config file doesn't exist.
            ValueError: If config is invalid (no nodes defined).
        """
        if path is None:
            path = cls.default_path()
        if not path.exists():
            msg = (
                f"vLLM config not found at {path}. Run 'autosre configure vllm --node <ip>' first."
            )
            raise FileNotFoundError(msg)

        data = load_yaml(path)
        if not data.get("nodes"):
            msg = "vLLM config has no nodes defined."
            raise ValueError(msg)

        nodes = [GB10Node.from_dict(n) for n in data["nodes"]]
        return cls(
            nodes=nodes,
            docker_image=str(data.get("docker_image", cls.docker_image)),
            docker_image_fallback=str(data.get("docker_image_fallback", cls.docker_image_fallback)),
            hf_cache_dir=str(data.get("hf_cache_dir", cls.hf_cache_dir)),
            nccl_socket_ifname=str(data.get("nccl_socket_ifname", cls.nccl_socket_ifname)),
        )

    def save(self, path: Path | None = None) -> None:
        """Save config to YAML file."""
        if path is None:
            path = self.default_path()
        data = {
            "nodes": [n.to_dict() for n in self.nodes],
            "docker_image": self.docker_image,
            "docker_image_fallback": self.docker_image_fallback,
            "hf_cache_dir": self.hf_cache_dir,
            "nccl_socket_ifname": self.nccl_socket_ifname,
        }
        save_yaml(path, data)

    @property
    def head_node(self) -> GB10Node:
        """The head node (first node with HEAD role, or first node)."""
        for node in self.nodes:
            if node.role is NodeRole.HEAD:
                return node
        # If no explicit head, first node is head
        return self.nodes[0]

    @property
    def worker_nodes(self) -> list[GB10Node]:
        """All non-head nodes."""
        head = self.head_node
        return [n for n in self.nodes if n is not head]

    @property
    def all_ips(self) -> list[str]:
        """All node IPs in order (head first)."""
        return [self.head_node.ip, *(n.ip for n in self.worker_nodes)]

    @property
    def is_cluster(self) -> bool:
        """True if more than one node is configured."""
        return len(self.nodes) > 1
