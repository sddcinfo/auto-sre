"""Shared types for GB10 node management."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class NodeRole(Enum):
    """Role of a GB10 node in the cluster."""

    HEAD = "head"
    WORKER = "worker"


@dataclass
class GB10Node:
    """A Dell Pro Max GB10 (NVIDIA Grace Blackwell) node.

    Attributes:
        hostname: Node hostname (e.g., "gb10-1")
        ip: Node IP address (e.g., "192.168.1.101")
        ssh_user: SSH username for remote operations
        ssh_key: Path to SSH private key (None = default ~/.ssh/id_ed25519)
        role: Node role in the cluster (head or worker)
    """

    hostname: str
    ip: str
    ssh_user: str = "root"
    ssh_key: str | None = None
    role: NodeRole = NodeRole.WORKER

    def to_dict(self) -> dict[str, str]:
        """Serialize to a dict suitable for YAML."""
        d: dict[str, str] = {
            "hostname": self.hostname,
            "ip": self.ip,
            "ssh_user": self.ssh_user,
            "role": self.role.value,
        }
        if self.ssh_key:
            d["ssh_key"] = self.ssh_key
        return d

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> GB10Node:
        """Deserialize from a dict (e.g., from YAML)."""
        return cls(
            hostname=str(data["hostname"]),
            ip=str(data["ip"]),
            ssh_user=str(data.get("ssh_user", "root")),
            ssh_key=str(data["ssh_key"]) if data.get("ssh_key") else None,
            role=NodeRole(str(data.get("role", "worker"))),
        )

    @property
    def ssh_target(self) -> str:
        """SSH connection string (user@ip)."""
        return f"{self.ssh_user}@{self.ip}"


# Solo models that can run on a single GB10 node (used for degraded mode)
SOLO_FALLBACK_MODEL = "nemotron-nano"

# Models requiring TP=2 (both nodes)
CLUSTER_MODELS: set[str] = {
    "nemotron-super",
    "qwen3.6-122b",
    "qwen3.6-397b",
}
