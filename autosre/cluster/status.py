"""Cluster status dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class NodeStatus:
    """Status of a single k3s node."""

    hostname: str
    ip: str
    ready: bool
    roles: list[str] = field(default_factory=list)
    k3s_version: str | None = None
    gpu_detected: bool = False


@dataclass
class ClusterStatus:
    """Overall k3s cluster status."""

    cluster_ready: bool
    k3s_server_running: bool
    nodes: list[NodeStatus] = field(default_factory=list)
    gpu_operator_ready: bool = False
    network_operator_ready: bool = False
    nccl_healthy: bool = False
    total_gpu_memory_gb: int = 0
    error: str | None = None
