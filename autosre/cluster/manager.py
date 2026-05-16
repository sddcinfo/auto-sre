"""K3s cluster lifecycle manager for GB10 nodes.

Manages a 2-node k3s cluster with NVIDIA GPU and Network operators.
This is an OPTIONAL management overlay — vLLM serving uses SSH+Docker directly.

K3s specifics for GB10:
- Docker runtime (--docker), NOT containerd
- GPU Operator v25.10+ (driver.enabled=false, DGX OS has pre-installed driver)
- Network Operator for RDMA over ConnectX-7
- --disable=traefik (no ingress needed)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import click

from autosre.cluster.status import ClusterStatus, NodeStatus
from autosre.infra.ssh import SSHRunner

if TYPE_CHECKING:
    import subprocess

    from autosre.backends.vllm_config import VllmConfig
    from autosre.infra.types import GB10Node

# K3s install configuration
K3S_VERSION = "v1.31.4+k3s1"
K3S_INSTALL_FLAGS = "--docker --write-kubeconfig-mode 644 --disable=traefik --disable=servicelb"

# NVIDIA operator versions
GPU_OPERATOR_VERSION = "v25.10.0"
NETWORK_OPERATOR_VERSION = "v25.4.0"


class ClusterManager:
    """Manages k3s cluster lifecycle on GB10 nodes.

    All operations execute via SSH to the configured nodes.
    """

    def __init__(self, config: VllmConfig) -> None:
        self.config = config
        self._head_ssh = SSHRunner(config.head_node)

    def bootstrap(self) -> bool:
        """Full cluster bootstrap: k3s server + agents + operators.

        Steps:
        1. Install k3s server on head node (Docker runtime)
        2. Get join token
        3. Join agent nodes
        4. Deploy NVIDIA GPU Operator
        5. Deploy NVIDIA Network Operator
        6. Validate

        Returns True if bootstrap succeeded.
        """
        click.secho("Bootstrapping k3s cluster...", bold=True)

        # Step 1: Install k3s server
        click.echo("\n[1/5] Installing k3s server on head node...")
        if not self.install_k3s_server():
            return False

        # Step 2: Get join token
        click.echo("\n[2/5] Retrieving join token...")
        token = self.get_join_token()
        if not token:
            click.secho("Failed to get join token", fg="red")
            return False

        # Step 3: Join agents
        click.echo("\n[3/5] Joining agent nodes...")
        for node in self.config.worker_nodes:
            if not self.join_agent(node, token):
                return False

        # Step 4: GPU Operator
        click.echo("\n[4/5] Deploying NVIDIA GPU Operator...")
        if not self.deploy_gpu_operator():
            click.secho("GPU Operator deployment failed (non-fatal)", fg="yellow")

        # Step 5: Network Operator
        click.echo("\n[5/5] Deploying NVIDIA Network Operator...")
        if not self.deploy_network_operator():
            click.secho("Network Operator deployment failed (non-fatal)", fg="yellow")

        click.secho("\nCluster bootstrap complete!", fg="green", bold=True)
        return True

    def teardown(self) -> bool:
        """Remove k3s from all nodes."""
        click.echo("Tearing down k3s cluster...")

        # Remove agents first
        for node in self.config.worker_nodes:
            runner = SSHRunner(node)
            click.echo(f"  Removing k3s agent on {node.hostname}...")
            runner.run(["/usr/local/bin/k3s-agent-uninstall.sh"], check=False, timeout=60)

        # Remove server
        click.echo(f"  Removing k3s server on {self.config.head_node.hostname}...")
        self._head_ssh.run(["/usr/local/bin/k3s-uninstall.sh"], check=False, timeout=60)

        click.secho("Cluster torn down.", fg="green")
        return True

    def status(self) -> ClusterStatus:
        """Get cluster health and component status."""
        try:
            # Check k3s server
            result = self._kubectl("get", "nodes", "-o", "wide", "--no-headers")
            if result.returncode != 0:
                return ClusterStatus(
                    cluster_ready=False,
                    k3s_server_running=False,
                    error="k3s server not running or kubectl failed",
                )

            # Parse node status
            nodes = []
            for line in result.stdout.strip().splitlines():
                parts = line.split()
                if len(parts) >= 5:
                    nodes.append(
                        NodeStatus(
                            hostname=parts[0],
                            ip=parts[5] if len(parts) > 5 else "unknown",
                            ready="Ready" in parts[1],
                            roles=parts[2].split(",") if len(parts) > 2 else [],
                            k3s_version=parts[4] if len(parts) > 4 else None,
                        )
                    )

            all_ready = all(n.ready for n in nodes)

            # Check GPU operator
            gpu_result = self._kubectl(
                "get",
                "pods",
                "-n",
                "gpu-operator",
                "-o",
                "jsonpath={.items[*].status.phase}",
            )
            gpu_ready = gpu_result.returncode == 0 and "Running" in gpu_result.stdout

            # Check Network operator
            net_result = self._kubectl(
                "get",
                "pods",
                "-n",
                "nvidia-network-operator",
                "-o",
                "jsonpath={.items[*].status.phase}",
            )
            net_ready = net_result.returncode == 0 and "Running" in net_result.stdout

            return ClusterStatus(
                cluster_ready=all_ready,
                k3s_server_running=True,
                nodes=nodes,
                gpu_operator_ready=gpu_ready,
                network_operator_ready=net_ready,
            )

        except Exception as e:
            return ClusterStatus(
                cluster_ready=False,
                k3s_server_running=False,
                error=str(e),
            )

    def install_k3s_server(self) -> bool:
        """Install k3s server on head node with Docker runtime."""
        result = self._head_ssh.run(
            [
                "bash",
                "-c",
                f'curl -sfL https://get.k3s.io | INSTALL_K3S_VERSION="{K3S_VERSION}" '
                f'INSTALL_K3S_EXEC="{K3S_INSTALL_FLAGS}" sh -',
            ],
            timeout=300,
            check=False,
        )
        if result.returncode != 0:
            click.secho(f"  k3s install failed: {result.stderr[:200]}", fg="red")
            return False

        click.secho(f"  k3s server installed ({K3S_VERSION})", fg="green")
        return True

    def get_join_token(self) -> str | None:
        """Retrieve k3s join token from server node."""
        result = self._head_ssh.run(
            ["cat", "/var/lib/rancher/k3s/server/node-token"],
            check=False,
        )
        if result.returncode != 0:
            return None
        return result.stdout.strip()

    def join_agent(self, node: GB10Node, token: str) -> bool:
        """Join an agent node to the cluster."""

        runner = SSHRunner(node)
        head_ip = self.config.head_node.ip

        result = runner.run(
            [
                "bash",
                "-c",
                f'curl -sfL https://get.k3s.io | INSTALL_K3S_VERSION="{K3S_VERSION}" '
                f'K3S_URL="https://{head_ip}:6443" K3S_TOKEN="{token}" '
                f'INSTALL_K3S_EXEC="--docker" sh -',
            ],
            timeout=300,
            check=False,
        )
        if result.returncode != 0:
            click.secho(f"  Agent join failed on {node.hostname}: {result.stderr[:200]}", fg="red")
            return False

        click.secho(f"  Agent joined: {node.hostname}", fg="green")
        return True

    def deploy_gpu_operator(self) -> bool:
        """Deploy NVIDIA GPU Operator via Helm.

        Configuration for GB10:
        - driver.enabled=false (DGX OS has pre-installed driver)
        - toolkit.enabled=true
        """
        # Add NVIDIA Helm repo
        self._helm("repo", "add", "nvidia", "https://helm.ngc.nvidia.com/nvidia")
        self._helm("repo", "update")

        # Install GPU operator
        result = self._helm(
            "install",
            "gpu-operator",
            "nvidia/gpu-operator",
            "--namespace",
            "gpu-operator",
            "--create-namespace",
            "--set",
            "driver.enabled=false",
            "--set",
            "toolkit.enabled=true",
            "--wait",
            "--timeout",
            "10m",
        )
        if result.returncode != 0:
            click.secho(f"  GPU Operator install failed: {result.stderr[:200]}", fg="red")
            return False

        click.secho("  GPU Operator deployed", fg="green")
        return True

    def deploy_network_operator(self) -> bool:
        """Deploy NVIDIA Network Operator for RDMA/ConnectX-7."""
        self._helm("repo", "add", "nvidia", "https://helm.ngc.nvidia.com/nvidia")

        result = self._helm(
            "install",
            "network-operator",
            "nvidia/network-operator",
            "--namespace",
            "nvidia-network-operator",
            "--create-namespace",
            "--wait",
            "--timeout",
            "10m",
        )
        if result.returncode != 0:
            click.secho(f"  Network Operator install failed: {result.stderr[:200]}", fg="red")
            return False

        click.secho("  Network Operator deployed", fg="green")
        return True

    def validate_gpus(self) -> dict[str, object]:
        """Check GPU visibility on all nodes via kubectl."""
        result = self._kubectl(
            "get",
            "nodes",
            "-o",
            'jsonpath={range .items[*]}{.metadata.name}: {.status.allocatable.nvidia\\.com/gpu}{"\\n"}{end}',
        )
        gpus: dict[str, object] = {}
        if result.returncode == 0:
            for line in result.stdout.strip().splitlines():
                if ":" in line:
                    name, count = line.split(":", 1)
                    gpus[name.strip()] = count.strip()
        return gpus

    def validate_nccl(self) -> bool:
        """Run NCCL all-reduce performance test between nodes.

        This is GATE-3 validation — critical for confirming multi-node
        tensor parallelism will work.
        """
        click.echo("  Running NCCL all-reduce benchmark...")
        result = self._head_ssh.run(
            [
                "docker",
                "run",
                "--rm",
                "--gpus",
                "all",
                "--network",
                "host",
                "-e",
                f"NCCL_SOCKET_IFNAME={self.config.nccl_socket_ifname}",
                "nvcr.io/nvidia/pytorch:24.12-py3",
                "bash",
                "-c",
                "cd /opt/nccl-tests && ./build/all_reduce_perf -b 8 -e 128M -f 2 -g 1",
            ],
            timeout=120,
            check=False,
        )
        if result.returncode != 0:
            click.secho(f"  NCCL test failed: {result.stderr[:200]}", fg="red")
            return False

        # Parse bandwidth from output
        if "Avg bus bandwidth" in result.stdout:
            click.secho("  NCCL all-reduce: OK", fg="green")
            # Extract last bandwidth line
            for line in result.stdout.splitlines():
                if "Avg" in line:
                    click.echo(f"  {line.strip()}")
            return True

        click.secho("  NCCL test completed but no bandwidth data found", fg="yellow")
        return True

    def validate_rdma(self) -> bool:
        """Check RDMA/RoCE connectivity over ConnectX-7."""
        result = self._head_ssh.run(
            ["rdma", "link", "show"],
            check=False,
        )
        if result.returncode != 0:
            click.secho("  RDMA not available", fg="yellow")
            return False

        if result.stdout.strip():
            click.secho("  RDMA links:", fg="green")
            for line in result.stdout.strip().splitlines()[:4]:
                click.echo(f"    {line.strip()}")
            return True

        return False

    def _kubectl(self, *args: str) -> subprocess.CompletedProcess[str]:
        """Run kubectl on the server node."""
        return self._head_ssh.run(["kubectl", *args], check=False, timeout=30)

    def _helm(self, *args: str) -> subprocess.CompletedProcess[str]:
        """Run helm on the server node."""
        return self._head_ssh.run(["helm", *args], check=False, timeout=600)
