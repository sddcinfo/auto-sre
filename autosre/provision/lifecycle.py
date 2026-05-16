"""Rolling rebuild and lifecycle management for GB10 node clusters.

Handles wipe/rebuild while maintaining service availability
by degrading to a solo model during maintenance windows.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import click

from autosre.infra.ssh import SSHRunner
from autosre.infra.types import SOLO_FALLBACK_MODEL, NodeRole
from autosre.provision.provisioner import Provisioner

if TYPE_CHECKING:
    from autosre.infra.types import GB10Node


class NodeLifecycle:
    """Manages wipe/rebuild across a multi-node GB10 cluster.

    Key constraint: TP=2 models (nemotron-super, qwen3.6-122b, qwen3.6-397b)
    require both nodes. During rebuild of either node, service degrades to
    a solo model on the surviving node.
    """

    def __init__(self, nodes: list[GB10Node]) -> None:
        self.nodes = nodes

    @property
    def head_node(self) -> GB10Node:
        for node in self.nodes:
            if node.role is NodeRole.HEAD:
                return node
        return self.nodes[0]

    @property
    def worker_nodes(self) -> list[GB10Node]:
        head = self.head_node
        return [n for n in self.nodes if n is not head]

    def rolling_rebuild(self, start_with: NodeRole = NodeRole.WORKER) -> bool:
        """Rebuild nodes one at a time with service degradation.

        Strategy:
        1. If TP=2 model is running: stop it, start solo fallback
        2. Sync data to surviving node
        3. Pre-wipe backup
        4. User physically wipes (F7 -> USB -> reinstall)
        5. Wait for node to come back
        6. Restore + re-provision
        7. Validate
        8. Repeat for second node
        9. Optionally restart TP=2 model after both nodes ready

        Returns True if all nodes rebuilt successfully.
        """
        # Determine rebuild order
        if start_with is NodeRole.WORKER and self.worker_nodes:
            rebuild_order = [*self.worker_nodes, self.head_node]
        else:
            rebuild_order = [self.head_node, *self.worker_nodes]

        click.secho("\nRolling Rebuild Plan", fg="cyan", bold=True)
        click.echo(f"  Nodes: {len(self.nodes)}")
        click.echo(f"  Order: {' -> '.join(n.hostname for n in rebuild_order)}")
        click.echo()
        click.secho(
            "WARNING: Cluster models (nemotron-super, qwen3.6-122b, qwen3.6-397b)\n"
            f"  require both nodes. During rebuild, service degrades to: {SOLO_FALLBACK_MODEL}",
            fg="yellow",
        )

        for i, node in enumerate(rebuild_order):
            surviving = [n for n in self.nodes if n is not node]
            click.echo(f"\n{'═' * 50}")
            click.secho(
                f"Rebuilding node {i + 1}/{len(rebuild_order)}: {node.hostname} ({node.ip})",
                bold=True,
            )

            if surviving:
                click.echo(f"  Surviving node: {surviving[0].hostname} ({surviving[0].ip})")

            # Step 1: Sync models to ensure data is on both nodes
            if surviving:
                click.echo("\n  Syncing models to surviving node...")
                self.sync_models(node, surviving[0])

            # Step 2: Pre-wipe backup
            click.echo("\n  Running pre-wipe backup...")
            provisioner = Provisioner(node)
            provisioner.pre_wipe_backup()

            # Step 3: Save Docker images
            click.echo("  Saving Docker images...")
            provisioner.save_docker_images()

            # Step 4: User performs physical wipe
            click.echo()
            click.secho("ACTION REQUIRED:", fg="yellow", bold=True)
            click.echo(f"  1. Physically wipe {node.hostname} ({node.ip})")
            click.echo("  2. Boot from USB (press F7 at startup)")
            click.echo("  3. Install DGX OS (~25 minutes)")
            click.echo("  4. Complete first-boot setup wizard")
            click.echo("  5. Ensure SSH key access is restored")
            click.echo()

            if not click.confirm(f"Has {node.hostname} been reimaged and is reachable via SSH?"):
                click.secho("Rebuild paused. Resume when ready.", fg="yellow")
                return False

            # Step 5: Wait for SSH
            click.echo("  Verifying SSH connectivity...")
            runner = SSHRunner(node)
            if not runner.is_reachable(timeout=30):
                click.secho(f"  Cannot reach {node.ip} via SSH", fg="red")
                return False
            click.secho("  SSH OK", fg="green")

            # Step 6: Post-wipe restore + re-provision
            click.echo("  Restoring state and re-provisioning...")
            provisioner = Provisioner(node)
            if not provisioner.post_wipe_restore():
                click.secho(f"  Restore failed for {node.hostname}", fg="red")
                return False

            # Step 7: Load saved Docker images
            click.echo("  Loading saved Docker images...")
            provisioner.load_docker_images()

            # Step 8: Validate
            click.echo("  Validating...")
            ok, issues = provisioner.validate()
            if not ok:
                click.secho(f"  Validation failed: {issues}", fg="red")
                return False

            click.secho(f"  {node.hostname} rebuilt successfully!", fg="green")

        click.echo(f"\n{'═' * 50}")
        click.secho("All nodes rebuilt successfully!", fg="green", bold=True)
        click.echo(
            "Cluster models available again. Start with: autosre start -b vllm -m nemotron-super"
        )
        return True

    def sync_models(self, source: GB10Node, dest: GB10Node) -> bool:
        """Rsync HuggingFace models between nodes.

        Uses ConnectX-7 for high-speed transfer (~185 Gbps).
        """
        runner = SSHRunner(source)
        src_path = "/data/huggingface/"
        dest_path = f"{dest.ssh_user}@{dest.ip}:/data/huggingface/"

        click.echo(f"  Syncing models: {source.ip} -> {dest.ip}")
        result = runner.rsync(
            src_path,
            dest_path,
            timeout=3600,
        )
        if result:
            click.secho("  Model sync complete", fg="green")
        else:
            click.secho("  Model sync failed (non-fatal)", fg="yellow")
        return result

    def sync_docker_images(self, source: GB10Node, dest: GB10Node) -> bool:
        """Transfer saved Docker images between nodes."""
        runner = SSHRunner(source)
        src_path = "/data/docker-images/"
        dest_path = f"{dest.ssh_user}@{dest.ip}:/data/docker-images/"

        click.echo(f"  Syncing Docker images: {source.ip} -> {dest.ip}")
        return runner.rsync(src_path, dest_path, timeout=3600)

    def full_wipe_both(self) -> bool:
        """Wipe and rebuild all nodes. Requires all models to be re-downloaded.

        This is the nuclear option: both nodes get wiped.
        Data in /data/ is preserved (separate partition), but cluster state
        and k3s must be fully re-bootstrapped.
        """
        click.secho("FULL WIPE: All nodes will be rebuilt.", fg="red", bold=True)
        click.echo("Data on /data/ partitions is preserved.")
        click.echo("K3s cluster state will be lost and must be re-bootstrapped.")

        if not click.confirm("Continue with full wipe of all nodes?"):
            return False

        for node in self.nodes:
            provisioner = Provisioner(node)
            provisioner.pre_wipe_backup()
            provisioner.save_docker_images()

        click.echo("\nAll nodes backed up. Proceed with physical reimaging.")
        click.echo("After reimaging all nodes, run:")
        click.echo("  autosre provision restore <node-ip>  (for each node)")
        return True
