"""GB10 node provisioner — vanilla DGX OS to production-ready.

All operations are idempotent: safe to re-run. Each step checks
if it's already been done before executing.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import click

from autosre.infra.ssh import SSHRunner

if TYPE_CHECKING:
    from autosre.infra.types import GB10Node

# Default data partition layout
DATA_PARTITION = "/dev/nvme0n1p3"
DATA_MOUNT = "/data"
DATA_DIRS = [
    "huggingface",
    "docker-images",
    "spark-vllm",
    "ssh-keys",
    "configs",
    "backups",
]

# Default Docker images to build
TURBOQUANT_REPO = "https://github.com/bjk110/spark_vllm_docker.git"
TURBOQUANT_BRANCH = "feat/turboquant"
FALLBACK_REPO = "https://github.com/eugr/spark-vllm-docker.git"


class Provisioner:
    """Automates GB10 node setup from vanilla DGX OS to production-ready.

    All steps are idempotent — safe to run multiple times.
    Uses SSH via SSHRunner for all remote operations.

    Storage layout on /data (preserved across OS wipes):
        /data/huggingface/    — HF model cache
        /data/docker-images/  — Saved Docker images
        /data/spark-vllm/     — spark-vllm-docker repo + cubins
        /data/ssh-keys/       — Backup of SSH keys
        /data/configs/        — Node configs
        /data/backups/        — Pre-wipe snapshots
    """

    def __init__(self, node: GB10Node, hf_token: str | None = None) -> None:
        self.node = node
        self.ssh = SSHRunner(node)
        self.hf_token = hf_token

    @property
    def _is_root(self) -> bool:
        return self.node.ssh_user == "root"

    @property
    def _home(self) -> str:
        """Home directory of the SSH user on the remote node."""
        return "/root" if self._is_root else f"/home/{self.node.ssh_user}"

    def _sudo(self, cmd: list[str]) -> list[str]:
        """Prefix ``cmd`` with ``sudo -n`` when the SSH user isn't root.

        Requires NOPASSWD sudo on the target for non-root provisioning.
        """
        if self._is_root:
            return cmd
        return ["sudo", "-n", *cmd]

    def provision(self) -> bool:
        """Full day-0 provisioning pipeline — generic GB10 prep.

        Returns True if all steps succeeded. Subclasses can override
        to layer operator-specific steps (dotfile sync, internal repo
        clone, etc.) on top of the generic prep.
        """
        steps = [
            ("Verify connectivity", self.verify_connectivity),
            ("Setup SSH keys", self.setup_ssh_keys),
            ("Configure network", self.configure_network),
            ("Setup data partition", self.setup_data_partition),
            ("Configure Docker", self.configure_docker),
            ("Setup HuggingFace cache", self.setup_hf_cache),
            ("Set performance mode", self.set_performance_mode),
            ("Configure firewall", self.configure_firewall),
        ]

        for name, step in steps:
            click.echo(f"\n{'─' * 40}")
            click.echo(f"Step: {name}")
            try:
                if not step():
                    click.secho(f"FAILED: {name}", fg="red")
                    return False
                click.secho(f"OK: {name}", fg="green")
            except Exception as e:
                click.secho(f"ERROR in {name}: {e}", fg="red")
                return False

        click.echo(f"\n{'═' * 40}")
        click.secho("Provisioning complete!", fg="green", bold=True)
        return True

    def verify_connectivity(self) -> bool:
        """Verify SSH, nvidia-smi, and Docker are working."""
        if not self.ssh.is_reachable():
            click.secho(f"  SSH unreachable: {self.node.ssh_target}", fg="red")
            return False
        click.echo(f"  SSH: OK ({self.node.ssh_target})")

        # nvidia-smi
        result = self.ssh.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            check=False,
        )
        if result.returncode != 0:
            click.secho("  nvidia-smi: FAILED", fg="red")
            return False
        click.echo(f"  GPU: {result.stdout.strip()}")

        # Docker — check via ``docker info``; if the SSH user isn't in
        # the docker group, retry through ``sg docker`` so we can
        # distinguish "daemon missing" from "user lacks group" (the
        # latter is fixable by ``configure_docker`` below).
        result = self.ssh.run(
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            check=False,
        )
        if result.returncode != 0:
            sg_result = self.ssh.run(
                ["sg", "docker", "-c", "docker info --format {{.ServerVersion}}"],
                check=False,
            )
            if sg_result.returncode == 0:
                click.secho(
                    f"  Docker: {sg_result.stdout.strip()} "
                    f"(user lacks docker group — configure_docker will fix)",
                    fg="yellow",
                )
                return True
            click.secho("  Docker: FAILED", fg="red")
            return False
        click.echo(f"  Docker: {result.stdout.strip()}")
        return True

    def setup_ssh_keys(self) -> bool:
        """Ensure SSH key-based auth is configured, disable password auth.

        Copies the local SSH public key to the node's authorized_keys
        and hardens sshd configuration.
        """
        ssh_dir = f"{self._home}/.ssh"
        auth_keys = f"{ssh_dir}/authorized_keys"

        result = self.ssh.run(["test", "-f", auth_keys], check=False)
        if result.returncode == 0:
            click.echo(f"  SSH authorized_keys exists ({auth_keys})")
        else:
            self.ssh.run(["mkdir", "-p", ssh_dir])
            self.ssh.run(["chmod", "700", ssh_dir])
            self.ssh.run(["touch", auth_keys])
            self.ssh.run(["chmod", "600", auth_keys])
            click.echo(f"  Created {auth_keys}")

        # Disable password auth (idempotent). Needs sudo for non-root users.
        self.ssh.run(
            self._sudo(
                [
                    "sed",
                    "-i",
                    r"s/^#\?PasswordAuthentication.*/PasswordAuthentication no/",
                    "/etc/ssh/sshd_config",
                ],
            ),
            check=False,
        )
        click.echo("  Password auth disabled in sshd_config")

        return True

    def configure_network(self) -> bool:
        """Configure ConnectX-7 interfaces for NCCL communication.

        Sets up static IP on the high-speed interconnect if needed,
        and identifies the NCCL socket interface name.
        """
        # Identify ConnectX-7 interfaces
        result = self.ssh.run(
            ["bash", "-c", "ip -o link show | grep -i 'enp.*np' | awk '{print $2}' | tr -d ':'"],
            check=False,
        )
        interfaces = [
            iface.strip() for iface in result.stdout.strip().splitlines() if iface.strip()
        ]

        if interfaces:
            click.echo(f"  ConnectX-7 interfaces: {', '.join(interfaces)}")
        else:
            click.echo("  No ConnectX-7 interfaces detected (will use default)")

        # Check if interfaces are UP
        for iface in interfaces[:2]:  # Check first 2 (one per QSFP port)
            result = self.ssh.run(
                ["bash", "-c", f"ip link show {iface} | grep -q 'state UP'"],
                check=False,
            )
            state = "UP" if result.returncode == 0 else "DOWN"
            click.echo(f"  {iface}: {state}")

        return True

    def setup_data_partition(self) -> bool:
        """Ensure /data has the directory structure. Handles two layouts:

        1. Separate partition :data:`DATA_PARTITION` — mounted at
           :data:`DATA_MOUNT` and added to ``/etc/fstab``.
        2. Single-partition install (root partition == DATA_PARTITION) —
           :data:`DATA_MOUNT` is a regular directory on the root
           filesystem. Never try to mount root over /data.

        IMPORTANT: Never reformats an existing /data mount.
        """
        # Resolve the source device for both / and DATA_PARTITION so we can
        # tell whether DATA_PARTITION is a distinct partition or just
        # the root.
        root_source = self.ssh.run(
            ["findmnt", "-n", "-o", "SOURCE", "/"],
            check=False,
        ).stdout.strip()
        data_partition_is_root = root_source == DATA_PARTITION

        if data_partition_is_root:
            click.echo(
                f"  {DATA_PARTITION} is the root filesystem — using "
                f"{DATA_MOUNT} as a directory (single-partition install).",
            )
            # If something (an earlier buggy run, or a manual operator)
            # mounted root over DATA_MOUNT, unmount so we don't have two
            # mount points for the same device.
            dup_mount = self.ssh.run(
                ["findmnt", "-n", "-o", "SOURCE", DATA_MOUNT],
                check=False,
            ).stdout.strip()
            if dup_mount == DATA_PARTITION:
                click.secho(
                    f"  WARNING: {DATA_MOUNT} is a duplicate mount of root — unmounting",
                    fg="yellow",
                )
                self.ssh.run(self._sudo(["umount", DATA_MOUNT]), check=False)
            # Strip the stale fstab line if a prior run added it.
            self.ssh.run(
                self._sudo(
                    [
                        "sed",
                        "-i",
                        f"\\|^{DATA_PARTITION}[[:space:]]\\+{DATA_MOUNT}\\b|d",
                        "/etc/fstab",
                    ],
                ),
                check=False,
            )
            self.ssh.run(self._sudo(["mkdir", "-p", DATA_MOUNT]))
        else:
            # Separate partition layout.
            result = self.ssh.run(["mountpoint", "-q", DATA_MOUNT], check=False)
            if result.returncode == 0:
                click.echo(f"  {DATA_MOUNT} already mounted")
            else:
                result = self.ssh.run(
                    ["lsblk", "-no", "NAME", DATA_PARTITION],
                    check=False,
                )
                if result.returncode != 0:
                    click.secho(
                        f"  Partition {DATA_PARTITION} not found. Manual partitioning required.",
                        fg="yellow",
                    )
                    return False

                result = self.ssh.run(
                    ["blkid", "-o", "value", "-s", "TYPE", DATA_PARTITION],
                    check=False,
                )
                if not result.stdout.strip():
                    click.echo(f"  Formatting {DATA_PARTITION} as ext4...")
                    self.ssh.run(self._sudo(["mkfs.ext4", "-L", "data", DATA_PARTITION]))

                self.ssh.run(self._sudo(["mkdir", "-p", DATA_MOUNT]))
                self.ssh.run(self._sudo(["mount", DATA_PARTITION, DATA_MOUNT]))

                result = self.ssh.run(["grep", "-q", DATA_MOUNT, "/etc/fstab"], check=False)
                if result.returncode != 0:
                    fstab_line = f"{DATA_PARTITION} {DATA_MOUNT} ext4 defaults 0 2"
                    self.ssh.run(
                        self._sudo(["bash", "-c", f"echo '{fstab_line}' >> /etc/fstab"]),
                    )
                    click.echo("  Added to /etc/fstab")

        # Create directory structure (owned by SSH user so non-root writes work).
        for dirname in DATA_DIRS:
            self.ssh.run(self._sudo(["mkdir", "-p", f"{DATA_MOUNT}/{dirname}"]))
        if not self._is_root:
            self.ssh.run(
                self._sudo(
                    ["chown", "-R", f"{self.node.ssh_user}:{self.node.ssh_user}", DATA_MOUNT],
                ),
                check=False,
            )
        click.echo(f"  Directories: {', '.join(DATA_DIRS)}")

        return True

    def configure_docker(self) -> bool:
        """Ensure NVIDIA runtime is default Docker runtime with log rotation,
        and the SSH user is in the docker group."""
        # Put the SSH user in the docker group so non-sudo docker works on
        # subsequent sessions. Idempotent — usermod -aG is a no-op if already
        # a member. (Active SSH session won't see the new group; new shells will.)
        self.ssh.run(
            ["sudo", "usermod", "-aG", "docker", self.node.ssh_user],
            check=False,
        )

        # Read current daemon.json (JSON parse, not substring match —
        # substring match was fooled by incomplete configs that contained
        # ``"nvidia"`` without the matching ``runtimes.nvidia.path`` entry).
        import json as _json

        result = self.ssh.run(
            ["bash", "-c", "cat /etc/docker/daemon.json 2>/dev/null || echo '{}'"],
            check=False,
        )
        try:
            current_cfg = _json.loads(result.stdout.strip() or "{}")
        except _json.JSONDecodeError:
            current_cfg = {}

        if (
            current_cfg.get("default-runtime") == "nvidia"
            and isinstance(current_cfg.get("runtimes"), dict)
            and "nvidia" in current_cfg["runtimes"]
        ):
            click.echo("  NVIDIA runtime already configured as default")
            return True

        # Write Docker daemon config
        config = (
            '{"default-runtime":"nvidia",'
            '"runtimes":{"nvidia":{"path":"nvidia-container-runtime","runtimeArgs":[]}},'
            '"log-driver":"json-file",'
            '"log-opts":{"max-size":"50m","max-file":"3"}}'
        )
        # Write daemon.json through sudo tee so non-root users can update it.
        self.ssh.run(
            self._sudo(["bash", "-c", f"echo '{config}' > /etc/docker/daemon.json"]),
        )
        self.ssh.run(self._sudo(["systemctl", "restart", "docker"]), timeout=60)
        click.echo("  Docker configured with NVIDIA default runtime + log rotation")

        return True

    def setup_hf_cache(self) -> bool:
        """Configure HuggingFace model cache directory."""
        cache_dir = f"{DATA_MOUNT}/huggingface"
        self.ssh.run(self._sudo(["mkdir", "-p", cache_dir]))
        if not self._is_root:
            self.ssh.run(
                self._sudo(
                    ["chown", "-R", f"{self.node.ssh_user}:{self.node.ssh_user}", cache_dir],
                ),
                check=False,
            )

        # Set HF_HOME for all users
        env_line = f'export HF_HOME="{cache_dir}"'
        result = self.ssh.run(
            ["grep", "-q", "HF_HOME", "/etc/profile.d/autosre.sh"],
            check=False,
        )
        if result.returncode != 0:
            self.ssh.run(
                self._sudo(
                    ["bash", "-c", f"echo '{env_line}' >> /etc/profile.d/autosre.sh"],
                ),
            )

        # Set HF_TOKEN if provided
        if self.hf_token:
            token_line = f'export HF_TOKEN="{self.hf_token}"'
            self.ssh.run(
                self._sudo(
                    ["bash", "-c", f"echo '{token_line}' >> /etc/profile.d/autosre.sh"],
                ),
            )
            click.echo("  HF_TOKEN configured")

        click.echo(f"  HF_HOME={cache_dir}")
        return True

    def set_performance_mode(self) -> bool:
        """Set CPU governor to performance mode."""
        self.ssh.run(
            self._sudo(
                [
                    "bash",
                    "-c",
                    "echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor",
                ],
            ),
            check=False,
        )
        click.echo("  CPU governor: performance")
        return True

    def configure_firewall(self) -> bool:
        """Open required ports (vLLM, Ray, NCCL)."""
        ports = [
            ("8000", "tcp", "vLLM API"),
            ("6379", "tcp", "Ray head"),
            ("6380-6400", "tcp", "Ray workers"),
            ("29500-29600", "tcp", "NCCL"),
            ("6443", "tcp", "k3s API"),
        ]

        # Check if ufw is active
        result = self.ssh.run(["ufw", "status"], check=False)
        if "inactive" in result.stdout.lower():
            click.echo("  Firewall inactive, no changes needed")
            return True

        for port, proto, desc in ports:
            self.ssh.run(["ufw", "allow", f"{port}/{proto}"], check=False)
            click.echo(f"  Opened {port}/{proto} ({desc})")

        return True

    def build_vllm_image(self, with_turboquant: bool = True) -> bool:
        """Build spark-vllm-docker image on the node.

        Primary: bjk110/spark_vllm_docker feat/turboquant branch
        Fallback: eugr/spark-vllm-docker main branch
        """
        build_dir = f"{DATA_MOUNT}/spark-vllm"

        if with_turboquant:
            repo = TURBOQUANT_REPO
            branch = TURBOQUANT_BRANCH
            tag = "bjk110/spark-vllm:turboquant"
        else:
            repo = FALLBACK_REPO
            branch = "main"
            tag = "eugr/spark-vllm:latest"

        # Clone or update repo
        result = self.ssh.run(
            ["test", "-d", f"{build_dir}/.git"],
            check=False,
        )
        if result.returncode == 0:
            click.echo("  Updating existing repo...")
            self.ssh.run(
                [
                    "bash",
                    "-c",
                    f"cd {build_dir} && git fetch origin && git checkout {branch} && git pull",
                ],
                timeout=120,
            )
        else:
            click.echo(f"  Cloning {repo} ({branch})...")
            self.ssh.run(
                ["git", "clone", "-b", branch, repo, build_dir],
                timeout=300,
            )

        # Build Docker image (this takes ~30 min first time)
        click.echo(f"  Building Docker image: {tag}")
        click.echo("  This may take 20-30 minutes on first build...")
        self.ssh.run(
            ["bash", "-c", f"cd {build_dir} && docker build -t {tag} ."],
            timeout=3600,  # 1 hour timeout
        )

        click.echo(f"  Image built: {tag}")
        return True

    def pull_default_models(self, models: list[str] | None = None) -> bool:
        """Pre-download NVFP4 models to HuggingFace cache."""
        if models is None:
            models = [
                "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4",
            ]

        cache_dir = f"{DATA_MOUNT}/huggingface"
        for model_id in models:
            click.echo(f"  Downloading {model_id}...")
            self.ssh.run(
                [
                    "bash",
                    "-c",
                    f'HF_HOME="{cache_dir}" huggingface-cli download "{model_id}"',
                ],
                timeout=3600,
                check=False,
            )

        return True

    # === User-level parity ===

    def _scp_to_target(self, local_path: str, remote_path: str) -> bool:
        """SCP a local file to the target. Returns True on success."""
        import subprocess as _sub

        cmd = [
            "scp",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            "ConnectTimeout=10",
            "-o",
            "BatchMode=yes",
        ]
        if self.node.ssh_key:
            cmd.extend(["-i", self.node.ssh_key])
        cmd.extend([local_path, f"{self.node.ssh_target}:{remote_path}"])
        result = _sub.run(cmd, capture_output=True, text=True, timeout=60, check=False)
        if result.returncode != 0:
            click.secho(f"  scp failed: {result.stderr.strip()[:200]}", fg="yellow")
            return False
        return True

    def validate(self) -> tuple[bool, list[str]]:
        """Run full validation suite.

        Returns (all_pass, list of issues).
        """
        issues: list[str] = []

        # SSH
        if not self.ssh.is_reachable():
            issues.append(f"SSH unreachable: {self.node.ssh_target}")
            return False, issues

        # nvidia-smi
        result = self.ssh.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=False,
        )
        if result.returncode != 0 or not result.stdout.strip():
            issues.append("nvidia-smi failed")

        # Docker — retry via ``sg docker`` if the SSH session's group
        # cache pre-dates configure_docker adding the user to the group.
        def _docker_info(fmt: str) -> tuple[int, str]:
            r = self.ssh.run(["docker", "info", "--format", fmt], check=False)
            if r.returncode == 0:
                return r.returncode, r.stdout
            r = self.ssh.run(
                ["sg", "docker", "-c", f"docker info --format {fmt}"],
                check=False,
            )
            return r.returncode, r.stdout

        rc, _ = _docker_info("{{.ServerVersion}}")
        if rc != 0:
            issues.append("Docker not responding")

        # NVIDIA runtime
        rc, out = _docker_info("{{.DefaultRuntime}}")
        if "nvidia" not in out.lower():
            issues.append("NVIDIA runtime not default Docker runtime")

        # /data present — either as a mountpoint (separate partition
        # layout) or as a directory on the root filesystem (single-
        # partition install). Both are valid.
        mp = self.ssh.run(["mountpoint", "-q", DATA_MOUNT], check=False)
        if mp.returncode != 0:
            dir_check = self.ssh.run(["test", "-d", DATA_MOUNT], check=False)
            if dir_check.returncode != 0:
                issues.append(f"{DATA_MOUNT} missing (neither a mount nor a directory)")

        # HF cache dir
        result = self.ssh.run(["test", "-d", f"{DATA_MOUNT}/huggingface"], check=False)
        if result.returncode != 0:
            issues.append("HuggingFace cache directory missing")

        return len(issues) == 0, issues

    # === Pre-wipe / Post-wipe ===

    def pre_wipe_backup(self) -> bool:
        """Backup critical state to /data/backups/ before OS wipe.

        Backs up: SSH keys, node configs, Docker image list.
        /data/ partition is preserved across wipes (separate partition).
        """
        backup_dir = f"{DATA_MOUNT}/backups"
        self.ssh.run(["mkdir", "-p", backup_dir])

        # SSH keys (from the SSH user's home, not hardcoded /root).
        self.ssh.run(
            [
                "bash",
                "-c",
                f"cp -a {self._home}/.ssh {DATA_MOUNT}/ssh-keys/ 2>/dev/null || true",
            ],
        )
        click.echo("  Backed up SSH keys")

        # Docker image list
        self.ssh.run(
            [
                "bash",
                "-c",
                f"docker images --format '{{{{.Repository}}}}:{{{{.Tag}}}}' > {backup_dir}/docker-images.txt",
            ],
        )
        click.echo("  Saved Docker image list")

        # Any autosre config (from the SSH user's XDG data dir).
        self.ssh.run(
            [
                "bash",
                "-c",
                f"cp -a {self._home}/.local/share/autosre/* {DATA_MOUNT}/configs/ 2>/dev/null || true",
            ],
        )
        click.echo("  Backed up autosre configs")

        return True

    def post_wipe_restore(self) -> bool:
        """Restore state from /data/backups/ after fresh OS install.

        Restores SSH keys, then re-runs the full provision pipeline.
        """
        # Restore SSH keys into the SSH user's home.
        ssh_dir = f"{self._home}/.ssh"
        self.ssh.run(
            ["bash", "-c", f"cp -a {DATA_MOUNT}/ssh-keys/.ssh {self._home}/ 2>/dev/null || true"],
        )
        self.ssh.run(["chmod", "700", ssh_dir], check=False)
        self.ssh.run(["chmod", "600", f"{ssh_dir}/authorized_keys"], check=False)
        click.echo(f"  Restored SSH keys to {ssh_dir}")

        # Restore autosre configs to the SSH user's XDG data dir.
        autosre_data = f"{self._home}/.local/share/autosre"
        self.ssh.run(["mkdir", "-p", autosre_data])
        self.ssh.run(
            [
                "bash",
                "-c",
                f"cp -a {DATA_MOUNT}/configs/* {autosre_data}/ 2>/dev/null || true",
            ],
        )
        click.echo("  Restored autosre configs")

        # Re-run provision
        click.echo("  Re-running provisioning pipeline...")
        return self.provision()

    def save_docker_images(self, images: list[str] | None = None) -> bool:
        """Save Docker images to /data/docker-images/ for reuse after wipe."""
        if images is None:
            # Default: save autosre vLLM images
            result = self.ssh.run(
                [
                    "bash",
                    "-c",
                    "docker images --format '{{.Repository}}:{{.Tag}}' | grep -E 'spark-vllm|autosre'",
                ],
                check=False,
            )
            images = [img for img in result.stdout.strip().splitlines() if img]

        save_dir = f"{DATA_MOUNT}/docker-images"
        for image in images:
            safe_name = image.replace("/", "_").replace(":", "_")
            click.echo(f"  Saving {image}...")
            self.ssh.run(
                ["bash", "-c", f"docker save {image} | gzip > {save_dir}/{safe_name}.tar.gz"],
                timeout=600,
            )

        click.echo(f"  Saved {len(images)} image(s)")
        return True

    def load_docker_images(self) -> bool:
        """Load saved Docker images from /data/docker-images/."""
        load_dir = f"{DATA_MOUNT}/docker-images"
        result = self.ssh.run(
            ["bash", "-c", f"ls {load_dir}/*.tar.gz 2>/dev/null"],
            check=False,
        )
        files = [f for f in result.stdout.strip().splitlines() if f]

        for filepath in files:
            click.echo(f"  Loading {filepath}...")
            self.ssh.run(
                ["bash", "-c", f"gunzip -c {filepath} | docker load"],
                timeout=600,
            )

        click.echo(f"  Loaded {len(files)} image(s)")
        return True
