"""Base backend interface."""

import json
import os
import platform
import signal
import subprocess
import time
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import ClassVar

import click
import httpx


class BackendType(Enum):
    """Available backend types."""

    OLLAMA = "ollama"
    LLAMACPP = "llamacpp"
    VLLM = "vllm"
    MLX_DFLASH = "mlx-dflash"


def detect_platform() -> BackendType:
    """Auto-detect the best backend for the current platform.

    Ollama is the universal default. llama.cpp is opt-in via -b llamacpp.
    """
    return BackendType.OLLAMA


# Active backend state file path
ACTIVE_STATE_FILE = (
    Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    / "autosre"
    / "active.json"
)


def load_active_state() -> dict[str, object] | None:
    """Load active backend state from active.json.

    If active.json is missing but a tracked vLLM container is still running,
    reconstruct the state from the live container and write it back. The vLLM
    container survives autosre process death / reboot via
    ``docker --restart unless-stopped``, but active.json does not — reconcile
    lets subsequent commands reattach instead of demanding ``autosre start``.
    """
    if ACTIVE_STATE_FILE.exists():
        try:
            data: dict[str, object] = json.loads(ACTIVE_STATE_FILE.read_text())
            return data
        except (json.JSONDecodeError, OSError):
            pass
    # Cheap guard: only attempt reconcile when vllm tracked a container.
    if not (ACTIVE_STATE_FILE.parent / "vllm_containers.json").exists():
        return None
    try:
        from .vllm import VllmBackend

        return VllmBackend.reconcile_state()
    except Exception:
        return None


def save_active_state(state: dict[str, object]) -> None:
    """Save active backend state to active.json."""
    ACTIVE_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    ACTIVE_STATE_FILE.write_text(json.dumps(state))


def clear_active_state() -> None:
    """Clear the active backend state file."""
    if ACTIVE_STATE_FILE.exists():
        ACTIVE_STATE_FILE.unlink()


class Backend(ABC):
    """Abstract base class for inference backends."""

    name: ClassVar[str] = "base"
    description: ClassVar[str] = "Base backend"

    # Default API port
    api_port: ClassVar[int] = 8080

    # Default model configurations
    models: ClassVar[dict[str, str]] = {}
    default_model: ClassVar[str] = ""

    def __init__(self, active_state: dict[str, object] | None = None) -> None:
        """Initialize the backend.

        Args:
            active_state: Previously saved state from active.json, used to
                reconstruct the backend after process restart. Contains
                backend-specific keys (e.g., api_host, containers for vLLM).
                Local backends (Ollama, llamacpp) can ignore this.
        """
        self._active_state = active_state

    def get_api_url(self) -> str:
        """Get the full API base URL.

        Default returns http://localhost:{api_port}. Remote backends (e.g., vLLM)
        override this to return the remote node's URL.
        """
        return f"http://localhost:{self.api_port}"

    @abstractmethod
    def check_requirements(self) -> tuple[bool, list[str]]:
        """Check if requirements are met for this backend.

        Returns:
            Tuple of (success, list of missing requirements)
        """
        pass

    @abstractmethod
    def setup(self, force: bool = False, **kwargs: object) -> bool:
        """Set up the backend environment.

        Args:
            force: Force reinstall even if already set up

        Returns:
            True if setup was successful
        """
        pass

    @abstractmethod
    def start(
        self,
        model: str | None = None,
        foreground: bool = False,
        **kwargs: object,
    ) -> dict[str, object]:
        """Start the inference server.

        Args:
            model: Model identifier to use
            foreground: Run in foreground with logs

        Returns:
            Dict with process info (pids, ports, etc.)
        """
        pass

    @abstractmethod
    def stop(self, **kwargs: object) -> bool:
        """Stop the inference server.

        Args:
            **kwargs: Backend-specific options (e.g., unload_model for vLLM).

        Returns:
            True if stopped successfully
        """
        pass

    @abstractmethod
    def status(self) -> dict[str, object]:
        """Get the current status of the backend.

        Returns:
            Dict with status info (running, model, ports, etc.)
        """
        pass

    @abstractmethod
    def get_claude_env(self) -> dict[str, str]:
        """Get environment variables for Claude Code.

        Returns:
            Dict with ANTHROPIC_BASE_URL, ANTHROPIC_AUTH_TOKEN, etc.
        """
        pass

    @abstractmethod
    def get_claude_model_arg(self, model: str) -> str:
        """Get the model ID to pass via claude --model.

        Args:
            model: Short model key or full model ID

        Returns:
            Model ID string for the --model flag
        """
        pass

    @abstractmethod
    def is_healthy(self) -> bool:
        """Check if the backend is ready to serve.

        Returns:
            True if the backend is healthy and responding
        """
        pass

    def get_model_id(self, model_key: str) -> str:
        """Get the full model ID from a short key."""
        return self.models.get(model_key, model_key)

    @property
    def data_dir(self) -> Path:
        """Get the data directory (XDG compliant)."""
        xdg_data = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
        data_dir = xdg_data / "autosre"
        data_dir.mkdir(parents=True, exist_ok=True)
        return data_dir

    @property
    def pids_file(self) -> Path:
        """Get the PIDs file path. Override in subclass for custom name."""
        return self.data_dir / f"{self.name}.pids"

    # Common process management methods

    def _load_pids(self) -> dict[str, object]:
        """Load saved PIDs from the pids file."""
        if self.pids_file.exists():
            try:
                result: dict[str, object] = json.loads(self.pids_file.read_text())
                return result
            except (json.JSONDecodeError, OSError):
                pass
        return {}

    def _save_pids(self, pids: dict[str, object]) -> None:
        """Save PIDs to the pids file."""
        self.pids_file.write_text(json.dumps(pids))

    def _clear_pids(self) -> None:
        """Remove the pids file."""
        if self.pids_file.exists():
            self.pids_file.unlink()

    def _is_our_process(self, pid: int, name: str) -> bool:
        """Check if a PID belongs to a process we started.

        Args:
            pid: Process ID to check
            name: Expected process name (e.g., 'llama-server', 'ollama')

        Returns:
            True if the process exists and matches expected name
        """
        try:
            # Check if process exists
            os.kill(pid, 0)

            # Get process command line
            if platform.system() == "Darwin":
                result = subprocess.run(
                    ["ps", "-p", str(pid), "-o", "command="],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                cmdline = result.stdout.strip()
            else:
                cmdline_path = Path(f"/proc/{pid}/cmdline")
                if cmdline_path.exists():
                    cmdline = cmdline_path.read_text().replace("\x00", " ")
                else:
                    return False

            # Verify it's our process by checking cmdline
            process_patterns = {
                "llama-server": "llama-server",
                "llama_server": "llama-server",
                "ollama": "ollama",
            }
            pattern = process_patterns.get(name, name)
            return pattern in cmdline

        except (ProcessLookupError, PermissionError, FileNotFoundError):
            return False

    def _stop_process(self, pid: int, name: str) -> bool:
        """Stop a process with graceful then forced kill.

        Args:
            pid: Process ID to stop
            name: Process name for verification

        Returns:
            True if process was stopped
        """
        if not self._is_our_process(pid, name):
            return False

        try:
            # Graceful shutdown
            os.kill(pid, signal.SIGTERM)
            click.echo(f"Stopped {name} (PID {pid})")

            # Wait briefly for graceful shutdown
            time.sleep(0.5)

            # Force kill if still running
            try:
                os.kill(pid, 0)  # Check if still alive
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass  # Already dead, good

            return True
        except ProcessLookupError:
            return False

    def _stop_tracked_processes(self) -> bool:
        """Stop all processes tracked in the PIDs file.

        Returns:
            True if any processes were stopped
        """
        pids = self._load_pids()
        stopped = False

        for name, pid in pids.items():
            # Skip non-PID entries (like 'model')
            if not isinstance(pid, int):
                continue
            if self._stop_process(pid, name):
                stopped = True

        self._clear_pids()
        return stopped

    def _is_port_in_use(self, port: int) -> bool:
        """Check if a local port is in use by making an HTTP request."""
        try:
            with httpx.Client(timeout=2) as client:
                client.get(f"http://localhost:{port}")
            return True
        except Exception:
            return False

    def _wait_for_server(self, url: str, timeout: int = 120) -> bool:
        """Wait for a server to become ready.

        Args:
            url: URL to check (should return 200 when ready)
            timeout: Maximum seconds to wait

        Returns:
            True if server is ready, False if timeout
        """
        start = time.time()
        while time.time() - start < timeout:
            try:
                with httpx.Client(timeout=5) as client:
                    resp = client.get(url)
                    if resp.status_code == 200:
                        return True
            except Exception:
                pass
            time.sleep(2)
        return False
