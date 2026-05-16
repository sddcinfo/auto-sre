"""Ollama backend with native Anthropic Messages API support."""

import re
import shutil
import subprocess
import time

import click
import httpx

from .base import Backend, clear_active_state, save_active_state

# Minimum Ollama version for native Anthropic Messages API support
# Ollama added /v1/messages in version 0.14.0
OLLAMA_MIN_ANTHROPIC_VERSION = (0, 14, 0)


class OllamaBackend(Backend):
    """Ollama backend with native Anthropic Messages API.

    Requires Ollama >= 0.14.0 for /v1/messages support.
    """

    name = "ollama"
    description = "Ollama (native Anthropic API, universal)"

    models = {
        "qwen3.6": "qwen3.6:35b-a3b",
    }
    default_model = "qwen3.6"

    api_port = 11434
    ollama_port = 11434  # alias for backward compat

    def check_requirements(self) -> tuple[bool, list[str]]:
        """Check if Ollama is available."""
        missing = []

        if not shutil.which("ollama"):
            missing.append("Ollama (install from https://ollama.ai)")

        return len(missing) == 0, missing

    def _get_ollama_version(self) -> tuple[int, int, int] | None:
        """Get the installed Ollama version.

        Returns:
            Version tuple (major, minor, patch) or None if unavailable.
        """
        try:
            result = subprocess.run(
                ["ollama", "--version"], capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                match = re.search(r"(\d+)\.(\d+)\.(\d+)", result.stdout)
                if match:
                    return (int(match.group(1)), int(match.group(2)), int(match.group(3)))
        except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
            pass
        return None

    def _supports_anthropic_api(self) -> bool:
        """Check if Ollama version supports native Anthropic Messages API."""
        version = self._get_ollama_version()
        if version is None:
            return False
        return version >= OLLAMA_MIN_ANTHROPIC_VERSION

    def setup(
        self,
        force: bool = False,  # noqa: ARG002
        **kwargs: object,  # noqa: ARG002
    ) -> bool:
        """Set up Ollama environment."""
        click.secho("=== Ollama Backend Setup ===", fg="cyan", bold=True)

        if not shutil.which("ollama"):
            click.secho("ERROR: Ollama not installed", fg="red")
            click.echo("Install from: https://ollama.ai")
            return False

        version = self._get_ollama_version()
        if version:
            version_str = ".".join(map(str, version))
            click.echo(f"Ollama version: {version_str}")
            if self._supports_anthropic_api():
                click.secho("Supports native Anthropic Messages API", fg="green")
            else:
                min_version = ".".join(map(str, OLLAMA_MIN_ANTHROPIC_VERSION))
                click.secho(
                    f"WARNING: Ollama >= {min_version} required for Anthropic API", fg="red"
                )
                click.echo("Update Ollama: ollama update")
                return False

        click.secho("Ollama setup complete!", fg="green")
        return True

    def get_claude_env(self) -> dict[str, str]:
        """Get environment variables for Claude Code."""
        return {
            "ANTHROPIC_BASE_URL": f"http://localhost:{self.ollama_port}",
            "ANTHROPIC_AUTH_TOKEN": "ollama",
            "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
        }

    def get_claude_model_arg(self, model: str) -> str:
        """Get the model ID to pass via claude --model."""
        return self.get_model_id(model)

    def is_healthy(self) -> bool:
        """Check if Ollama is healthy."""
        return self._is_ollama_running()

    def start(
        self,
        model: str | None = None,
        foreground: bool = False,  # noqa: ARG002
        **kwargs: object,  # noqa: ARG002
    ) -> dict[str, object]:
        """Start Ollama with native Anthropic Messages API.

        Args:
            model: Model to use
            foreground: Run in foreground with logs
        """
        if not self._supports_anthropic_api():
            version = self._get_ollama_version()
            version_str = ".".join(map(str, version)) if version else "unknown"
            min_version = ".".join(map(str, OLLAMA_MIN_ANTHROPIC_VERSION))
            raise RuntimeError(
                f"Ollama {version_str} does not support Anthropic API "
                f"(requires >= {min_version}). Update with: ollama update"
            )

        model = model or self.default_model
        model_id = self.get_model_id(model)

        click.echo(f"Using model: {model_id}")

        # Stop any tracked processes we started
        self._stop_tracked_processes()
        time.sleep(1)

        pids: dict[str, object] = {}

        # Ensure Ollama is running
        if not self._is_ollama_running():
            click.echo("Starting Ollama...")
            proc = subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            pids["ollama"] = proc.pid
            time.sleep(3)

        # Check/pull model
        click.echo(f"Checking for model: {model_id}")
        if not self._has_model(model_id):
            click.echo(f"Pulling model {model_id}...")
            subprocess.run(["ollama", "pull", model_id], check=True)

        click.echo()
        click.secho("=== Ollama Anthropic API Ready ===", fg="cyan", bold=True)
        click.echo(f"Anthropic Messages API: http://localhost:{self.ollama_port}/v1/messages")

        # Save PIDs
        self._save_pids(pids)

        # Write active state
        save_active_state(
            {
                "backend": "ollama",
                "model": model,
                "api_port": self.ollama_port,
            }
        )

        return {
            "pids": pids,
            "model": model_id,
            "api_port": self.ollama_port,
        }

    def stop(self, **kwargs: object) -> bool:  # noqa: ARG002
        """Stop Ollama if we started it."""
        stopped = self._stop_tracked_processes()
        clear_active_state()
        if not stopped:
            click.echo("Note: Ollama may still be running as a system service")
        return stopped

    def status(self) -> dict[str, object]:
        """Get Ollama backend status."""
        pids = self._load_pids()

        ollama_running = self._is_ollama_running()

        version = self._get_ollama_version()
        version_str = ".".join(map(str, version)) if version else "unknown"

        models: list[str] = []
        if ollama_running:
            try:
                result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
                if result.returncode == 0:
                    lines = result.stdout.strip().split("\n")[1:]  # Skip header
                    for line in lines:
                        if line.strip():
                            models.append(line.split()[0])
            except Exception:
                pass

        return {
            "backend": self.name,
            "ollama_version": version_str,
            "supports_anthropic": self._supports_anthropic_api(),
            "ollama_running": ollama_running,
            "pids": pids,
            "models": models,
            "api_port": self.ollama_port,
        }

    def _is_ollama_running(self) -> bool:
        """Check if Ollama server is running."""
        try:
            with httpx.Client(timeout=2) as client:
                resp = client.get(f"http://localhost:{self.ollama_port}/api/tags")
                return resp.status_code == 200
        except Exception:
            return False

    def _has_model(self, model_id: str) -> bool:
        """Check if model is available locally."""
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        model_base = model_id.split(":", maxsplit=1)[0]
        return model_base in result.stdout
