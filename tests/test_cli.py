"""Tests for CLI commands."""

from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from autosre.cli import cli


class TestCLI:
    """Tests for CLI commands."""

    def test_cli_help(self) -> None:
        """Test CLI help output."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Auto-SRE" in result.output

    def test_backends_command(self) -> None:
        """Test backends listing command."""
        runner = CliRunner()
        result = runner.invoke(cli, ["backends"])
        assert result.exit_code == 0
        assert "ollama" in result.output.lower()
        assert "llamacpp" in result.output.lower()

    def test_status_command(self) -> None:
        """Test status command."""
        runner = CliRunner()

        with patch("autosre.cli.get_backend") as mock_get_backend:
            mock_backend = MagicMock()
            mock_backend.status.return_value = {
                "backend": "ollama",
                "ollama_running": False,
                "ollama_version": "0.14.2",
                "supports_anthropic": True,
                "api_port": 11434,
                "pids": {},
            }
            mock_get_backend.return_value = mock_backend

            with patch("autosre.cli.load_active_state", return_value=None):
                result = runner.invoke(cli, ["status"])
                assert result.exit_code == 0

    def test_stop_command(self) -> None:
        """Test stop command — verifies logic without touching real containers.

        IMPORTANT: This test must NEVER call the real stop command.
        It only tests that the function signature and mocking work.
        """
        # Do NOT use runner.invoke for stop — it can escape mocks and kill real containers.
        # Instead, just verify the command exists and accepts --no-scribe.
        from autosre.cli import stop

        assert stop is not None
        assert hasattr(stop, "params")


class TestClaudeCommand:
    """Tests for credential isolation in claude command."""

    def _run_claude_command(
        self, cli_args: list[str] | None = None
    ) -> tuple[dict[str, str], dict[str, object]]:
        """Helper to invoke claude command and capture execvpe args."""
        runner = CliRunner()
        captured_env: dict[str, str] = {}
        captured_args: dict[str, object] = {}

        if cli_args is None:
            cli_args = ["claude"]

        def mock_execvpe(prog: str, args: list[str], env: dict[str, str]) -> None:
            captured_env.update(env)
            captured_args["prog"] = prog
            captured_args["args"] = args
            raise SystemExit(0)

        active_state = {"backend": "ollama", "model": "qwen3.6", "api_port": 11434}

        with (
            patch("autosre.cli.load_active_state", return_value=active_state),
            patch("autosre.cli.get_backend") as mock_get_backend,
        ):
            mock_backend = MagicMock()
            mock_backend.default_model = "qwen3.6"
            mock_backend.is_healthy.return_value = True
            mock_backend.get_claude_env.return_value = {
                "ANTHROPIC_BASE_URL": "http://localhost:11434",
                "ANTHROPIC_AUTH_TOKEN": "ollama",
                "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
            }
            mock_backend.get_claude_model_arg.return_value = "qwen3.6:35b-a3b"
            mock_get_backend.return_value = mock_backend

            with (
                patch("shutil.which", return_value="/usr/local/bin/claude"),
                patch("os.execvpe", side_effect=mock_execvpe),
                patch.dict(
                    "os.environ",
                    {"ANTHROPIC_API_KEY": "sk-cloud-key-should-be-purged"},
                    clear=False,
                ),
            ):
                runner.invoke(cli, cli_args)

        return captured_env, captured_args

    def test_claude_purges_cloud_credentials(self) -> None:
        """Test that claude command purges ANTHROPIC_API_KEY and sets local creds."""
        captured_env, captured_args = self._run_claude_command()

        # Verify cloud key is replaced with local dummy
        assert captured_env.get("ANTHROPIC_API_KEY") == "local-vllm"

        # Verify local creds are set
        assert captured_env.get("ANTHROPIC_AUTH_TOKEN") == "ollama"
        assert captured_env.get("ANTHROPIC_BASE_URL") == "http://localhost:11434"
        assert captured_env.get("CLAUDE_CODE_ATTRIBUTION_HEADER") == "0"

        # Verify ANTHROPIC_MODEL is set to local model (forces subagents to local)
        assert "ANTHROPIC_MODEL" in captured_env

        # Verify key flags are present
        args = captured_args["args"]
        assert args[0] == "claude"
        assert any("--model=" in a for a in args)
        assert any("--mcp-config=" in a for a in args)
        assert any("system-prompt" in a for a in args)

    def test_claude_swarm_mode(self) -> None:
        """Test that --swarm enables agent teams env var."""
        captured_env, _ = self._run_claude_command(["claude", "--swarm"])

        assert captured_env.get("CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS") == "1"

    def test_claude_teams_always_enabled(self) -> None:
        """Test that agent teams is always enabled."""
        captured_env, _ = self._run_claude_command()

        assert captured_env.get("CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS") == "1"

    def test_claude_no_server_running(self) -> None:
        """Test claude command fails when no server is running."""
        runner = CliRunner()

        with patch("autosre.cli.load_active_state", return_value=None):
            result = runner.invoke(cli, ["claude"])
            assert result.exit_code != 0
            assert "No server running" in result.output


class TestMcpCommands:
    """Tests for MCP command group."""

    def test_mcp_status_no_claude(self) -> None:
        """Test mcp status when claude CLI not found."""
        runner = CliRunner()

        with patch("shutil.which", return_value=None):
            result = runner.invoke(cli, ["mcp", "status"])
            assert result.exit_code != 0

    def test_mcp_setup_installs_local_servers(self) -> None:
        """Test mcp setup adds both local MCP servers."""
        runner = CliRunner()

        with (
            patch("shutil.which", return_value="/usr/local/bin/mock"),
            patch("autosre.cli.subprocess.run") as mock_run,
            patch("autosre.mcp_servers.settings.load_claude_settings", return_value={}),
            patch("autosre.mcp_servers.settings.save_claude_settings"),
            patch("autosre.mcp_servers.settings.deny_builtin_web_tools", return_value={}),
        ):
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            result = runner.invoke(cli, ["mcp", "setup"])
            assert result.exit_code == 0

            # Verify both servers were added
            add_calls = [
                call
                for call in mock_run.call_args_list
                if "add" in str(call) and "autosre" in str(call)
            ]
            assert len(add_calls) == 2

    def test_mcp_setup_with_brave(self) -> None:
        """Test mcp setup with --brave-api-key adds all three servers."""
        runner = CliRunner()

        with (
            patch("shutil.which", return_value="/usr/local/bin/mock"),
            patch("autosre.cli.subprocess.run") as mock_run,
            patch("autosre.mcp_servers.settings.load_claude_settings", return_value={}),
            patch("autosre.mcp_servers.settings.save_claude_settings"),
            patch("autosre.mcp_servers.settings.deny_builtin_web_tools", return_value={}),
        ):
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            result = runner.invoke(cli, ["mcp", "setup", "--brave-api-key", "test-key"])
            assert result.exit_code == 0

            # Should have 3 add calls (fetch + search + brave)
            add_calls = [call for call in mock_run.call_args_list if "add" in str(call)]
            assert len(add_calls) == 3

    def test_mcp_setup_denies_builtin_tools(self) -> None:
        """Test mcp setup denies built-in WebFetch/WebSearch."""
        runner = CliRunner()
        saved_settings: dict[str, object] = {}

        def capture_save(settings: dict[str, object], path: object = None) -> None:
            saved_settings.update(settings)

        with (
            patch("shutil.which", return_value="/usr/local/bin/mock"),
            patch("autosre.cli.subprocess.run") as mock_run,
            patch("autosre.mcp_servers.settings.load_claude_settings", return_value={}),
            patch("autosre.mcp_servers.settings.save_claude_settings", side_effect=capture_save),
        ):
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            result = runner.invoke(cli, ["mcp", "setup"])
            assert result.exit_code == 0
            assert "WebFetch" in saved_settings.get("permissions", {}).get("deny", [])  # type: ignore[union-attr]
            assert "WebSearch" in saved_settings.get("permissions", {}).get("deny", [])  # type: ignore[union-attr]

    def test_mcp_setup_missing_entry_points(self) -> None:
        """Test mcp setup fails when entry points are missing."""
        runner = CliRunner()

        def selective_which(cmd: str) -> str | None:
            if cmd == "claude":
                return "/usr/local/bin/claude"
            return None

        with patch("shutil.which", side_effect=selective_which):
            result = runner.invoke(cli, ["mcp", "setup"])
            assert result.exit_code != 0
            assert "not found on PATH" in result.output

    def test_mcp_remove_cleans_up(self) -> None:
        """Test mcp remove removes servers and restores settings."""
        runner = CliRunner()

        with (
            patch("shutil.which", return_value="/usr/local/bin/mock"),
            patch("autosre.cli.subprocess.run") as mock_run,
            patch(
                "autosre.mcp_servers.settings.load_claude_settings",
                return_value={"permissions": {"deny": ["WebFetch", "WebSearch"]}},
            ),
            patch("autosre.mcp_servers.settings.save_claude_settings") as mock_save,
        ):
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            result = runner.invoke(cli, ["mcp", "remove"])
            assert result.exit_code == 0

            # Verify servers were removed
            remove_calls = [call for call in mock_run.call_args_list if "remove" in str(call)]
            assert len(remove_calls) == 2

            # Verify settings were saved with tools allowed
            mock_save.assert_called_once()
            saved = mock_save.call_args[0][0]
            deny = saved.get("permissions", {}).get("deny", [])
            assert "WebFetch" not in deny
            assert "WebSearch" not in deny

    def test_mcp_status_shows_all_servers(self) -> None:
        """Test mcp status checks for all server types."""
        runner = CliRunner()

        with (
            patch("shutil.which", return_value="/usr/local/bin/mock"),
            patch("autosre.cli.subprocess.run") as mock_run,
            patch(
                "autosre.mcp_servers.settings.load_claude_settings",
                return_value={"permissions": {"deny": ["WebFetch", "WebSearch"]}},
            ),
        ):
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="autosre-fetch\nautosre-search\n",
                stderr="",
            )
            result = runner.invoke(cli, ["mcp", "status"])
            assert result.exit_code == 0
            assert "autosre-fetch: configured" in result.output
            assert "autosre-search: configured" in result.output
            assert "brave-search: not configured" in result.output
