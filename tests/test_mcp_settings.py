"""Tests for Claude Code settings management."""

import json
from pathlib import Path

from autosre.mcp_servers.settings import (
    allow_builtin_web_tools,
    deny_builtin_web_tools,
    load_claude_settings,
    save_claude_settings,
)


class TestDenyBuiltinWebTools:
    """Tests for denying/allowing built-in web tools."""

    def test_creates_deny_list_from_empty(self) -> None:
        """Test deny creates permissions.deny from empty settings."""
        settings = deny_builtin_web_tools({})
        assert settings["permissions"]["deny"] == ["WebFetch", "WebSearch"]  # type: ignore[index]

    def test_idempotent(self) -> None:
        """Test calling deny twice doesn't duplicate entries."""
        settings = deny_builtin_web_tools({})
        settings = deny_builtin_web_tools(settings)
        deny = settings["permissions"]["deny"]  # type: ignore[index]
        assert deny.count("WebFetch") == 1
        assert deny.count("WebSearch") == 1

    def test_preserves_existing_deny_entries(self) -> None:
        """Test that existing deny entries are preserved."""
        settings: dict[str, object] = {"permissions": {"deny": ["SomeOtherTool"]}}
        settings = deny_builtin_web_tools(settings)
        deny = settings["permissions"]["deny"]  # type: ignore[index]
        assert "SomeOtherTool" in deny
        assert "WebFetch" in deny
        assert "WebSearch" in deny

    def test_preserves_other_settings(self) -> None:
        """Test that other settings keys are not modified."""
        settings: dict[str, object] = {"theme": "dark", "permissions": {}}
        settings = deny_builtin_web_tools(settings)
        assert settings["theme"] == "dark"


class TestAllowBuiltinWebTools:
    """Tests for removing web tools from deny list."""

    def test_removes_web_tools(self) -> None:
        """Test allow removes WebFetch and WebSearch from deny."""
        settings: dict[str, object] = {
            "permissions": {"deny": ["WebFetch", "WebSearch", "OtherTool"]}
        }
        settings = allow_builtin_web_tools(settings)
        deny = settings["permissions"]["deny"]  # type: ignore[index]
        assert "WebFetch" not in deny
        assert "WebSearch" not in deny
        assert "OtherTool" in deny

    def test_no_permissions_is_noop(self) -> None:
        """Test allow on settings with no permissions is a no-op."""
        settings: dict[str, object] = {"theme": "dark"}
        result = allow_builtin_web_tools(settings)
        assert result == {"theme": "dark"}

    def test_no_deny_list_is_noop(self) -> None:
        """Test allow on settings with no deny list is a no-op."""
        settings: dict[str, object] = {"permissions": {"allow": ["Read"]}}
        result = allow_builtin_web_tools(settings)
        assert result == {"permissions": {"allow": ["Read"]}}


class TestLoadSaveSettings:
    """Tests for loading and saving settings files."""

    def test_load_missing_file(self, tmp_path: Path) -> None:
        """Test loading from a non-existent file returns empty dict."""
        result = load_claude_settings(tmp_path / "nonexistent.json")
        assert result == {}

    def test_load_empty_file(self, tmp_path: Path) -> None:
        """Test loading an empty file returns empty dict."""
        f = tmp_path / "settings.json"
        f.write_text("")
        result = load_claude_settings(f)
        assert result == {}

    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        """Test that save followed by load preserves content."""
        f = tmp_path / "settings.json"
        settings: dict[str, object] = {
            "permissions": {"deny": ["WebFetch", "WebSearch"]},
            "theme": "dark",
        }
        save_claude_settings(settings, f)
        loaded = load_claude_settings(f)
        assert loaded == settings

    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        """Test that save creates parent directories."""
        f = tmp_path / "nested" / "dir" / "settings.json"
        save_claude_settings({"key": "value"}, f)
        assert f.exists()
        content = json.loads(f.read_text())
        assert content == {"key": "value"}
