"""Tests for the local web search MCP server."""

from unittest.mock import MagicMock, patch

import pytest

from autosre.mcp_servers.search import _search, web_search


class TestSearchLogic:
    """Tests for the DuckDuckGo search logic."""

    @patch("autosre.mcp_servers.search.DDGS")
    def test_returns_formatted_results(self, mock_ddgs_cls: MagicMock) -> None:
        """Test search results are formatted as markdown."""
        mock_ddgs = MagicMock()
        mock_ddgs.text.return_value = [
            {"title": "Result One", "href": "https://one.com", "body": "First result"},
            {"title": "Result Two", "href": "https://two.com", "body": "Second result"},
        ]
        mock_ddgs_cls.return_value = mock_ddgs

        result = _search("test query", max_results=5)
        assert "### [Result One](https://one.com)" in result
        assert "First result" in result
        assert "### [Result Two](https://two.com)" in result
        assert "Second result" in result
        assert "---" in result

    @patch("autosre.mcp_servers.search.DDGS")
    def test_respects_max_results(self, mock_ddgs_cls: MagicMock) -> None:
        """Test max_results parameter is passed through."""
        mock_ddgs = MagicMock()
        mock_ddgs.text.return_value = []
        mock_ddgs_cls.return_value = mock_ddgs

        _search("test", max_results=3)
        mock_ddgs.text.assert_called_once_with("test", max_results=3)

    @patch("autosre.mcp_servers.search.DDGS")
    def test_caps_at_max_results_env(self, mock_ddgs_cls: MagicMock) -> None:
        """Test that results are capped at MAX_RESULTS."""
        mock_ddgs = MagicMock()
        mock_ddgs.text.return_value = []
        mock_ddgs_cls.return_value = mock_ddgs

        with patch("autosre.mcp_servers.search.MAX_RESULTS", 5):
            _search("test", max_results=20)
            mock_ddgs.text.assert_called_once_with("test", max_results=5)

    @patch("autosre.mcp_servers.search.DDGS")
    def test_no_results(self, mock_ddgs_cls: MagicMock) -> None:
        """Test handling of empty results."""
        mock_ddgs = MagicMock()
        mock_ddgs.text.return_value = []
        mock_ddgs_cls.return_value = mock_ddgs

        result = _search("obscure query", max_results=5)
        assert "No results found" in result

    @patch("autosre.mcp_servers.search.DDGS")
    def test_single_result(self, mock_ddgs_cls: MagicMock) -> None:
        """Test formatting with a single result (no separator)."""
        mock_ddgs = MagicMock()
        mock_ddgs.text.return_value = [
            {"title": "Only One", "href": "https://one.com", "body": "Solo result"},
        ]
        mock_ddgs_cls.return_value = mock_ddgs

        result = _search("query", max_results=5)
        assert "### [Only One](https://one.com)" in result
        assert "---" not in result


class TestWebSearchTool:
    """Tests for the web_search MCP tool."""

    @patch("autosre.mcp_servers.search._search")
    def test_returns_results_on_success(self, mock_search: MagicMock) -> None:
        """Test successful search returns results."""
        mock_search.return_value = "### [Title](https://example.com)\n\nbody"
        result = web_search("test query")
        assert "Title" in result

    @patch("autosre.mcp_servers.search._search")
    def test_handles_exception(self, mock_search: MagicMock) -> None:
        """Test exception handling returns error string."""
        mock_search.side_effect = RuntimeError("rate limited")
        result = web_search("test query")
        assert "Error" in result
        assert "test query" in result

    @patch("autosre.mcp_servers.search._search")
    def test_default_max_results(self, mock_search: MagicMock) -> None:
        """Test default max_results is 5."""
        mock_search.return_value = "results"
        web_search("test")
        mock_search.assert_called_once_with("test", 5)

    def test_max_results_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test AUTOSRE_SEARCH_MAX_RESULTS environment variable."""
        monkeypatch.setenv("AUTOSRE_SEARCH_MAX_RESULTS", "3")
        import importlib

        import autosre.mcp_servers.search as search_module

        importlib.reload(search_module)
        assert search_module.MAX_RESULTS == 3
        monkeypatch.delenv("AUTOSRE_SEARCH_MAX_RESULTS", raising=False)
        importlib.reload(search_module)
