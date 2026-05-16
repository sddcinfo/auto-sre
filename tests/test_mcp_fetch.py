"""Tests for the local web fetch MCP server."""

from unittest.mock import MagicMock, patch

import pytest

from autosre.mcp_servers.fetch import _fetch_and_convert, web_fetch


class TestFetchAndConvert:
    """Tests for the HTML fetch and conversion logic."""

    def _mock_response(
        self, text: str, content_type: str = "text/html", status_code: int = 200
    ) -> MagicMock:
        resp = MagicMock()
        resp.text = text
        resp.status_code = status_code
        resp.headers = {"content-type": content_type}
        resp.raise_for_status = MagicMock()
        return resp

    @patch("autosre.mcp_servers.fetch.curl_requests.get")
    def test_html_converts_to_markdown(self, mock_get: MagicMock) -> None:
        """Test basic HTML to markdown conversion."""
        html = "<html><body><h1>Hello</h1><p>World</p></body></html>"
        mock_get.return_value = self._mock_response(html)

        result = _fetch_and_convert("https://example.com")
        assert "Hello" in result
        assert "World" in result

    @patch("autosre.mcp_servers.fetch.curl_requests.get")
    def test_strips_script_and_style(self, mock_get: MagicMock) -> None:
        """Test that script and style tags are stripped."""
        html = (
            "<html><body>"
            "<script>alert('xss')</script>"
            "<style>.foo{color:red}</style>"
            "<nav>nav content</nav>"
            "<p>Keep this</p>"
            "</body></html>"
        )
        mock_get.return_value = self._mock_response(html)

        result = _fetch_and_convert("https://example.com")
        assert "alert" not in result
        assert "color:red" not in result
        assert "nav content" not in result
        assert "Keep this" in result

    @patch("autosre.mcp_servers.fetch.curl_requests.get")
    def test_truncates_long_content(self, mock_get: MagicMock) -> None:
        """Test content truncation at MAX_LENGTH."""
        html = f"<html><body><p>{'x' * 200000}</p></body></html>"
        mock_get.return_value = self._mock_response(html)

        with patch("autosre.mcp_servers.fetch.MAX_LENGTH", 1000):
            result = _fetch_and_convert("https://example.com")
            assert len(result) <= 1000 + len("\n\n[Content truncated]")
            assert "[Content truncated]" in result

    @patch("autosre.mcp_servers.fetch.curl_requests.get")
    def test_non_html_returns_raw_text(self, mock_get: MagicMock) -> None:
        """Test that non-HTML content is returned as raw text."""
        mock_get.return_value = self._mock_response("plain text content", content_type="text/plain")

        result = _fetch_and_convert("https://example.com/file.txt")
        assert result == "plain text content"

    @patch("autosre.mcp_servers.fetch.curl_requests.get")
    def test_collapses_blank_lines(self, mock_get: MagicMock) -> None:
        """Test that excessive blank lines are collapsed."""
        html = "<html><body><p>A</p><br><br><br><br><p>B</p></body></html>"
        mock_get.return_value = self._mock_response(html)

        result = _fetch_and_convert("https://example.com")
        assert "\n\n\n" not in result

    @patch("autosre.mcp_servers.fetch.curl_requests.get")
    def test_uses_browser_impersonation(self, mock_get: MagicMock) -> None:
        """Test that curl_cffi is called with browser impersonation."""
        mock_get.return_value = self._mock_response("<html><body>ok</body></html>")

        _fetch_and_convert("https://example.com")
        mock_get.assert_called_once_with(
            "https://example.com",
            impersonate="chrome131",
            timeout=30,
            allow_redirects=True,
        )


class TestWebFetchTool:
    """Tests for the web_fetch MCP tool error handling."""

    @patch("autosre.mcp_servers.fetch._fetch_and_convert")
    def test_returns_content_on_success(self, mock_fetch: MagicMock) -> None:
        """Test successful fetch returns content."""
        mock_fetch.return_value = "# Hello World"
        result = web_fetch("https://example.com")
        assert result == "# Hello World"

    @patch("autosre.mcp_servers.fetch._fetch_and_convert")
    def test_handles_timeout(self, mock_fetch: MagicMock) -> None:
        """Test timeout error handling."""
        from curl_cffi.requests.errors import RequestsError

        mock_fetch.side_effect = RequestsError("Connection timeout")
        result = web_fetch("https://slow.example.com")
        assert "timed out" in result.lower()
        assert "slow.example.com" in result

    @patch("autosre.mcp_servers.fetch._fetch_and_convert")
    def test_handles_connection_error(self, mock_fetch: MagicMock) -> None:
        """Test connection error handling."""
        from curl_cffi.requests.errors import RequestsError

        mock_fetch.side_effect = RequestsError("Could not resolve host or connect")
        result = web_fetch("https://down.example.com")
        assert "connect" in result.lower()
        assert "down.example.com" in result

    @patch("autosre.mcp_servers.fetch._fetch_and_convert")
    def test_handles_generic_error(self, mock_fetch: MagicMock) -> None:
        """Test generic exception handling."""
        mock_fetch.side_effect = RuntimeError("unexpected error")
        result = web_fetch("https://example.com")
        assert "Error" in result
        assert "example.com" in result

    def test_max_length_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test AUTOSRE_FETCH_MAX_LENGTH environment variable."""
        monkeypatch.setenv("AUTOSRE_FETCH_MAX_LENGTH", "500")
        import importlib

        import autosre.mcp_servers.fetch as fetch_module

        importlib.reload(fetch_module)
        assert fetch_module.MAX_LENGTH == 500
        # Restore
        monkeypatch.delenv("AUTOSRE_FETCH_MAX_LENGTH", raising=False)
        importlib.reload(fetch_module)
