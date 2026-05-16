"""Pytest configuration and fixtures."""

import pytest


@pytest.fixture
def mock_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set up test environment variables."""
    monkeypatch.setenv("XDG_DATA_HOME", "/tmp/autosre-test")
    monkeypatch.setenv("HF_TOKEN", "test-token")
