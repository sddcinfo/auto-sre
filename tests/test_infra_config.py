"""Tests for autosre.infra.config module."""

from pathlib import Path

import pytest

from autosre.infra.config import load_yaml, save_yaml


class TestDataPath:
    def test_returns_path_in_data_dir(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
        # Re-import to pick up new env var
        from autosre.infra import config

        monkeypatch.setattr(config, "DATA_DIR", tmp_path / "autosre")
        result = config.data_path("test.yaml")
        assert result == tmp_path / "autosre" / "test.yaml"
        assert result.parent.exists()


class TestLoadYaml:
    def test_load_existing_file(self, tmp_path: Path) -> None:
        p = tmp_path / "config.yaml"
        p.write_text("key: value\nnested:\n  a: 1\n")
        result = load_yaml(p)
        assert result == {"key": "value", "nested": {"a": 1}}

    def test_load_nonexistent_file(self, tmp_path: Path) -> None:
        result = load_yaml(tmp_path / "missing.yaml")
        assert result == {}

    def test_load_empty_file(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.yaml"
        p.write_text("")
        result = load_yaml(p)
        assert result == {}

    def test_load_non_dict(self, tmp_path: Path) -> None:
        p = tmp_path / "list.yaml"
        p.write_text("- item1\n- item2\n")
        result = load_yaml(p)
        assert result == {}


class TestSaveYaml:
    def test_save_and_load(self, tmp_path: Path) -> None:
        p = tmp_path / "output.yaml"
        data = {"nodes": [{"hostname": "gb10-1", "ip": "192.168.1.101"}]}
        save_yaml(p, data)
        assert p.exists()
        loaded = load_yaml(p)
        assert loaded == data

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        p = tmp_path / "deep" / "nested" / "config.yaml"
        save_yaml(p, {"key": "value"})
        assert p.exists()
        assert load_yaml(p) == {"key": "value"}
