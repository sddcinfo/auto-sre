"""Tests for autosre.eval.lenient_json."""

from __future__ import annotations

import json

import pytest

from autosre.eval.lenient_json import clean, loads, try_loads


class TestClean:
    def test_strips_markdown_fences(self) -> None:
        src = '```json\n{"a": 1}\n```'
        assert clean(src) == '{"a": 1}'

    def test_strips_jsonc_fence(self) -> None:
        src = '```jsonc\n{"a": 1}\n```'
        assert clean(src) == '{"a": 1}'

    def test_strips_line_comments(self) -> None:
        src = '{"a": 1, // a comment\n "b": 2}'
        out = clean(src)
        assert "//" not in out
        assert json.loads(out) == {"a": 1, "b": 2}

    def test_strips_block_comments(self) -> None:
        src = '{"a": 1 /* inline */, "b": 2}'
        out = clean(src)
        assert "/*" not in out
        assert json.loads(out) == {"a": 1, "b": 2}

    def test_strips_trailing_commas(self) -> None:
        src = '{"a": [1, 2, 3,], "b": 2,}'
        out = clean(src)
        assert json.loads(out) == {"a": [1, 2, 3], "b": 2}

    def test_preserves_urls_with_double_slash(self) -> None:
        """Line comment regex must not eat ``https://`` in values."""
        src = '{"url": "https://example.com/path"}'
        out = clean(src)
        assert json.loads(out) == {"url": "https://example.com/path"}


class TestLoads:
    def test_loads_roundtrip(self) -> None:
        assert loads('{"x": 1}') == {"x": 1}

    def test_loads_raises_on_still_bad(self) -> None:
        with pytest.raises(json.JSONDecodeError):
            loads("not json at all {")

    def test_try_loads_returns_none(self) -> None:
        assert try_loads("not json") is None

    def test_try_loads_returns_value(self) -> None:
        assert try_loads('{"a": [1, 2,]}') == {"a": [1, 2]}
