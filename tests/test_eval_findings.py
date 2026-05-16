"""Tests for autosre.eval.findings."""

from __future__ import annotations

import pytest

from autosre.eval.findings import (
    Finding,
    compute_id,
    normalize_findings,
)


class TestIdStability:
    def test_same_inputs_same_id(self) -> None:
        a = compute_id(
            suite="security",
            category="security",
            file="src/foo.py",
            line=42,
            title="SQL injection in user lookup",
        )
        b = compute_id(
            suite="security",
            category="security",
            file="src/foo.py",
            line=42,
            title="SQL injection in user lookup",
        )
        assert a == b

    def test_path_canonicalized_before_hashing(self) -> None:
        a = compute_id(
            suite="s",
            category="security",
            file="./src/foo.py",
            line=10,
            title="x",
        )
        b = compute_id(
            suite="s",
            category="security",
            file="src/foo.py",
            line=10,
            title="x",
        )
        assert a == b

    def test_title_token_reorder_matches(self) -> None:
        """Token-set equality ignores word order and casing."""
        a = compute_id(
            suite="s",
            category="security",
            file="f",
            line=1,
            title="missing csrf token login form",
        )
        b = compute_id(
            suite="s",
            category="security",
            file="f",
            line=1,
            title="login form missing CSRF token",
        )
        assert a == b

    def test_title_punctuation_variations_match(self) -> None:
        """Markdown-ish noise in titles doesn't perturb the id."""
        a = compute_id(
            suite="s",
            category="security",
            file="f",
            line=1,
            title="**SQL injection in users table**",
        )
        b = compute_id(
            suite="s",
            category="security",
            file="f",
            line=1,
            title="SQL injection in users table",
        )
        assert a == b

    def test_different_line_different_id(self) -> None:
        a = compute_id(suite="s", category="security", file="f", line=10, title="x")
        b = compute_id(suite="s", category="security", file="f", line=11, title="x")
        assert a != b

    def test_different_file_different_id(self) -> None:
        a = compute_id(suite="s", category="security", file="a.py", line=1, title="t")
        b = compute_id(suite="s", category="security", file="b.py", line=1, title="t")
        assert a != b


class TestFindingValidation:
    def test_valid_minimal(self) -> None:
        f = Finding(
            suite="security",
            category="security",
            file="src/x.py",
            title="Issue",
        )
        assert f.severity == "medium"
        assert f.confidence == 0.5

    def test_path_canonicalized(self) -> None:
        f = Finding(
            suite="security",
            category="security",
            file=".//src///x.py",
            title="Issue",
        )
        assert f.file == "src/x.py"

    def test_title_stripped_of_ansi_and_markdown(self) -> None:
        f = Finding(
            suite="security",
            category="security",
            file="x.py",
            title="\x1b[31m**Danger**\x1b[0m",
        )
        assert "*" not in f.title
        assert "\x1b" not in f.title
        assert "Danger" in f.title

    def test_bad_category_rejected(self) -> None:
        with pytest.raises(ValueError):
            Finding(
                suite="security",
                category="not-a-category",  # type: ignore[arg-type]
                file="x.py",
                title="t",
            )

    def test_confidence_bounds(self) -> None:
        with pytest.raises(ValueError):
            Finding(
                suite="s",
                category="security",
                file="x.py",
                title="t",
                confidence=1.5,
            )

    def test_with_id_populates_field(self) -> None:
        f = Finding(
            suite="security",
            category="security",
            file="x.py",
            line=10,
            title="Issue here",
        )
        g = f.with_id()
        assert g.id
        assert len(g.id) == 16


class TestNormalizeFindings:
    def test_drops_invalid_rows(self) -> None:
        raw = [
            {
                "category": "security",
                "file": "a.py",
                "title": "Good one",
                "severity": "high",
            },
            {"not": "a finding"},
            "garbage",
            {
                "category": "security",
                "file": "b.py",
                "title": "Another",
                "confidence": 0.9,
            },
        ]
        out = normalize_findings(
            raw,  # type: ignore[arg-type]
            suite="security",
            agent="injection",
            provider="local",
        )
        assert len(out) == 2
        assert all(f.id for f in out)
        assert all(f.provider == "local" for f in out)
        assert all(f.agent == "injection" for f in out)
        assert all(f.suite == "security" for f in out)

    def test_respects_provided_suite_agent_provider(self) -> None:
        raw = [
            {
                "category": "a11y",
                "file": "x.py",
                "title": "Low contrast",
                "severity": "low",
                "agent": "contrast",
                "provider": "anthropic",
                "suite": "a11y",
            }
        ]
        out = normalize_findings(
            raw,  # type: ignore[arg-type]
            suite="a11y",
            agent="default",
            provider="local",
        )
        assert len(out) == 1
        # Explicit row values win over runner-supplied defaults.
        assert out[0].agent == "contrast"
        assert out[0].provider == "anthropic"
