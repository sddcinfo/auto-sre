"""Tests for autosre.eval.differ — matching, buckets, refusal rules."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autosre.eval.differ import (
    CompareOverrides,
    CompareRefusedError,
    RunManifest,
    _dedupe,
    check_comparability,
    compare,
    diff_suite,
    load_findings,
    similarity,
    write_compare_dir,
)
from autosre.eval.findings import Finding
from autosre.eval.judge import JudgeResult, StubJudge


def _f(
    title: str,
    *,
    file: str = "src/app.py",
    line: int | None = 10,
    category: str = "security",
    severity: str = "medium",
    description: str = "",
    suite: str = "security",
    provider: str = "local",
) -> Finding:
    return Finding(
        suite=suite,
        category=category,  # type: ignore[arg-type]
        severity=severity,  # type: ignore[arg-type]
        file=file,
        line=line,
        title=title,
        description=description,
        provider=provider,  # type: ignore[arg-type]
    ).with_id()


class TestSimilarity:
    def test_same_id_perfect(self) -> None:
        a = _f("SQL injection")
        b = _f("SQL injection")
        s = similarity(a, b)
        assert s >= 0.85

    def test_different_file_zero(self) -> None:
        a = _f("Issue", file="a.py")
        b = _f("Issue", file="b.py")
        assert similarity(a, b) == 0.0

    def test_adjacent_category_nonzero(self) -> None:
        a = _f("Hardcoded API key", category="security")
        b = _f("Hardcoded API key", category="leakage")
        s = similarity(a, b)
        assert 0.0 < s < 1.0

    def test_unrelated_category_low(self) -> None:
        a = _f("Dead function", category="dead-code")
        b = _f("Dead function", category="quality")
        s = similarity(a, b)
        # Same file + same title gives some score, but category=0 caps it.
        assert s < 0.85

    def test_line_distance_softens(self) -> None:
        a = _f("Issue", line=10)
        b = _f("Issue", line=14)
        c = _f("Issue", line=50)
        assert similarity(a, b) > similarity(a, c)


class TestDedupe:
    def test_exact_duplicate_merged(self) -> None:
        f1 = _f("Hardcoded API key", line=10, description="foo")
        f2 = _f("Hardcoded API key", line=10, description="bar")
        out = _dedupe([f1, f2])
        assert len(out) == 1
        assert "foo" in out[0].evidence or out[0].evidence == ""

    def test_distinct_findings_not_merged(self) -> None:
        f1 = _f("SQL injection", line=10)
        f2 = _f("Command injection", line=100)
        out = _dedupe([f1, f2])
        assert len(out) == 2


class TestDiffSuite:
    def test_conservation_of_mass_exact_match(self) -> None:
        a = [_f("SQL injection in login", line=10), _f("Unvalidated redirect", line=50)]
        b = [
            _f("SQL injection in login", line=10),
            _f("Unvalidated redirect", line=50),
            _f("Weak RNG in token", line=200),
        ]
        d = diff_suite(a, b, "security")
        d.check_conservation()
        assert len(d.both) == 2
        assert len(d.a_only) == 0
        assert len(d.b_only) == 1

    def test_conservation_with_no_overlap(self) -> None:
        a = [_f("A", line=1, file="a.py")]
        b = [_f("B", line=1, file="b.py")]
        d = diff_suite(a, b, "security")
        d.check_conservation()
        assert len(d.both) == 0
        assert len(d.a_only) == 1
        assert len(d.b_only) == 1

    def test_one_to_one_even_with_overlapping_candidates(self) -> None:
        """A single B must not match two distinct As — no double-counting.

        Two A findings at nearby lines with different titles (so id-dedup
        cannot collapse them) must both present as candidates for the
        single B, but only one wins assignment.
        """
        a = [
            _f("SQL injection in login", line=10),
            _f("Command injection in shell", line=12),
        ]
        b = [_f("SQL injection in login", line=11)]
        d = diff_suite(a, b, "security")
        d.check_conservation()
        total_matched = len(d.both) + len(d.partial)
        assert total_matched == 1
        assert len(d.a_only) == 1
        assert len(d.b_only) == 0

    def test_deterministic_output(self) -> None:
        a = [_f("alpha", line=1), _f("beta", line=5)]
        b = [_f("alpha", line=1), _f("beta", line=5)]
        d1 = diff_suite(a, b, "security")
        d2 = diff_suite(a, b, "security")
        assert [(p.a_idx, p.b_idx) for p in d1.both] == [(p.a_idx, p.b_idx) for p in d2.both]

    def test_stub_judge_can_upgrade_partial_to_both(self) -> None:
        a = [
            _f(
                "Possible injection risk",
                line=10,
                description="user input reaches sql",
            )
        ]
        b = [
            _f(
                "SQL injection user input",
                line=12,
                description="input concatenated into sql query",
            )
        ]

        def always_yes(fa: Finding, fb: Finding) -> JudgeResult:
            return JudgeResult(same="yes", confidence=0.9, rationale="stub")

        judge = StubJudge(verdict_fn=always_yes)
        d = diff_suite(a, b, "security", judge=judge)
        d.check_conservation()
        assert len(d.both) == 1
        assert judge.calls >= 1


class TestRefusalRules:
    def _m(self, **kw: object) -> RunManifest:
        defaults = {
            "run_id": "r",
            "provider": "local",
            "target_repo": "/repo",
            "target_sha": "sha1",
            "snapshot_digest": "d1",
            "suites": ["security"],
            "run_dir": Path("/tmp"),
            "model": "m",
        }
        defaults.update(kw)
        return RunManifest(**defaults)  # type: ignore[arg-type]

    def test_different_repos_always_refused(self) -> None:
        a = self._m(target_repo="/repo-a")
        b = self._m(target_repo="/repo-b", provider="anthropic")
        with pytest.raises(CompareRefusedError):
            check_comparability(a, b, overrides=CompareOverrides(allow_sha_mismatch=True))

    def test_sha_mismatch_refused_by_default(self) -> None:
        a = self._m(target_sha="sha1", snapshot_digest="d1")
        b = self._m(
            target_sha="sha2",
            snapshot_digest="d2",
            provider="anthropic",
        )
        with pytest.raises(CompareRefusedError):
            check_comparability(a, b)

    def test_sha_mismatch_allowed_with_override(self) -> None:
        a = self._m(target_sha="sha1", snapshot_digest="d1")
        b = self._m(
            target_sha="sha2",
            snapshot_digest="d2",
            provider="anthropic",
        )
        warnings = check_comparability(a, b, overrides=CompareOverrides(allow_sha_mismatch=True))
        assert any("SHA" in w for w in warnings)

    def test_digest_match_bypasses_sha_rule(self) -> None:
        a = self._m(target_sha=None, snapshot_digest="d1")
        b = self._m(
            target_sha=None,
            snapshot_digest="d1",
            provider="anthropic",
        )
        assert check_comparability(a, b) == []

    def test_same_provider_refused(self) -> None:
        a = self._m(provider="local")
        b = self._m(provider="local")
        with pytest.raises(CompareRefusedError):
            check_comparability(a, b)

    def test_suite_mismatch_refused(self) -> None:
        a = self._m(suites=["security"])
        b = self._m(suites=["leakage"], provider="anthropic")
        with pytest.raises(CompareRefusedError):
            check_comparability(a, b)


class TestCompareEndToEnd:
    def _write_run(
        self,
        root: Path,
        *,
        run_id: str,
        provider: str,
        sha: str = "shaX",
        findings: dict[str, list[Finding]] | None = None,
    ) -> Path:
        run_dir = root / run_id
        run_dir.mkdir()
        manifest = {
            "run_id": run_id,
            "provider": provider,
            "target_repo": "/fake/repo",
            "target_sha": sha,
            "snapshot_digest": None,
            "suites": sorted((findings or {}).keys()),
            "model": "m",
        }
        (run_dir / "manifest.json").write_text(json.dumps(manifest))
        for suite, flist in (findings or {}).items():
            suite_dir = run_dir / "suites" / suite
            suite_dir.mkdir(parents=True)
            with (suite_dir / "findings.jsonl").open("w") as f:
                for finding in flist:
                    f.write(finding.model_dump_json() + "\n")
            (suite_dir / "parse_report.json").write_text(json.dumps({"suite": suite, "agents": []}))
        return run_dir

    def test_end_to_end_compare(self, tmp_path: Path) -> None:
        a_dir = self._write_run(
            tmp_path,
            run_id="local-run",
            provider="local",
            findings={
                "security": [
                    _f(
                        "SQL injection in login",
                        line=10,
                        description="user input flows into a concatenated sql query",
                    ),
                    _f(
                        "Weak RNG in token generator",
                        line=99,
                        file="src/token.py",
                        description="uses random.random for session ids",
                    ),
                ]
            },
        )
        b_dir = self._write_run(
            tmp_path,
            run_id="anthropic-run",
            provider="anthropic",
            findings={
                "security": [
                    _f(
                        "SQL injection in login",
                        line=10,
                        description="user input flows into a concatenated sql query",
                    ),
                    _f(
                        "Missing CSRF token in settings form",
                        line=200,
                        file="src/settings.py",
                        description="no csrf check on POST /settings",
                    ),
                ]
            },
        )
        result = compare(a_dir, b_dir)
        d = result.per_suite["security"]
        d.check_conservation()
        assert len(d.both) == 1
        assert len(d.a_only) == 1
        assert len(d.b_only) == 1
        payload = result.to_json()
        assert payload["valid_as_model_quality"] is True
        per_suite = payload["per_suite"]
        assert isinstance(per_suite, dict)
        assert per_suite["security"]["both"] == 1

    def test_write_compare_dir_and_index(self, tmp_path: Path) -> None:
        a_dir = self._write_run(
            tmp_path,
            run_id="local-run",
            provider="local",
            findings={"security": [_f("SQL injection", line=10)]},
        )
        b_dir = self._write_run(
            tmp_path,
            run_id="anthropic-run",
            provider="anthropic",
            findings={"security": [_f("SQL injection", line=10)]},
        )
        result = compare(a_dir, b_dir)
        out_dir = tmp_path / "compares" / "c1"
        index = tmp_path / "compares.jsonl"
        write_compare_dir(result, out_dir, index_path=index)
        assert (out_dir / "compare.json").exists()
        assert index.exists()
        rows = [json.loads(line) for line in index.read_text().splitlines() if line.strip()]
        assert rows and rows[0]["run_a"] == "local-run"


class TestLoadFindings:
    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        assert load_findings(tmp_path, "security") == []
