"""Bidirectional diff between two eval runs.

Given two independent runs that agree on target SHA, snapshot digest,
and suite set, produce a per-suite bucket breakdown showing which
findings both runs caught, which each caught uniquely, and which are
partial matches. Strict refusal rules guard against comparisons that
would not be valid as model-quality data.

The matcher is a deterministic greedy algorithm on a similarity matrix,
optionally augmented by an LLM judge for ambiguous pairs. The output
satisfies conservation of mass:

    |both| + |partial| + |a_only| = |A|
    |both| + |partial| + |b_only| = |B|
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from autosre.eval.findings import Finding

if TYPE_CHECKING:
    from pathlib import Path

    from autosre.eval.judge import Judge, JudgeResult


class CompareRefusedError(RuntimeError):
    """The two runs are not comparable under the current rule set."""


Bucket = Literal["both", "partial", "a_only", "b_only"]


# ── Manifest loading ────────────────────────────────────────────────


@dataclass(frozen=True)
class RunManifest:
    """Minimal manifest view used by the differ.

    The full manifest has more fields (timestamps, duration, token
    totals, harness version, ...). We only use what is load-bearing for
    the refusal rules and the per-suite lookup.
    """

    run_id: str
    provider: str
    target_repo: str
    target_sha: str | None
    snapshot_digest: str | None
    suites: list[str]
    run_dir: Path
    model: str = ""


def load_manifest(run_dir: Path) -> RunManifest:
    """Load the minimal manifest fields from ``<run_dir>/manifest.json``."""
    path = run_dir / "manifest.json"
    data = json.loads(path.read_text())
    return RunManifest(
        run_id=str(data.get("run_id") or run_dir.name),
        provider=str(data.get("provider") or ""),
        target_repo=str(data.get("target_repo") or ""),
        target_sha=(str(data["target_sha"]) if data.get("target_sha") is not None else None),
        snapshot_digest=(
            str(data["snapshot_digest"]) if data.get("snapshot_digest") is not None else None
        ),
        suites=list(data.get("suites") or []),
        run_dir=run_dir,
        model=str(data.get("model") or ""),
    )


def load_findings(run_dir: Path, suite: str) -> list[Finding]:
    """Load ``findings.jsonl`` for one suite from a run directory."""
    path = run_dir / "suites" / suite / "findings.jsonl"
    if not path.exists():
        return []
    out: list[Finding] = []
    for raw_line in path.read_text().splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        try:
            out.append(Finding.model_validate_json(stripped))
        except Exception:
            continue
    return out


def load_parse_report(run_dir: Path, suite: str) -> dict[str, object]:
    path = run_dir / "suites" / suite / "parse_report.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


# ── Refusal rules ──────────────────────────────────────────────────


@dataclass
class CompareOverrides:
    allow_sha_mismatch: bool = False
    allow_suite_mismatch: bool = False
    allow_same_provider: bool = False
    include_failed_suites: bool = False

    def to_list(self) -> list[str]:
        out: list[str] = []
        if self.allow_sha_mismatch:
            out.append("allow_sha_mismatch")
        if self.allow_suite_mismatch:
            out.append("allow_suite_mismatch")
        if self.allow_same_provider:
            out.append("allow_same_provider")
        if self.include_failed_suites:
            out.append("include_failed_suites")
        return out


def check_comparability(
    a: RunManifest,
    b: RunManifest,
    *,
    overrides: CompareOverrides | None = None,
) -> list[str]:
    """Raise :class:`CompareRefusedError` if the runs are not comparable.

    Returns a list of human-readable warning strings for mismatches the
    overrides allowed. An empty list means "strictly comparable."
    """
    overrides = overrides or CompareOverrides()
    warnings: list[str] = []

    if a.target_repo != b.target_repo:
        raise CompareRefusedError(
            f"target_repo mismatch (a={a.target_repo!r} b={b.target_repo!r}); "
            f"comparing findings across different repos is never valid"
        )

    sha_match = a.target_sha == b.target_sha and a.target_sha is not None
    digest_match = a.snapshot_digest == b.snapshot_digest and a.snapshot_digest is not None
    if not (sha_match or digest_match):
        if not overrides.allow_sha_mismatch:
            raise CompareRefusedError(
                f"target_sha / snapshot_digest mismatch "
                f"(a_sha={a.target_sha} b_sha={b.target_sha} "
                f"a_digest={a.snapshot_digest} b_digest={b.snapshot_digest}); "
                f"pass allow_sha_mismatch=True to override and mark the "
                f"compare as not valid as model-quality data"
            )
        warnings.append("SHA mismatch: this compare is not valid as model-quality data")

    if set(a.suites) != set(b.suites):
        if not overrides.allow_suite_mismatch:
            raise CompareRefusedError(
                f"suite set mismatch: a={sorted(a.suites)} "
                f"b={sorted(b.suites)}; pass allow_suite_mismatch=True to "
                f"diff only the intersection"
            )
        warnings.append("Suite mismatch: only the intersection is diffed")

    if a.provider == b.provider:
        if not overrides.allow_same_provider:
            raise CompareRefusedError(
                f"both runs use provider={a.provider!r}; a provider cannot "
                f"be compared to itself through this command. Use a "
                f"consistency check instead"
            )
        warnings.append("Same-provider compare: this is a consistency check")

    return warnings


# ── Similarity ──────────────────────────────────────────────────────


def _bigrams(text: str) -> frozenset[str]:
    t = text.lower().strip()
    if len(t) < 2:
        return frozenset({t}) if t else frozenset()
    return frozenset(t[i : i + 2] for i in range(len(t) - 1))


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    """Return the Jaccard index of two token sets.

    Both-empty is treated as 0.5 (not 1.0) so that two findings that
    happen to lack descriptions do not get credit for "matching"
    descriptions. Only-one-empty stays 0.0.
    """
    if not a and not b:
        return 0.5
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _triangular(dist: float, *, half_width: float) -> float:
    if dist >= half_width:
        return 0.0
    return max(0.0, 1.0 - dist / half_width)


_ADJACENT_CATEGORIES: dict[str, set[str]] = {
    "security": {"leakage"},
    "leakage": {"security"},
    "dead-code": {"tech-debt"},
    "tech-debt": {"dead-code"},
    "a11y": {"ui-consistency"},
    "ui-consistency": {"a11y"},
}


def similarity(a: Finding, b: Finding) -> float:
    """Return a similarity score in ``[0, 1]`` for two findings.

    Hard gate: if the canonicalized file paths disagree, similarity is
    zero. This prevents matches across files, which are never real.
    """
    if a.file != b.file or not a.file:
        return 0.0

    # Line distance: triangular kernel over ±15 lines, neutral 0.5 if
    # either side is null.
    if a.line is None or b.line is None:
        line_sim = 0.5
    else:
        line_sim = _triangular(abs(a.line - b.line), half_width=15.0)

    if a.category == b.category:
        category_sim = 1.0
    elif b.category in _ADJACENT_CATEGORIES.get(a.category, set()):
        category_sim = 0.25
    else:
        category_sim = 0.0

    title_sim = _jaccard(_bigrams(a.title), _bigrams(b.title))
    desc_sim = _jaccard(_bigrams(a.description), _bigrams(b.description))

    # Path is guaranteed to match here, so path_sim is 1.0 and weight folds in.
    return 0.35 * 1.0 + 0.10 * line_sim + 0.20 * category_sim + 0.15 * title_sim + 0.20 * desc_sim


# ── Matching ────────────────────────────────────────────────────────


@dataclass
class MatchedPair:
    a_idx: int
    b_idx: int
    score: float
    judge: JudgeResult | None = None


@dataclass
class DiffResult:
    """Per-suite bidirectional diff result."""

    suite: str
    a_count: int
    b_count: int
    both: list[MatchedPair] = field(default_factory=list)
    partial: list[MatchedPair] = field(default_factory=list)
    a_only: list[int] = field(default_factory=list)
    b_only: list[int] = field(default_factory=list)

    def check_conservation(self) -> None:
        """Assert conservation of mass; invariant the harness relies on."""
        used_a = {p.a_idx for p in self.both} | {p.a_idx for p in self.partial}
        used_b = {p.b_idx for p in self.both} | {p.b_idx for p in self.partial}
        assert len(used_a) == len(self.both) + len(self.partial), "A double-used"
        assert len(used_b) == len(self.both) + len(self.partial), "B double-used"
        assert len(used_a) + len(self.a_only) == self.a_count, (
            f"A conservation: {len(used_a)} + {len(self.a_only)} != {self.a_count}"
        )
        assert len(used_b) + len(self.b_only) == self.b_count, (
            f"B conservation: {len(used_b)} + {len(self.b_only)} != {self.b_count}"
        )


def diff_suite(
    a_findings: list[Finding],
    b_findings: list[Finding],
    suite: str,
    *,
    judge: Judge | None = None,
    auto_accept_threshold: float = 0.85,
    ambiguous_floor: float = 0.35,
    partial_floor: float = 0.50,
) -> DiffResult:
    """Diff one suite's findings from two runs.

    Pipeline: intra-side dedup → similarity matrix → judge pass on
    ambiguous candidates → greedy one-to-one assignment → bucket.
    """
    a_deduped = _dedupe(a_findings)
    b_deduped = _dedupe(b_findings)

    result = DiffResult(
        suite=suite,
        a_count=len(a_deduped),
        b_count=len(b_deduped),
    )

    # Build similarity matrix — sparse dict keyed by (i, j).
    sim: dict[tuple[int, int], float] = {}
    for i, fa in enumerate(a_deduped):
        for j, fb in enumerate(b_deduped):
            s = similarity(fa, fb)
            if s > 0.0:
                sim[(i, j)] = s

    # Judge pass on ambiguous pairs.
    judge_results: dict[tuple[int, int], JudgeResult] = {}
    if judge is not None:
        for (i, j), s in list(sim.items()):
            if ambiguous_floor <= s < auto_accept_threshold:
                verdict = judge.compare(a_deduped[i], b_deduped[j])
                judge_results[(i, j)] = verdict
                sim[(i, j)] = verdict.score()

    # Greedy one-to-one assignment.
    ordered = sorted(
        ((s, i, j) for (i, j), s in sim.items() if s >= partial_floor),
        key=lambda x: (-x[0], x[1], x[2]),
    )
    used_a: set[int] = set()
    used_b: set[int] = set()
    for s, i, j in ordered:
        if i in used_a or j in used_b:
            continue
        pair = MatchedPair(
            a_idx=i,
            b_idx=j,
            score=s,
            judge=judge_results.get((i, j)),
        )
        used_a.add(i)
        used_b.add(j)
        if s >= auto_accept_threshold:
            result.both.append(pair)
        else:
            result.partial.append(pair)

    result.a_only = [i for i in range(result.a_count) if i not in used_a]
    result.b_only = [j for j in range(result.b_count) if j not in used_b]
    result.check_conservation()
    return result


def _dedupe(
    findings: list[Finding],
    *,
    threshold: float = 0.92,
) -> list[Finding]:
    """Collapse near-duplicate findings on the same side.

    Two paths produce a merge:

    1. **Content-addressed match**: two findings have the same ``Finding.id``.
       This is always a merge, independent of the similarity score, because
       the id is derived from (file, line, category, canonicalized title)
       and collision means the agents described the same thing.
    2. **Fuzzy match**: two findings have similarity ``≥ threshold``.

    Used to prevent one side from inflating its count and skewing
    agreement metrics on the other side.
    """
    if len(findings) <= 1:
        return list(findings)
    n = len(findings)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # Pass 1: id-equality (deterministic, no similarity math).
    by_id: dict[str, list[int]] = {}
    for i, f in enumerate(findings):
        if f.id:
            by_id.setdefault(f.id, []).append(i)
    for members in by_id.values():
        if len(members) > 1:
            first = members[0]
            for other in members[1:]:
                union(first, other)

    # Pass 2: fuzzy similarity for anything not already merged.
    for i in range(n):
        for j in range(i + 1, n):
            if find(i) == find(j):
                continue
            if similarity(findings[i], findings[j]) >= threshold:
                union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    out: list[Finding] = []
    for root in sorted(groups.keys()):
        members = sorted(groups[root])
        keeper = findings[members[0]]
        if len(members) > 1:
            # Merge evidence strings from duplicates into the keeper.
            extras = [findings[k].evidence for k in members[1:] if findings[k].evidence]
            if extras:
                keeper = keeper.model_copy(
                    update={"evidence": "\n---\n".join([keeper.evidence, *extras]).strip()}
                )
        out.append(keeper)
    return out


# ── Compare orchestrator ────────────────────────────────────────────


@dataclass
class CompareResult:
    """Full compare output across all suites."""

    run_a: RunManifest
    run_b: RunManifest
    per_suite: dict[str, DiffResult]
    warnings: list[str]
    overrides: CompareOverrides
    compare_dir: Path | None = None

    def to_json(self) -> dict[str, object]:
        return {
            "run_a": self.run_a.run_id,
            "run_b": self.run_b.run_id,
            "providers": {
                "a": self.run_a.provider,
                "b": self.run_b.provider,
            },
            "target_repo": self.run_a.target_repo,
            "target_sha_match": self.run_a.target_sha == self.run_b.target_sha,
            "valid_as_model_quality": not self.warnings,
            "overrides": self.overrides.to_list(),
            "warnings": self.warnings,
            "per_suite": {
                s: {
                    "a": d.a_count,
                    "b": d.b_count,
                    "both": len(d.both),
                    "partial": len(d.partial),
                    "a_only": len(d.a_only),
                    "b_only": len(d.b_only),
                    "agreement_rate": _agreement_rate(d),
                }
                for s, d in self.per_suite.items()
            },
        }


def _agreement_rate(d: DiffResult) -> float:
    n = max(d.a_count, d.b_count)
    if n == 0:
        return 1.0
    return len(d.both) / n


def compare(
    a_dir: Path,
    b_dir: Path,
    *,
    overrides: CompareOverrides | None = None,
    suite_filter: list[str] | None = None,
    judge: Judge | None = None,
) -> CompareResult:
    """Compare two runs and return a :class:`CompareResult`.

    Does not write any files — the caller decides where to persist via
    :func:`write_compare_dir`.
    """
    a = load_manifest(a_dir)
    b = load_manifest(b_dir)
    warnings = check_comparability(a, b, overrides=overrides)

    suites = sorted(set(a.suites) & set(b.suites))
    if suite_filter:
        suites = [s for s in suites if s in suite_filter]

    per_suite: dict[str, DiffResult] = {}
    for suite in suites:
        if not (overrides and overrides.include_failed_suites):
            pa = load_parse_report(a_dir, suite)
            pb = load_parse_report(b_dir, suite)
            if _has_failures(pa) or _has_failures(pb):
                continue
        a_findings = load_findings(a_dir, suite)
        b_findings = load_findings(b_dir, suite)
        per_suite[suite] = diff_suite(
            a_findings,
            b_findings,
            suite=suite,
            judge=judge,
        )

    return CompareResult(
        run_a=a,
        run_b=b,
        per_suite=per_suite,
        warnings=warnings,
        overrides=overrides or CompareOverrides(),
    )


def _has_failures(parse_report: dict[str, object]) -> bool:
    agents = parse_report.get("agents")
    if not isinstance(agents, list):
        return False
    return any(isinstance(a, dict) and a.get("status") == "failed" for a in agents)


def write_compare_dir(
    result: CompareResult,
    out_dir: Path,
    *,
    index_path: Path | None = None,
) -> Path:
    """Persist a :class:`CompareResult` to disk and return the dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "compare.json").write_text(json.dumps(result.to_json(), indent=2))
    result.compare_dir = out_dir

    if index_path is not None:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "compare_id": out_dir.name,
            "ts": time.time(),
            "run_a": result.run_a.run_id,
            "run_b": result.run_b.run_id,
            "target_sha_match": result.run_a.target_sha == result.run_b.target_sha,
            "suite_intersection": sorted(result.per_suite.keys()),
            "per_suite": result.to_json()["per_suite"],
            "overrides": result.overrides.to_list(),
        }
        with index_path.open("a") as f:
            f.write(json.dumps(row) + "\n")

    return out_dir
