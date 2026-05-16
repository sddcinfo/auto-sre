"""Tests for autosre.mcp_servers.capabilities — MCP command discovery server.

The server introspects the live ``autosre.cli.cli`` click group at import
time, so the catalog shape depends on the real autosre CLI. These tests
assert contract-level properties (top-level commands are present, search
returns relevant matches, get_command returns parameter details) rather
than exact strings.
"""

from __future__ import annotations

import pytest

from autosre.mcp_servers import capabilities


@pytest.fixture(autouse=True)
def _reset_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure each test sees a freshly-built catalog.

    The capabilities module caches the catalog in a module-level global, so
    tests that might be run in any order should reset it explicitly.
    """
    monkeypatch.setattr(capabilities, "_CATALOG", None)


class TestCatalogBuilder:
    def test_builds_nonempty_catalog(self) -> None:
        catalog = capabilities._get_catalog()
        assert len(catalog) > 10

    def test_has_top_level_review_group(self) -> None:
        catalog = capabilities._get_catalog()
        review = next((e for e in catalog if e["path"] == "review"), None)
        assert review is not None
        assert review["is_group"]
        assert "plan" in review["subcommands"]

    def test_has_review_plan_leaf(self) -> None:
        catalog = capabilities._get_catalog()
        plan = next((e for e in catalog if e["path"] == "review plan"), None)
        assert plan is not None
        assert not plan["is_group"]
        # Should have the --chain, --json-output, --reset options we added
        opt_names = {p["name"] for p in plan["params"] if p["kind"] == "option"}
        assert "chain_str" in opt_names or any("chain" in n for n in opt_names)

    def test_has_hooks_backend_group(self) -> None:
        catalog = capabilities._get_catalog()
        hb = next((e for e in catalog if e["path"] == "hooks-backend"), None)
        assert hb is not None
        assert hb["is_group"]
        assert "guard" in hb["subcommands"]
        assert "stop-check" in hb["subcommands"]

    def test_has_hooks_backend_guard_leaf(self) -> None:
        catalog = capabilities._get_catalog()
        guard = next((e for e in catalog if e["path"] == "hooks-backend guard"), None)
        assert guard is not None
        assert not guard["is_group"]


class TestListModules:
    def test_lists_top_level_commands(self) -> None:
        out = capabilities.list_modules()
        assert "review" in out
        assert "hooks-backend" in out
        assert "claude" in out
        assert "backends" in out
        # Should NOT contain nested paths
        assert "review plan" not in out
        assert "hooks-backend guard" not in out

    def test_header_has_count(self) -> None:
        out = capabilities.list_modules()
        assert "autosre commands" in out


class TestSearchCommands:
    def test_finds_review_plan_by_keyword(self) -> None:
        out = capabilities.search_commands("plan review")
        assert "review plan" in out

    def test_finds_guard_by_keyword(self) -> None:
        out = capabilities.search_commands("guard")
        assert "hooks-backend guard" in out

    def test_module_filter(self) -> None:
        out = capabilities.search_commands("plan", module="review")
        # Must only contain commands starting with "review"
        lines = out.splitlines()
        command_lines = [line for line in lines if line.strip().startswith("autosre ")]
        assert command_lines  # at least one match
        for line in command_lines:
            # "autosre review plan" should be present, not "autosre hooks-backend ..."
            assert "autosre review" in line

    def test_empty_query(self) -> None:
        out = capabilities.search_commands("")
        assert "No search terms" in out

    def test_no_matches(self) -> None:
        out = capabilities.search_commands("zzzzzz-absolutely-nothing-matches-this")
        assert "No commands matching" in out


class TestGetCommand:
    def test_get_review_plan_details(self) -> None:
        out = capabilities.get_command("review plan")
        assert "autosre review plan" in out
        # The plan review command has PLAN_PATH as an argument
        assert "PLAN_PATH" in out.upper() or "plan_path" in out.lower()

    def test_autosre_prefix_optional(self) -> None:
        out_with = capabilities.get_command("autosre review plan")
        out_without = capabilities.get_command("review plan")
        assert out_with == out_without

    def test_shows_subcommands_for_groups(self) -> None:
        out = capabilities.get_command("hooks-backend")
        assert "Subcommands:" in out
        assert "guard" in out
        assert "stop-check" in out

    def test_unknown_command_suggests_fuzzy_matches(self) -> None:
        out = capabilities.get_command("revew plan")  # typo
        # Either fuzzy suggests or says not found
        assert "not found" in out or "review plan" in out

    def test_truly_unknown_returns_not_found(self) -> None:
        out = capabilities.get_command("completely-nonexistent-command-xyz")
        assert "not found" in out


class TestScoringInternals:
    def test_score_exact_path_match(self) -> None:
        entry = {
            "path": "review plan",
            "short_help": "",
            "help": "",
            "params": [],
        }
        score = capabilities._score(entry, ["review"])
        assert score == 10  # exact path component match

    def test_score_no_tokens(self) -> None:
        entry = {"path": "anything", "short_help": "", "help": "", "params": []}
        assert capabilities._score(entry, []) == 0

    def test_score_short_help_match(self) -> None:
        entry = {
            "path": "other",
            "short_help": "run plan reviews",
            "help": "",
            "params": [],
        }
        assert capabilities._score(entry, ["plan"]) >= 5
