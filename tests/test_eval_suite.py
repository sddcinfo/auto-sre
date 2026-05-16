"""Tests for autosre.eval.suite — loader + YAML suite validation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from autosre.eval.suite import SUITES_DIR, EvalSuite, load_all_suites, load_suite

if TYPE_CHECKING:
    from pathlib import Path

EXPECTED_SUITES = {
    "security",
    "leakage",
    "quality",
    "duplication",
    "dead-code",
    "tech-debt",
    "coverage",
    "ui-consistency",
    "a11y",
    "i18n",
}


class TestBuiltinSuites:
    def test_all_ten_suites_present(self) -> None:
        suites = load_all_suites()
        assert set(suites.keys()) == EXPECTED_SUITES

    @pytest.mark.parametrize("suite_name", sorted(EXPECTED_SUITES))
    def test_each_suite_valid(self, suite_name: str) -> None:
        path = SUITES_DIR / f"{suite_name}.yaml"
        suite = load_suite(path)
        assert suite.name == suite_name
        assert suite.agent_roles
        assert len(suite.agent_roles) == suite.num_agents
        assert "{findings_file}" in suite.initial_prompt

    @pytest.mark.parametrize("suite_name", sorted(EXPECTED_SUITES))
    def test_prompt_renders(self, suite_name: str, tmp_path: Path) -> None:
        suite = load_suite(SUITES_DIR / f"{suite_name}.yaml")
        rendered = suite.render_prompt(findings_file=tmp_path / "out.json")
        assert str(tmp_path / "out.json") in rendered
        assert "Write tool" in rendered or "Write the JSON document" in rendered

    def test_prompts_never_request_chat_findings(self) -> None:
        """The contract is JSON-to-file, never fenced JSON in chat."""
        for suite in load_all_suites().values():
            lower = suite.initial_prompt.lower()
            # The phrase "print findings in chat" may appear as a negative
            # instruction; we're checking that there's no positive
            # instruction to emit a fenced findings: block.
            assert "emit a fenced" not in lower
            assert "emit findings:" not in lower

    def test_category_matches_enum(self) -> None:
        for suite in load_all_suites().values():
            assert suite.category in EXPECTED_SUITES


class TestSuiteValidation:
    def test_rejects_role_count_mismatch(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.yaml"
        path.write_text(
            "name: bad\n"
            "description: bad suite\n"
            "category: security\n"
            "num_agents: 3\n"
            "agent_roles: ['a', 'b']\n"
            "system_prompt: s\n"
            'initial_prompt: "Write to {findings_file}"\n'
        )
        with pytest.raises((ValueError, TypeError)):
            load_suite(path)

    def test_rejects_missing_findings_file_placeholder(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.yaml"
        path.write_text(
            "name: bad\n"
            "description: bad suite\n"
            "category: security\n"
            "num_agents: 1\n"
            "agent_roles: ['a']\n"
            "system_prompt: s\n"
            'initial_prompt: "just do the thing"\n'
        )
        with pytest.raises((ValueError, TypeError)):
            load_suite(path)

    def test_rejects_bad_category(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.yaml"
        path.write_text(
            "name: bad\n"
            "description: bad suite\n"
            "category: not-a-thing\n"
            "num_agents: 1\n"
            "agent_roles: ['a']\n"
            "system_prompt: s\n"
            'initial_prompt: "Write to {findings_file}"\n'
        )
        with pytest.raises((ValueError, TypeError)):
            load_suite(path)


class TestEvalSuiteRender:
    def test_render_substitutes_roles_and_count(self, tmp_path: Path) -> None:
        suite = EvalSuite(
            name="demo",
            description="demo",
            category="security",
            num_agents=2,
            agent_roles=["role-a", "role-b"],
            system_prompt="sys",
            initial_prompt=("Spawn {num_agents} agents:\n{agent_roles}\nWrite to {findings_file}."),
        )
        out = suite.render_prompt(findings_file=tmp_path / "x.json")
        assert "Spawn 2 agents" in out
        assert "- Agent 1: role-a" in out
        assert "- Agent 2: role-b" in out
        assert str(tmp_path / "x.json") in out
