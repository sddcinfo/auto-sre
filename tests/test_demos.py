"""Tests for autosre.demos module."""

import pytest

from autosre.demos.audience import AUDIENCE_PROFILES
from autosre.demos.runner import DemoRunner
from autosre.demos.scenario import DemoPhase, DemoScenario, list_scenarios


class TestAudienceProfiles:
    def test_all_profiles_defined(self) -> None:
        expected = {"cxo", "engineering", "finance", "hr", "marketing", "product"}
        assert set(AUDIENCE_PROFILES.keys()) == expected

    def test_cxo_profile(self) -> None:
        p = AUDIENCE_PROFILES["cxo"]
        assert p.technical_depth == "low"
        assert "ROI" in p.focus_areas
        assert len(p.talking_points) > 0

    def test_engineering_profile(self) -> None:
        p = AUDIENCE_PROFILES["engineering"]
        assert p.technical_depth == "high"
        assert "performance" in p.focus_areas

    def test_all_profiles_have_talking_points(self) -> None:
        for name, profile in AUDIENCE_PROFILES.items():
            assert isinstance(profile.talking_points, dict), f"{name} missing talking_points"


class TestDemoPhase:
    def test_from_dict_minimal(self) -> None:
        data = {"name": "setup", "duration_minutes": 5}
        phase = DemoPhase.from_dict(data)
        assert phase.name == "setup"
        assert phase.duration_minutes == 5
        assert phase.commands == []
        assert phase.talking_points == []

    def test_from_dict_full(self) -> None:
        data = {
            "name": "agent-swarm",
            "duration_minutes": 20,
            "description": "Launch swarm",
            "commands": ["autosre claude --swarm"],
            "talking_points": ["Point 1"],
            "success_criteria": ["Agents running"],
        }
        phase = DemoPhase.from_dict(data)
        assert len(phase.commands) == 1
        assert len(phase.talking_points) == 1
        assert len(phase.success_criteria) == 1


class TestDemoScenario:
    def test_list_scenarios(self) -> None:
        scenarios = list_scenarios()
        assert len(scenarios) >= 6
        assert "enterprise-overview" in scenarios
        assert "deep-tech" in scenarios
        assert "quick-impact" in scenarios

    def test_load_enterprise_overview(self) -> None:
        s = DemoScenario.load("enterprise-overview")
        assert s.name == "enterprise-overview"
        assert s.model == "qwen3.6-fp8"
        assert s.cluster_required is False
        assert s.total_minutes == 45
        assert len(s.phases) > 0

    def test_load_deep_tech(self) -> None:
        s = DemoScenario.load("deep-tech")
        assert s.model == "qwen3.6-fp8"
        assert s.cluster_required is False

    def test_load_quick_impact(self) -> None:
        s = DemoScenario.load("quick-impact")
        assert s.total_minutes == 20

    def test_load_nonexistent(self) -> None:
        with pytest.raises(FileNotFoundError, match="not found"):
            DemoScenario.load("nonexistent-scenario")

    def test_computed_duration(self) -> None:
        s = DemoScenario.load("enterprise-overview")
        assert s.computed_duration == sum(p.duration_minutes for p in s.phases)

    def test_all_scenarios_valid(self) -> None:
        for name in list_scenarios():
            s = DemoScenario.load(name)
            assert s.name == name
            assert s.model
            assert s.backend == "vllm"
            assert len(s.phases) > 0


class TestDemoRunner:
    def test_preflight_valid_scenario(self) -> None:
        s = DemoScenario.load("enterprise-overview")
        runner = DemoRunner(s)
        _ok, issues = runner.preflight()
        # May fail on "config not found" but model should be valid
        model_issue = [i for i in issues if "not in VllmBackend.models" in i]
        assert model_issue == [], f"Model should be valid: {issues}"

    def test_preflight_with_audience(self) -> None:
        s = DemoScenario.load("enterprise-overview")
        profile = AUDIENCE_PROFILES["cxo"]
        runner = DemoRunner(s, audience=profile)
        assert runner.audience is not None
        assert runner.audience.name == "cxo"

    def test_adapt_audience(self) -> None:
        s = DemoScenario.load("enterprise-overview")
        runner = DemoRunner(s)
        assert runner.audience is None

        runner.adapt_audience(AUDIENCE_PROFILES["engineering"])
        assert runner.audience.name == "engineering"

    def test_status_not_started(self) -> None:
        s = DemoScenario.load("enterprise-overview")
        runner = DemoRunner(s)
        status = runner.status()
        assert status["current_phase"] == "not started"
        assert status["scenario"] == "enterprise-overview"
