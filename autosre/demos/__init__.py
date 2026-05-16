"""Enterprise demo framework for GB10 showcases.

Provides audience-adaptive demo scenarios that orchestrate
vLLM backends and Claude Code agent swarms for live presentations.
"""

from autosre.demos.audience import AUDIENCE_PROFILES, AudienceProfile
from autosre.demos.runner import DemoRunner
from autosre.demos.scenario import DemoPhase, DemoScenario

__all__ = [
    "AUDIENCE_PROFILES",
    "AudienceProfile",
    "DemoPhase",
    "DemoRunner",
    "DemoScenario",
]
