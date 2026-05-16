"""Agent swarm orchestration for Claude Code agent teams."""

from autosre.swarm.launcher import SwarmLauncher
from autosre.swarm.templates import TASK_TEMPLATES, TaskTemplate

__all__ = [
    "TASK_TEMPLATES",
    "SwarmLauncher",
    "TaskTemplate",
]
