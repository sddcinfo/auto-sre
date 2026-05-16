"""Pre-defined task templates for agent swarms."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TaskTemplate:
    """Pre-defined task template for agent swarm launches.

    Provides structured initial prompts and role assignments
    for Claude Code agent teams.
    """

    name: str
    description: str
    num_agents: int
    agent_roles: list[str]
    initial_prompt: str

    def format_prompt(self) -> str:
        """Format the initial prompt with agent role descriptions."""
        roles_text = "\n".join(
            f"- Agent {i + 1}: {role}" for i, role in enumerate(self.agent_roles)
        )
        return f"{self.initial_prompt}\n\nAgent roles:\n{roles_text}"


TASK_TEMPLATES: dict[str, TaskTemplate] = {
    "code-review": TaskTemplate(
        name="code-review",
        description="Multi-agent code review from security, performance, quality, and documentation perspectives",
        num_agents=4,
        agent_roles=[
            "Security Reviewer — identify vulnerabilities, injection risks, auth issues",
            "Performance Analyst — find bottlenecks, memory leaks, inefficient patterns",
            "Code Quality — assess readability, maintainability, design patterns",
            "Documentation & Tests — check coverage, missing tests, unclear docs",
        ],
        initial_prompt=(
            "Create an agent team to perform a comprehensive code review of this project. "
            "Each agent should focus on their specialty area, review independently, "
            "then share findings with the team."
        ),
    ),
    "architecture-analysis": TaskTemplate(
        name="architecture-analysis",
        description="Architecture analysis from scalability, security, and maintainability angles",
        num_agents=3,
        agent_roles=[
            "Scalability Architect — analyze scaling limits, bottlenecks, horizontal/vertical scaling",
            "Security Architect — assess attack surface, data flow, auth boundaries",
            "Maintainability Architect — evaluate modularity, coupling, upgrade paths",
        ],
        initial_prompt=(
            "Create an agent team to analyze the architecture of this system. "
            "Each agent should evaluate from their specialty perspective and "
            "produce actionable recommendations."
        ),
    ),
    "incident-response": TaskTemplate(
        name="incident-response",
        description="SRE incident response simulation with 5 specialized agents",
        num_agents=5,
        agent_roles=[
            "Incident Commander — coordinate the team, track timeline, communicate status",
            "Log Analyst — search through system logs for anomalies and error patterns",
            "Metrics Analyst — examine monitoring data for performance degradation",
            "Root Cause Analyst — trace the failure chain from symptom to origin",
            "Remediation Engineer — prepare fixes, validate them, and plan rollback",
        ],
        initial_prompt=(
            "Create an agent team to investigate and resolve a system incident. "
            "The Incident Commander coordinates while specialists investigate in parallel. "
            "Goal: identify root cause and prepare a verified fix."
        ),
    ),
    "content-generation": TaskTemplate(
        name="content-generation",
        description="Content creation pipeline with research, writing, and review agents",
        num_agents=3,
        agent_roles=[
            "Researcher — gather information, find sources, outline key points",
            "Writer — draft content based on research, maintain voice and style",
            "Editor/Reviewer — review for accuracy, clarity, tone, and completeness",
        ],
        initial_prompt=(
            "Create an agent team for content creation. "
            "The Researcher gathers information, the Writer drafts content, "
            "and the Editor reviews and refines the output."
        ),
    ),
    "data-analysis": TaskTemplate(
        name="data-analysis",
        description="Data exploration, visualization, and insight generation",
        num_agents=3,
        agent_roles=[
            "Data Explorer — examine data structure, quality, distributions",
            "Visualization Specialist — create charts, graphs, and visual summaries",
            "Insights Analyst — identify patterns, trends, and actionable insights",
        ],
        initial_prompt=(
            "Create an agent team to analyze this dataset. "
            "Explore the data, create visualizations, and generate actionable insights."
        ),
    ),
}
