"""Audience profiles for demo customization."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AudienceProfile:
    """Audience profile for adapting demo content in real-time.

    Each profile defines focus areas, technical depth, and talking points
    that overlay onto any demo scenario.
    """

    name: str
    description: str
    focus_areas: list[str]
    technical_depth: str  # "low", "medium", "high"
    key_metrics: list[str]
    talking_points: dict[str, list[str]] = field(default_factory=dict)


AUDIENCE_PROFILES: dict[str, AudienceProfile] = {
    "cxo": AudienceProfile(
        name="cxo",
        description="C-suite executives (CEO, CTO, CIO, CISO)",
        focus_areas=["ROI", "competitive advantage", "data sovereignty", "time-to-value"],
        technical_depth="low",
        key_metrics=[
            "Cost savings vs cloud API ($/month)",
            "Data never leaves your network",
            "Single command to AI-ready",
            "Zero vendor lock-in (open-source models)",
        ],
        talking_points={
            "setup": [
                "From bare metal to AI in one command",
                "No cloud dependencies, no API keys, fully on-premise",
                "Your data never leaves your building",
            ],
            "agent-swarm": [
                "Multiple AI agents working in parallel — like a team of analysts",
                "Each agent specializes in a different area",
                "Results in minutes, not days",
            ],
            "showcase": [
                "Total cost of ownership vs cloud API at scale",
                "Data sovereignty: zero external data transfer",
                "Scales with your team, not your API bill",
            ],
        },
    ),
    "engineering": AudienceProfile(
        name="engineering",
        description="Engineering leaders, architects, and senior developers",
        focus_areas=["performance", "architecture", "model quality", "developer experience"],
        technical_depth="high",
        key_metrics=[
            "Tokens/sec (decode throughput)",
            "NCCL inter-node bandwidth (GB/s)",
            "Context window (tokens with TurboQuant)",
            "Tool calling accuracy",
            "Time to first token (TTFT)",
        ],
        talking_points={
            "setup": [
                "GB10: SM121a Blackwell, 128GB unified LPDDR5X, 273 GB/s bandwidth",
                "NVFP4 quantization: 3-4x model compression, minimal quality loss",
                "TurboQuant: 2.6x context expansion, zero quality degradation",
            ],
            "cluster": [
                "NCCL over ConnectX-7 RoCE: ~185 Gbps inter-node",
                "Tensor parallelism: TP=2 for 120B+ parameter models",
                "Ray-based distributed serving with automatic failover",
            ],
            "agent-swarm": [
                "Claude Code agent teams: independent context windows per agent",
                "Shared task list with dependency tracking",
                "Each agent has full tool access: file I/O, shell, web search",
            ],
        },
    ),
    "finance": AudienceProfile(
        name="finance",
        description="Finance, compliance, and risk management leaders",
        focus_areas=["compliance", "audit trails", "cost control", "data governance"],
        technical_depth="medium",
        key_metrics=[
            "Total cost of ownership",
            "Data residency compliance",
            "Audit trail completeness",
            "Operational cost per inference",
        ],
        talking_points={
            "setup": [
                "All data processing happens on-premise",
                "No third-party API calls — full data residency",
                "Hardware is a capital expense, not recurring API costs",
            ],
            "agent-swarm": [
                "AI-assisted financial analysis in parallel",
                "Document review across multiple compliance frameworks",
                "Audit trail: every agent action is logged",
            ],
        },
    ),
    "hr": AudienceProfile(
        name="hr",
        description="HR, people operations, and talent leaders",
        focus_areas=["employee experience", "policy automation", "confidential data"],
        technical_depth="low",
        key_metrics=[
            "Time saved on policy review",
            "Employee data stays on-premise",
            "Consistency of AI-assisted decisions",
        ],
        talking_points={
            "setup": [
                "Employee data never leaves your systems",
                "AI assists with policy interpretation, not decisions",
            ],
            "agent-swarm": [
                "Multiple agents analyze policy documents simultaneously",
                "Cross-reference employment law across jurisdictions",
                "Generate consistent, auditable recommendations",
            ],
        },
    ),
    "marketing": AudienceProfile(
        name="marketing",
        description="Marketing, brand, and content leaders",
        focus_areas=["content generation", "brand voice", "campaign analysis", "speed"],
        technical_depth="low",
        key_metrics=[
            "Content generation speed",
            "Brand voice consistency",
            "Campaign analysis depth",
        ],
        talking_points={
            "setup": [
                "On-premise AI means your brand data stays private",
                "No risk of training data leaking to competitors",
            ],
            "agent-swarm": [
                "One agent researches, one writes, one reviews — in parallel",
                "Consistent brand voice across all outputs",
                "Real-time campaign analysis from multiple angles",
            ],
        },
    ),
    "product": AudienceProfile(
        name="product",
        description="Product management and design leaders",
        focus_areas=["feature ideation", "user research", "roadmap planning", "prototyping"],
        technical_depth="medium",
        key_metrics=[
            "Time from idea to prototype",
            "Breadth of analysis per feature",
            "Integration with existing workflows",
        ],
        talking_points={
            "setup": [
                "AI co-pilot for product development, running locally",
                "Prototype features with AI assistance in minutes",
            ],
            "agent-swarm": [
                "Agents tackle user research, competitive analysis, and spec writing simultaneously",
                "Each agent brings a different perspective to the same problem",
                "Synthesize findings into actionable recommendations",
            ],
        },
    ),
}
