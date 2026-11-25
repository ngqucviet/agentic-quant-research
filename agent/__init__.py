"""Agentic quantitative research loop, schemas, prompts, and providers."""

from agent.providers import (
    GeminiProvider,
    MockResearchProvider,
    OpenAIProvider,
    ResearchProvider,
    create_provider,
)
from agent.schemas import (
    CriticReport,
    ExperimentRecord,
    StrategyHypothesis,
    ValidationResult,
    validate_hypothesis_against_data,
)

__all__ = [
    "CriticReport",
    "ExperimentRecord",
    "GeminiProvider",
    "MockResearchProvider",
    "OpenAIProvider",
    "ResearchProvider",
    "StrategyHypothesis",
    "ValidationResult",
    "create_provider",
    "validate_hypothesis_against_data",
]
