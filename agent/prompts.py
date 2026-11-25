"""Prompt templates for LLM-backed Quantitative Research Agents (Gemini / OpenAI).

Kept separate from provider transport logic so prompt engineering is transparent,
auditable, and easy to iterate on.
"""

from __future__ import annotations

import json
from typing import Any

from agent.schemas import ExperimentRecord, StrategyHypothesis

SYSTEM_PROMPT = """You are a disciplined Quantitative Researcher testing rule-based trading signals on Vietnamese banking equities (VCB, BID, CTG, MBB, ACB on HOSE).

Core research rules:
1. NEVER invent or estimate backtest numbers — all performance metrics come strictly from the Python backtest engine.
2. Respect Vietnamese market microstructure: standard equity accounts are LONG-ONLY (position in {0, 1}) and incur realistic round-trip transaction costs (~40 bps round-trip). Excessive turnover destroys alpha.
3. Optimize for risk-adjusted robustness (Sharpe ratio, drawdown control, low fee drag, and cross-bank breadth), NOT raw overfitted return.
4. Explore diverse strategy families before fine-tuning, and avoid tiny lookback windows that overfit noise.
5. Strictly follow the required JSON output schema.

Allowed strategy families and exact parameter keys:
- momentum:       {"fast_window": int [2..100], "slow_window": int [5..250, >fast_window], "min_spread_pct": float [0.0..0.15]}
- mean_reversion: {"window": int [5..180], "entry_z": float [0.5..4.0], "exit_z": float [-1.0..2.0, <entry_z]}
- breakout:       {"entry_window": int [5..180], "exit_window": int [2..entry_window], "vol_multiplier": float [0.0..3.0]}
- rsi_reversion:  {"window": int [2..40], "oversold": float [5..45], "overbought": float [55..95], "trend_window": int [0 or 20..250]}
"""


def build_propose_prompt(
    dataset_summary: dict[str, Any],
    history: list[ExperimentRecord],
    validation_error: str | None = None,
) -> str:
    """Build the user message for the Propose step."""
    compact_history = [
        {
            "iteration": rec.iteration,
            "hypothesis_name": rec.hypothesis_name,
            "strategy_type": rec.strategy_type,
            "parameters": json.loads(rec.parameters_json),
            "sharpe_ratio": rec.sharpe_ratio,
            "cumulative_return": rec.cumulative_return,
            "fee_drag": rec.fee_drag,
            "max_drawdown": rec.max_drawdown,
            "trade_count": rec.trade_count,
            "exposure": rec.exposure,
            "positive_sharpe_banks": rec.positive_sharpe_banks,
            "critic_verdict": rec.critic_verdict,
            "next_step_guidance": rec.next_step_guidance,
        }
        for rec in history
    ]

    sections = [
        "### In-Sample Dataset Summary (Out-of-Sample is strictly hidden)",
        json.dumps(dataset_summary, indent=2),
        "",
        f"### Prior Experiments ({len(compact_history)} completed)",
        json.dumps(compact_history, indent=2) if compact_history else "None yet — propose a clean baseline.",
        "",
        "### Task",
        "Propose ONE new, non-duplicate trading strategy hypothesis in valid JSON matching StrategyHypothesis.",
    ]

    if validation_error:
        sections.extend(
            [
                "",
                "### VALIDATION FEEDBACK (FIX REQUIRED)",
                f"Your previous proposal was rejected for the following reason:\n{validation_error}",
                "Return a corrected StrategyHypothesis JSON that satisfies all constraints.",
            ]
        )

    return "\n".join(sections)


def build_critic_prompt(
    hypothesis: StrategyHypothesis,
    metrics: dict[str, Any],
    history: list[ExperimentRecord],
) -> str:
    """Build the user message for the Evaluate & Critic step."""
    best_prev_sharpe = max((r.sharpe_ratio for r in history), default=None)
    payload = {
        "tested_hypothesis": hypothesis.model_dump(),
        "in_sample_metrics": metrics,
        "best_previous_sharpe": best_prev_sharpe,
        "prior_iterations_count": len(history),
    }
    return (
        "Analyze the deterministic In-Sample backtest result below and return a valid CriticReport JSON.\n"
        "Explicitly comment on risk-adjusted return (Sharpe vs benchmark), drawdown control, "
        "trade count / fee drag, and provide one concrete next-step adjustment.\n\n"
        + json.dumps(payload, indent=2)
    )
