"""Pydantic v2 schemas and domain guardrails for the Agentic Quant Research Loop.

Every hypothesis proposed by an LLM (or mock researcher) MUST validate cleanly
against these schemas AND against dataset-level safety constraints before the
backtest engine is ever invoked.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

StrategyType = Literal["momentum", "mean_reversion", "breakout", "rsi_reversion"]
CriticVerdict = Literal["promising", "mixed", "overtrading", "underperforming", "no_trades"]


class MomentumParams(BaseModel):
    """Parameters for dual moving average crossover."""

    model_config = ConfigDict(extra="forbid")

    fast_window: int = Field(..., ge=2, le=100, description="Fast SMA lookback in trading days.")
    slow_window: int = Field(..., ge=5, le=250, description="Slow SMA lookback in trading days.")
    min_spread_pct: float = Field(
        default=0.0,
        ge=0.0,
        le=0.15,
        description="Hysteresis deadband on (SMA_fast - SMA_slow) / SMA_slow to suppress whipsaw.",
    )

    @model_validator(mode="after")
    def check_window_order(self) -> "MomentumParams":
        if self.fast_window >= self.slow_window:
            raise ValueError(
                f"fast_window ({self.fast_window}) must be strictly less than slow_window ({self.slow_window})."
            )
        return self


class MeanReversionParams(BaseModel):
    """Parameters for rolling Z-score mean-reversion."""

    model_config = ConfigDict(extra="forbid")

    window: int = Field(..., ge=5, le=180, description="Rolling mean/std lookback window.")
    entry_z: float = Field(..., ge=0.5, le=4.0, description="Z-score magnitude to enter position.")
    exit_z: float = Field(
        default=0.0,
        ge=-1.0,
        le=2.0,
        description="Z-score threshold to exit back to flat (must be < entry_z).",
    )

    @model_validator(mode="after")
    def check_z_order(self) -> "MeanReversionParams":
        if self.exit_z >= self.entry_z:
            raise ValueError(
                f"exit_z ({self.exit_z}) must be strictly less than entry_z ({self.entry_z})."
            )
        return self


class BreakoutParams(BaseModel):
    """Parameters for Donchian channel / volatility-confirmed breakout."""

    model_config = ConfigDict(extra="forbid")

    entry_window: int = Field(..., ge=5, le=180, description="Prior-high lookback for breakout entry.")
    exit_window: int = Field(..., ge=2, le=180, description="Prior-low lookback for trailing exit.")
    vol_multiplier: float = Field(
        default=0.0,
        ge=0.0,
        le=3.0,
        description="Optional rolling price-std multiplier added above breakout channel.",
    )

    @model_validator(mode="after")
    def check_window_order(self) -> "BreakoutParams":
        if self.exit_window > self.entry_window:
            raise ValueError(
                f"exit_window ({self.exit_window}) must be <= entry_window ({self.entry_window})."
            )
        return self


class RSIReversionParams(BaseModel):
    """Parameters for Wilder RSI mean-reversion with optional trend filter."""

    model_config = ConfigDict(extra="forbid")

    window: int = Field(..., ge=2, le=40, description="RSI smoothing window.")
    oversold: float = Field(..., ge=5.0, le=45.0, description="RSI entry threshold for long.")
    overbought: float = Field(..., ge=55.0, le=95.0, description="RSI exit threshold.")
    trend_window: int = Field(
        default=0,
        ge=0,
        le=250,
        description="Optional SMA trend filter (0 = disabled, or 20..250).",
    )

    @model_validator(mode="after")
    def check_rsi_bounds(self) -> "RSIReversionParams":
        if self.oversold >= self.overbought:
            raise ValueError(
                f"oversold ({self.oversold}) must be < overbought ({self.overbought})."
            )
        if 0 < self.trend_window < 20:
            raise ValueError(
                f"trend_window must be 0 (disabled) or >= 20, got {self.trend_window}."
            )
        return self


PARAM_MODEL_MAP: dict[str, type[BaseModel]] = {
    "momentum": MomentumParams,
    "mean_reversion": MeanReversionParams,
    "breakout": BreakoutParams,
    "rsi_reversion": RSIReversionParams,
}


class StrategyHypothesis(BaseModel):
    """Structured trading hypothesis emitted by the Research Agent."""

    model_config = ConfigDict(extra="forbid")

    hypothesis_name: str = Field(
        ...,
        min_length=3,
        max_length=80,
        description="Short descriptive name for the strategy hypothesis.",
    )
    strategy_type: StrategyType = Field(
        ...,
        description="One of: momentum, mean_reversion, breakout, rsi_reversion.",
    )
    parameters: dict[str, int | float] = Field(
        ...,
        description="Parameter dictionary matching the selected strategy_type.",
    )
    rationale: str = Field(
        ...,
        min_length=10,
        max_length=600,
        description="Concise quantitative reason for testing this hypothesis based on prior runs.",
    )

    @model_validator(mode="after")
    def validate_strategy_parameters(self) -> "StrategyHypothesis":
        param_cls = PARAM_MODEL_MAP[self.strategy_type]
        validated = param_cls.model_validate(self.parameters)
        self.parameters = validated.model_dump()
        return self

    def signature(self) -> str:
        """Canonical string key used to detect duplicate proposals."""
        canonical = json.dumps(self.parameters, sort_keys=True)
        return f"{self.strategy_type}:{canonical}"

    def max_lookback(self) -> int:
        """Return the largest rolling lookback window required by this hypothesis."""
        lookback_keys = ("slow_window", "fast_window", "window", "entry_window", "exit_window", "trend_window")
        return max(int(self.parameters.get(k, 0)) for k in lookback_keys)


class ValidationResult(BaseModel):
    """Outcome of domain-level hypothesis validation."""

    is_valid: bool
    errors: list[str] = Field(default_factory=list)


def validate_hypothesis_against_data(
    hypothesis: StrategyHypothesis,
    n_in_sample_days: int,
    seen_signatures: set[str] | None = None,
    max_lookback_fraction: float = 0.35,
) -> ValidationResult:
    """Apply dataset-aware safety checks beyond static Pydantic validation.

    Guards against:
        1. Lookback windows too large relative to the In-Sample length (which
           would leave too few post-warmup days for reliable Sharpe estimation).
        2. Duplicate configurations already evaluated in the current session.
    """
    errors: list[str] = []
    max_allowed = max(20, int(n_in_sample_days * max_lookback_fraction))
    lookback = hypothesis.max_lookback()
    if lookback > max_allowed:
        errors.append(
            f"Max lookback window ({lookback} days) exceeds safe limit of "
            f"{max_allowed} days ({int(max_lookback_fraction * 100)}% of {n_in_sample_days} IS days)."
        )

    if seen_signatures is not None and hypothesis.signature() in seen_signatures:
        errors.append(
            f"Duplicate configuration already tested: {hypothesis.signature()}. "
            "Propose a different parameter set or strategy family."
        )

    return ValidationResult(is_valid=len(errors) == 0, errors=errors)


class CriticReport(BaseModel):
    """Structured quantitative critique of a completed backtest iteration."""

    model_config = ConfigDict(extra="forbid")

    verdict: CriticVerdict = Field(
        ...,
        description="High-level diagnosis: promising, mixed, overtrading, underperforming, or no_trades.",
    )
    summary: str = Field(
        ...,
        min_length=10,
        max_length=500,
        description="Objective interpretation of Sharpe, drawdown, fee drag, and trade count.",
    )
    key_risk: str = Field(
        ...,
        min_length=5,
        max_length=300,
        description="Primary weakness or overfitting/cost risk observed in this run.",
    )
    next_step_guidance: str = Field(
        ...,
        min_length=5,
        max_length=300,
        description="Actionable adjustment for the next research iteration.",
    )


class ExperimentRecord(BaseModel):
    """Flat, CSV-ready log entry for one In-Sample research iteration."""

    iteration: int
    timestamp_utc: str
    provider: str
    target: str
    hypothesis_name: str
    strategy_type: StrategyType
    parameters_json: str
    rationale: str
    validation_retries: int
    n_days: int
    cumulative_return: float
    gross_cumulative_return: float
    fee_drag: float
    benchmark_return: float
    excess_return: float
    cagr: float
    annualized_volatility: float
    sharpe_ratio: float
    benchmark_sharpe: float
    sortino_ratio: float
    max_drawdown: float
    benchmark_max_drawdown: float
    calmar_ratio: float
    trade_count: int
    total_turnover: float
    exposure: float
    win_rate: float
    positive_sharpe_banks: int | None = None
    total_banks: int | None = None
    critic_verdict: CriticVerdict
    critic_summary: str
    critic_key_risk: str
    next_step_guidance: str

    @classmethod
    def from_iteration(
        cls,
        *,
        iteration: int,
        timestamp_utc: str,
        provider: str,
        target: str,
        hypothesis: StrategyHypothesis,
        validation_retries: int,
        metrics: dict[str, Any],
        critic: CriticReport,
    ) -> "ExperimentRecord":
        return cls(
            iteration=iteration,
            timestamp_utc=timestamp_utc,
            provider=provider,
            target=target,
            hypothesis_name=hypothesis.hypothesis_name,
            strategy_type=hypothesis.strategy_type,
            parameters_json=json.dumps(hypothesis.parameters, sort_keys=True),
            rationale=hypothesis.rationale,
            validation_retries=validation_retries,
            n_days=int(metrics["n_days"]),
            cumulative_return=float(metrics["cumulative_return"]),
            gross_cumulative_return=float(metrics["gross_cumulative_return"]),
            fee_drag=float(metrics["fee_drag"]),
            benchmark_return=float(metrics["benchmark_return"]),
            excess_return=float(metrics["excess_return"]),
            cagr=float(metrics["cagr"]),
            annualized_volatility=float(metrics["annualized_volatility"]),
            sharpe_ratio=float(metrics["sharpe_ratio"]),
            benchmark_sharpe=float(metrics["benchmark_sharpe"]),
            sortino_ratio=float(metrics["sortino_ratio"]),
            max_drawdown=float(metrics["max_drawdown"]),
            benchmark_max_drawdown=float(metrics["benchmark_max_drawdown"]),
            calmar_ratio=float(metrics["calmar_ratio"]),
            trade_count=int(metrics["trade_count"]),
            total_turnover=float(metrics["total_turnover"]),
            exposure=float(metrics["exposure"]),
            win_rate=float(metrics["win_rate"]),
            positive_sharpe_banks=(
                int(metrics["positive_sharpe_banks"])
                if "positive_sharpe_banks" in metrics
                else None
            ),
            total_banks=int(metrics["total_banks"]) if "total_banks" in metrics else None,
            critic_verdict=critic.verdict,
            critic_summary=critic.summary,
            critic_key_risk=critic.key_risk,
            next_step_guidance=critic.next_step_guidance,
        )
