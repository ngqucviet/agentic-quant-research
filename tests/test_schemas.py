"""Unit tests for Pydantic v2 hypothesis schemas and dataset-aware guardrails."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agent.schemas import (
    StrategyHypothesis,
    validate_hypothesis_against_data,
)


def test_valid_hypotheses_all_families() -> None:
    cases = [
        ("SMA Test", "momentum", {"fast_window": 10, "slow_window": 40, "min_spread_pct": 0.01}),
        ("ZScore Test", "mean_reversion", {"window": 20, "entry_z": 1.5, "exit_z": 0.0}),
        ("Breakout Test", "breakout", {"entry_window": 20, "exit_window": 10, "vol_multiplier": 0.2}),
        ("RSI Test", "rsi_reversion", {"window": 14, "oversold": 30.0, "overbought": 70.0, "trend_window": 100}),
    ]
    for name, stype, params in cases:
        hyp = StrategyHypothesis(
            hypothesis_name=name,
            strategy_type=stype,
            parameters=params,
            rationale="Testing valid baseline schema serialization and normalization.",
        )
        assert hyp.strategy_type == stype
        assert hyp.max_lookback() > 0


def test_invalid_parameter_ordering_and_extra_keys_rejected() -> None:
    # fast_window >= slow_window
    with pytest.raises(ValidationError):
        StrategyHypothesis(
            hypothesis_name="Bad Momentum",
            strategy_type="momentum",
            parameters={"fast_window": 50, "slow_window": 20},
            rationale="Should fail because fast_window >= slow_window.",
        )

    # Extra unknown parameter key forbidden
    with pytest.raises(ValidationError):
        StrategyHypothesis(
            hypothesis_name="Unknown Key",
            strategy_type="mean_reversion",
            parameters={"window": 20, "entry_z": 1.5, "magic_lever": 99},
            rationale="Should fail due to extra forbidden parameter.",
        )

    # Invalid trend_window between 1 and 19
    with pytest.raises(ValidationError):
        StrategyHypothesis(
            hypothesis_name="Tiny Trend Filter",
            strategy_type="rsi_reversion",
            parameters={"window": 14, "oversold": 30.0, "overbought": 70.0, "trend_window": 5},
            rationale="Should fail because trend_window must be 0 or >= 20.",
        )


def test_domain_guardrails_catch_excessive_lookback_and_duplicates() -> None:
    long_window_hyp = StrategyHypothesis(
        hypothesis_name="Excessive Lookback",
        strategy_type="momentum",
        parameters={"fast_window": 20, "slow_window": 200, "min_spread_pct": 0.0},
        rationale="Testing lookback guardrail against short In-Sample window.",
    )
    # 35% of 200 days = 70 days -> 200 should be rejected
    res = validate_hypothesis_against_data(long_window_hyp, n_in_sample_days=200)
    assert not res.is_valid
    assert any("exceeds safe limit" in err for err in res.errors)

    # Duplicate signature detection
    normal_hyp = StrategyHypothesis(
        hypothesis_name="Normal SMA",
        strategy_type="momentum",
        parameters={"fast_window": 10, "slow_window": 40, "min_spread_pct": 0.0},
        rationale="Testing duplicate signature rejection in session history.",
    )
    seen = {normal_hyp.signature()}
    dup_res = validate_hypothesis_against_data(normal_hyp, n_in_sample_days=600, seen_signatures=seen)
    assert not dup_res.is_valid
    assert any("Duplicate" in err for err in dup_res.errors)
