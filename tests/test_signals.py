"""Unit tests for trading signals, including automated causality (zero look-ahead) proofs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tools.signals import (
    breakout_signal,
    generate_signal,
    mean_reversion_signal,
    momentum_signal,
    rsi_reversion_signal,
)


def _make_sample_ohlcv(n: int = 220, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n)
    rets = rng.normal(0.0004, 0.018, size=n)
    close = 20_000.0 * np.cumprod(1.0 + rets)
    open_ = close * (1.0 + rng.normal(0.0, 0.003, size=n))
    high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.006, 0.004, size=n)))
    low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.006, 0.004, size=n)))
    volume = rng.integers(1_000_000, 15_000_000, size=n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


@pytest.mark.parametrize(
    "strategy_type,params",
    [
        ("momentum", {"fast_window": 10, "slow_window": 30, "min_spread_pct": 0.005}),
        ("mean_reversion", {"window": 20, "entry_z": 1.5, "exit_z": 0.0}),
        ("breakout", {"entry_window": 20, "exit_window": 10, "vol_multiplier": 0.2}),
        ("rsi_reversion", {"window": 14, "oversold": 35.0, "overbought": 65.0, "trend_window": 50}),
    ],
)
def test_signals_are_strictly_causal_zero_lookahead(strategy_type: str, params: dict) -> None:
    """Prove that mutating or truncating future bars NEVER alters past signals."""
    df = _make_sample_ohlcv(220)
    cutoff = 140

    sig_full = generate_signal(df, strategy_type=strategy_type, parameters=params, allow_short=False)
    sig_prefix = generate_signal(
        df.iloc[:cutoff], strategy_type=strategy_type, parameters=params, allow_short=False
    )

    # 1. Truncation invariance
    pd.testing.assert_series_equal(sig_full.iloc[:cutoff], sig_prefix)

    # 2. Future shock invariance (wildly distort all prices after cutoff)
    df_shocked = df.copy()
    df_shocked.iloc[cutoff:, df_shocked.columns.get_loc("close")] *= 5.0
    df_shocked.iloc[cutoff:, df_shocked.columns.get_loc("high")] *= 5.0
    df_shocked.iloc[cutoff:, df_shocked.columns.get_loc("low")] *= 0.2
    sig_shocked = generate_signal(
        df_shocked, strategy_type=strategy_type, parameters=params, allow_short=False
    )
    pd.testing.assert_series_equal(sig_full.iloc[:cutoff], sig_shocked.iloc[:cutoff])


@pytest.mark.parametrize(
    "strategy_type,params",
    [
        ("momentum", {"fast_window": 5, "slow_window": 20, "min_spread_pct": 0.0}),
        ("mean_reversion", {"window": 15, "entry_z": 1.2, "exit_z": 0.0}),
        ("breakout", {"entry_window": 15, "exit_window": 5, "vol_multiplier": 0.0}),
        ("rsi_reversion", {"window": 10, "oversold": 35.0, "overbought": 65.0, "trend_window": 0}),
    ],
)
def test_long_only_hose_position_bounds(strategy_type: str, params: dict) -> None:
    """In long-only mode (default for HOSE), positions must lie strictly in {0.0, 1.0}."""
    df = _make_sample_ohlcv(200)
    pos = generate_signal(df, strategy_type=strategy_type, parameters=params, allow_short=False)
    assert set(pos.unique()).issubset({0.0, 1.0})


def test_invalid_signal_parameters_raise() -> None:
    df = _make_sample_ohlcv(80)
    with pytest.raises(ValueError):
        momentum_signal(df, fast_window=30, slow_window=20)
    with pytest.raises(ValueError):
        mean_reversion_signal(df, window=20, entry_z=1.0, exit_z=1.5)
    with pytest.raises(ValueError):
        breakout_signal(df, entry_window=10, exit_window=20)
    with pytest.raises(ValueError):
        rsi_reversion_signal(df, window=14, oversold=70.0, overbought=30.0)
