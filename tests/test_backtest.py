"""Unit tests verifying deterministic backtest math, 1-bar lag, fees, and 5-bank aggregation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.download_data import DEFAULT_TICKERS, generate_synthetic_bank_data
from tools.backtest import evaluate_strategy, run_backtest


def test_one_bar_execution_lag_and_transaction_cost_exact_math() -> None:
    """Hand-verify exact daily net returns, fee drag, trade count, and max drawdown."""
    dates = pd.bdate_range("2024-01-01", periods=5)
    # Daily returns: [0%, +10%, +10%, 0%, -10%]
    prices = [100.0, 110.0, 121.0, 121.0, 108.9]
    df = pd.DataFrame(
        {"open": prices, "high": prices, "low": prices, "close": prices, "volume": [1000] * 5},
        index=dates,
    )

    # Signal decided at close: [0, 1, 1, 0, 0]
    # Executed position (shift(1)): [0, 0, 1, 1, 0]
    # Turnover:                     [0, 0, 1, 0, 1] -> 2 trades
    # Gross daily returns:          [0, 0, +0.10, 0.0, 0.0]
    # Fee (0.01 per side, 0 slip):  [0, 0, 0.01,  0.0, 0.01]
    # Net daily returns:            [0, 0, +0.09, 0.0, -0.01]
    position = pd.Series([0.0, 1.0, 1.0, 0.0, 0.0], index=dates)

    res = run_backtest(df, position, fee_rate=0.01, slippage_rate=0.0)

    expected_gross_equity = 1.10
    expected_net_equity = 1.09 * 0.99  # 1.0791
    expected_max_dd = (1.0791 / 1.09) - 1.0  # -0.01

    assert res.metrics["trade_count"] == 2
    assert res.metrics["total_turnover"] == pytest.approx(2.0, abs=1e-6)
    assert res.metrics["gross_cumulative_return"] == pytest.approx(expected_gross_equity - 1.0, abs=1e-4)
    assert res.metrics["cumulative_return"] == pytest.approx(expected_net_equity - 1.0, abs=1e-4)
    assert res.metrics["fee_drag"] == pytest.approx(
        (expected_gross_equity - 1.0) - (expected_net_equity - 1.0), abs=1e-4
    )
    assert res.metrics["max_drawdown"] == pytest.approx(expected_max_dd, abs=1e-4)
    # Day 1 (+10%) must NOT be captured because signal only turned 1 at day 1 close!
    assert float(res.net_returns.iloc[1]) == pytest.approx(0.0, abs=1e-12)


def test_zero_trades_returns_safe_zeros() -> None:
    dates = pd.bdate_range("2024-01-01", periods=20)
    prices = np.linspace(100.0, 120.0, 20)
    df = pd.DataFrame(
        {"open": prices, "high": prices, "low": prices, "close": prices, "volume": [1000] * 20},
        index=dates,
    )
    pos = pd.Series(0.0, index=dates)
    res = run_backtest(df, pos)
    assert res.metrics["trade_count"] == 0
    assert res.metrics["sharpe_ratio"] == 0.0
    assert res.metrics["cumulative_return"] == 0.0
    assert res.metrics["exposure"] == 0.0


def test_five_bank_equal_weight_universe_evaluation() -> None:
    """Verify multi-bank evaluation runs cleanly across all 5 VN banks."""
    long_df = generate_synthetic_bank_data(DEFAULT_TICKERS, years=2, seed=123)
    res = evaluate_strategy(
        long_df,
        strategy_type="momentum",
        parameters={"fast_window": 10, "slow_window": 30, "min_spread_pct": 0.005},
        target="ALL",
        allow_short=False,
    )
    assert res.metrics["total_banks"] == 5
    assert 0 <= int(res.metrics["positive_sharpe_banks"]) <= 5
    assert res.per_ticker_metrics is not None
    assert set(res.per_ticker_metrics.keys()) == set(DEFAULT_TICKERS)
    assert 0.0 <= float(res.metrics["exposure"]) <= 1.0
