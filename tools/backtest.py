"""Deterministic backtesting engine with look-ahead protection and transaction costs.

All financial metrics are computed strictly in verified Python/NumPy/Pandas code —
the LLM agent NEVER calculates performance metrics itself.

Key safeguards:
    1. **Zero Look-Ahead Bias**: ``executed_position = position.shift(1).fillna(0.0)``
       so a signal decided at day ``t``'s close only earns returns from day ``t+1``.
    2. **Realistic Transaction Costs**: Every position change incurs
       ``|pos_t - pos_{t-1}| * (fee_rate + slippage_rate)``.
       Default: ``fee_rate = 0.0015`` (0.15% brokerage/tax proxy) +
       ``slippage_rate = 0.0005`` (0.05% market impact) = 20 bps per side.
    3. **Leak-Free Slice Evaluation**: Supports computing causal signals on full
       history (for rolling-window warm-up) and slicing strictly to ``[eval_start, eval_end]``
       for Out-of-Sample evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from tools.data_loader import extract_ticker_df, get_available_tickers
from tools.signals import generate_signal

DEFAULT_FEE_RATE = 0.0015       # 15 bps (VN brokerage + sell-tax amortized proxy)
DEFAULT_SLIPPAGE_RATE = 0.0005  # 5 bps slippage
TRADING_DAYS_PER_YEAR = 252


@dataclass
class BacktestResult:
    """Container holding deterministic performance metrics and time series."""

    metrics: dict[str, float | int]
    equity_curve: pd.Series
    benchmark_curve: pd.Series
    drawdown_curve: pd.Series
    benchmark_drawdown: pd.Series
    net_returns: pd.Series
    executed_position: pd.Series | None = None
    per_ticker_metrics: dict[str, dict[str, float | int]] | None = None


def compute_drawdown(equity: pd.Series) -> pd.Series:
    """Compute underwater drawdown series ``(equity / running_max) - 1``."""
    running_max = equity.cummax()
    return (equity / running_max) - 1.0


def compute_metrics_from_returns(
    net_returns: pd.Series,
    gross_returns: pd.Series,
    benchmark_returns: pd.Series,
    turnover: pd.Series,
    exposure_series: pd.Series,
    trade_count: int,
    annualization: int = TRADING_DAYS_PER_YEAR,
) -> tuple[dict[str, float | int], pd.Series, pd.Series, pd.Series, pd.Series]:
    """Compute standardized quant performance metrics from aligned daily returns."""
    n_days = int(len(net_returns))
    if n_days == 0:
        raise ValueError("Cannot compute backtest metrics on an empty return series.")

    equity = (1.0 + net_returns).cumprod()
    gross_equity = (1.0 + gross_returns).cumprod()
    benchmark_equity = (1.0 + benchmark_returns).cumprod()

    drawdown = compute_drawdown(equity)
    benchmark_dd = compute_drawdown(benchmark_equity)

    cum_return = float(equity.iloc[-1] - 1.0)
    gross_cum_return = float(gross_equity.iloc[-1] - 1.0)
    bench_return = float(benchmark_equity.iloc[-1] - 1.0)
    excess_return = cum_return - bench_return
    fee_drag = gross_cum_return - cum_return

    years = max(n_days / float(annualization), 1.0 / float(annualization))
    cagr = float(equity.iloc[-1] ** (1.0 / years) - 1.0) if equity.iloc[-1] > 0 else -1.0

    daily_std = float(net_returns.std(ddof=1)) if n_days > 1 else 0.0
    ann_vol = daily_std * np.sqrt(annualization)
    sharpe = (
        float((net_returns.mean() / daily_std) * np.sqrt(annualization))
        if daily_std > 1e-12
        else 0.0
    )

    bench_std = float(benchmark_returns.std(ddof=1)) if n_days > 1 else 0.0
    bench_sharpe = (
        float((benchmark_returns.mean() / bench_std) * np.sqrt(annualization))
        if bench_std > 1e-12
        else 0.0
    )

    downside = net_returns[net_returns < 0.0]
    downside_std = float(downside.std(ddof=1)) if len(downside) > 1 else 0.0
    sortino = (
        float((net_returns.mean() / downside_std) * np.sqrt(annualization))
        if downside_std > 1e-12
        else 0.0
    )

    max_dd = float(drawdown.min())
    bench_max_dd = float(benchmark_dd.min())
    calmar = float(cagr / abs(max_dd)) if abs(max_dd) > 1e-12 else 0.0

    active_mask = exposure_series.abs() > 1e-9
    exposure = float(exposure_series.abs().clip(upper=1.0).mean())
    if active_mask.any():
        win_rate = float((net_returns[active_mask] > 0.0).mean())
    else:
        win_rate = 0.0

    metrics: dict[str, float | int] = {
        "n_days": n_days,
        "cumulative_return": round(cum_return, 4),
        "gross_cumulative_return": round(gross_cum_return, 4),
        "fee_drag": round(fee_drag, 4),
        "benchmark_return": round(bench_return, 4),
        "excess_return": round(excess_return, 4),
        "cagr": round(cagr, 4),
        "annualized_volatility": round(ann_vol, 4),
        "sharpe_ratio": round(sharpe, 4),
        "benchmark_sharpe": round(bench_sharpe, 4),
        "sortino_ratio": round(sortino, 4),
        "max_drawdown": round(max_dd, 4),
        "benchmark_max_drawdown": round(bench_max_dd, 4),
        "calmar_ratio": round(calmar, 4),
        "trade_count": int(trade_count),
        "total_turnover": round(float(turnover.sum()), 4),
        "exposure": round(exposure, 4),
        "win_rate": round(win_rate, 4),
    }

    return metrics, equity, benchmark_equity, drawdown, benchmark_dd


def run_backtest(
    df: pd.DataFrame,
    position: pd.Series,
    fee_rate: float = DEFAULT_FEE_RATE,
    slippage_rate: float = DEFAULT_SLIPPAGE_RATE,
    annualization: int = TRADING_DAYS_PER_YEAR,
    eval_start: pd.Timestamp | str | None = None,
    eval_end: pd.Timestamp | str | None = None,
) -> BacktestResult:
    """Run deterministic single-asset backtest with 1-bar execution lag and costs.

    Args:
        df: Single-ticker OHLCV DataFrame indexed by ``date``.
        position: Target position series aligned to ``df.index`` (values in {-1, 0, 1}).
        fee_rate: One-way commission/tax rate per 1.0 unit of turnover.
        slippage_rate: One-way slippage rate per 1.0 unit of turnover.
        annualization: Trading days per year (252).
        eval_start: Optional inclusive start date for slicing AFTER shifting
            (preserves rolling-window warm-up without any look-ahead leakage).
        eval_end: Optional inclusive end date for slicing.
    """
    if fee_rate < 0.0 or slippage_rate < 0.0:
        raise ValueError("fee_rate and slippage_rate must be non-negative.")

    close = df["close"].astype(float).sort_index()
    aligned_pos = position.reindex(close.index).fillna(0.0).astype(float)

    # CRITICAL: Shift position by 1 trading day to eliminate look-ahead bias.
    executed_pos = aligned_pos.shift(1).fillna(0.0)
    asset_returns = close.pct_change().fillna(0.0)

    # Slice to requested evaluation window (e.g., Out-of-Sample holdout).
    if eval_start is not None:
        start_ts = pd.Timestamp(eval_start)
        mask = close.index >= start_ts
        executed_pos = executed_pos.loc[mask]
        asset_returns = asset_returns.loc[mask]
    if eval_end is not None:
        end_ts = pd.Timestamp(eval_end)
        mask = executed_pos.index <= end_ts
        executed_pos = executed_pos.loc[mask]
        asset_returns = asset_returns.loc[mask]

    if executed_pos.empty:
        raise ValueError("Evaluation window produced an empty time series.")

    # Turnover on day 0 of the evaluation window accounts for entering initial exposure.
    turnover = executed_pos.diff().abs()
    turnover.iloc[0] = abs(float(executed_pos.iloc[0]))

    total_cost_rate = fee_rate + slippage_rate
    gross_returns = executed_pos * asset_returns
    transaction_costs = turnover * total_cost_rate
    net_returns = gross_returns - transaction_costs
    trade_count = int((turnover > 1e-9).sum())

    metrics, equity, bench_equity, drawdown, bench_dd = compute_metrics_from_returns(
        net_returns=net_returns,
        gross_returns=gross_returns,
        benchmark_returns=asset_returns,
        turnover=turnover,
        exposure_series=executed_pos,
        trade_count=trade_count,
        annualization=annualization,
    )

    return BacktestResult(
        metrics=metrics,
        equity_curve=equity,
        benchmark_curve=bench_equity,
        drawdown_curve=drawdown,
        benchmark_drawdown=bench_dd,
        net_returns=net_returns,
        executed_position=executed_pos,
    )


def evaluate_strategy(
    long_df: pd.DataFrame,
    strategy_type: str,
    parameters: dict[str, Any],
    target: str = "ALL",
    allow_short: bool = False,
    fee_rate: float = DEFAULT_FEE_RATE,
    slippage_rate: float = DEFAULT_SLIPPAGE_RATE,
    annualization: int = TRADING_DAYS_PER_YEAR,
    eval_start: pd.Timestamp | str | None = None,
    eval_end: pd.Timestamp | str | None = None,
) -> BacktestResult:
    """Evaluate a parameterized strategy on either one bank or an equal-weight 5-bank basket.

    When ``target == "ALL"``, the exact same rule and parameters are applied
    independently to every bank in ``long_df`` (zero cross-stock cherry-picking),
    and daily net returns are combined into an equal-weighted portfolio.
    """
    tickers = get_available_tickers(long_df) if target == "ALL" else [target]

    if len(tickers) == 1:
        ticker_df = extract_ticker_df(long_df, tickers[0])
        pos = generate_signal(
            ticker_df,
            strategy_type=strategy_type,
            parameters=parameters,
            allow_short=allow_short,
        )
        res = run_backtest(
            ticker_df,
            pos,
            fee_rate=fee_rate,
            slippage_rate=slippage_rate,
            annualization=annualization,
            eval_start=eval_start,
            eval_end=eval_end,
        )
        res.per_ticker_metrics = {tickers[0]: res.metrics}
        return res

    # Multi-bank Equal-Weight Basket evaluation
    net_ret_map: dict[str, pd.Series] = {}
    gross_ret_map: dict[str, pd.Series] = {}
    bench_ret_map: dict[str, pd.Series] = {}
    turnover_map: dict[str, pd.Series] = {}
    exposure_map: dict[str, pd.Series] = {}
    per_ticker_metrics: dict[str, dict[str, float | int]] = {}
    total_trades = 0

    for ticker in tickers:
        ticker_df = extract_ticker_df(long_df, ticker)
        pos = generate_signal(
            ticker_df,
            strategy_type=strategy_type,
            parameters=parameters,
            allow_short=allow_short,
        )
        single = run_backtest(
            ticker_df,
            pos,
            fee_rate=fee_rate,
            slippage_rate=slippage_rate,
            annualization=annualization,
            eval_start=eval_start,
            eval_end=eval_end,
        )
        per_ticker_metrics[ticker] = single.metrics
        total_trades += int(single.metrics["trade_count"])

        # Reconstruct aligned gross/benchmark/turnover series on the sliced dates
        idx = single.net_returns.index
        exec_pos = single.executed_position
        assert exec_pos is not None
        asset_ret = ticker_df["close"].astype(float).pct_change().fillna(0.0).reindex(idx).fillna(0.0)
        turn = exec_pos.diff().abs()
        turn.iloc[0] = abs(float(exec_pos.iloc[0]))

        net_ret_map[ticker] = single.net_returns
        gross_ret_map[ticker] = exec_pos * asset_ret
        bench_ret_map[ticker] = asset_ret
        turnover_map[ticker] = turn
        exposure_map[ticker] = exec_pos.abs()

    ew_net = pd.DataFrame(net_ret_map).mean(axis=1).dropna()
    ew_gross = pd.DataFrame(gross_ret_map).reindex(ew_net.index).mean(axis=1)
    ew_bench = pd.DataFrame(bench_ret_map).reindex(ew_net.index).mean(axis=1)
    ew_turnover = pd.DataFrame(turnover_map).reindex(ew_net.index).mean(axis=1)
    ew_exposure = pd.DataFrame(exposure_map).reindex(ew_net.index).mean(axis=1)

    metrics, equity, bench_equity, drawdown, bench_dd = compute_metrics_from_returns(
        net_returns=ew_net,
        gross_returns=ew_gross,
        benchmark_returns=ew_bench,
        turnover=ew_turnover,
        exposure_series=ew_exposure,
        trade_count=total_trades,
        annualization=annualization,
    )

    # Track cross-bank breadth: how many banks achieved positive Sharpe?
    positive_sharpe_banks = sum(
        1 for m in per_ticker_metrics.values() if float(m["sharpe_ratio"]) > 0.0
    )
    metrics["positive_sharpe_banks"] = positive_sharpe_banks
    metrics["total_banks"] = len(tickers)

    return BacktestResult(
        metrics=metrics,
        equity_curve=equity,
        benchmark_curve=bench_equity,
        drawdown_curve=drawdown,
        benchmark_drawdown=bench_dd,
        net_returns=ew_net,
        executed_position=None,
        per_ticker_metrics=per_ticker_metrics,
    )
