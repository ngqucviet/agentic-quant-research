"""Deterministic quantitative research tools (data loading, signals, backtesting)."""

from tools.backtest import (
    DEFAULT_FEE_RATE,
    DEFAULT_SLIPPAGE_RATE,
    TRADING_DAYS_PER_YEAR,
    BacktestResult,
    compute_drawdown,
    evaluate_strategy,
    run_backtest,
)
from tools.data_loader import (
    VN_BANK_UNIVERSE,
    SplitInfo,
    compute_split_info,
    extract_ticker_df,
    get_available_tickers,
    load_prices,
    split_in_out_sample,
    summarize_in_sample,
)
from tools.signals import (
    STRATEGY_REGISTRY,
    breakout_signal,
    generate_signal,
    mean_reversion_signal,
    momentum_signal,
    rsi_reversion_signal,
)

__all__ = [
    "BacktestResult",
    "DEFAULT_FEE_RATE",
    "DEFAULT_SLIPPAGE_RATE",
    "STRATEGY_REGISTRY",
    "SplitInfo",
    "TRADING_DAYS_PER_YEAR",
    "VN_BANK_UNIVERSE",
    "breakout_signal",
    "compute_drawdown",
    "compute_split_info",
    "evaluate_strategy",
    "extract_ticker_df",
    "generate_signal",
    "get_available_tickers",
    "load_prices",
    "mean_reversion_signal",
    "momentum_signal",
    "rsi_reversion_signal",
    "run_backtest",
    "split_in_out_sample",
    "summarize_in_sample",
]
