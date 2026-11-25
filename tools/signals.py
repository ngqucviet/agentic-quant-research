"""Strictly causal, parameterized trading signal generators.

Every function in this module takes a chronologically sorted OHLCV DataFrame
(indexed by ``date``) and returns a ``pd.Series`` named ``"position"`` with
values in:
    - ``{0.0, 1.0}`` when ``allow_short=False`` (default — matches HOSE rules
      for Vietnamese banking equities where standard short selling is disallowed)
    - ``{-1.0, 0.0, 1.0}`` when ``allow_short=True``

Causality guarantee:
    ``position.iloc[t]`` is computed strictly from observations ``<= t``.
    To prevent look-ahead bias, ``tools/backtest.py`` ALWAYS applies
    ``position.shift(1)`` before multiplying by asset returns.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def momentum_signal(
    df: pd.DataFrame,
    fast_window: int = 10,
    slow_window: int = 50,
    min_spread_pct: float = 0.0,
    allow_short: bool = False,
) -> pd.Series:
    """Dual Simple Moving Average crossover with optional hysteresis deadband.

    - Enters long (+1) when ``(SMA_fast - SMA_slow) / SMA_slow > min_spread_pct``.
    - Exits to flat (0) or short (-1) when ``(SMA_fast - SMA_slow) / SMA_slow < -min_spread_pct``.
    - Holds previous position inside the deadband to reduce whipsaw turnover.
    """
    if fast_window < 2 or slow_window <= fast_window:
        raise ValueError(
            f"Invalid momentum windows: fast_window={fast_window}, slow_window={slow_window} "
            "(require 2 <= fast_window < slow_window)."
        )
    if min_spread_pct < 0.0:
        raise ValueError(f"min_spread_pct must be >= 0, got {min_spread_pct}")

    close = df["close"].astype(float)
    sma_fast = close.rolling(window=fast_window, min_periods=fast_window).mean()
    sma_slow = close.rolling(window=slow_window, min_periods=slow_window).mean()

    spread = ((sma_fast - sma_slow) / sma_slow.replace(0.0, np.nan)).to_numpy(dtype=float)
    n = len(spread)
    pos = np.zeros(n, dtype=float)
    current = 0.0

    for t in range(n):
        val = spread[t]
        if np.isnan(val):
            current = 0.0
        elif val > min_spread_pct:
            current = 1.0
        elif val < -min_spread_pct:
            current = -1.0 if allow_short else 0.0
        pos[t] = current

    return pd.Series(pos, index=df.index, name="position")


def mean_reversion_signal(
    df: pd.DataFrame,
    window: int = 20,
    entry_z: float = 1.5,
    exit_z: float = 0.0,
    allow_short: bool = False,
) -> pd.Series:
    """Rolling Z-score mean-reversion signal with stateful entry/exit hysteresis.

    Computes ``z_t = (close_t - SMA_t) / STD_t`` using a strictly causal rolling window:
    - Enters long (+1) when ``z_t <= -entry_z`` (oversold).
    - Exits long to flat (0) when ``z_t >= -exit_z`` (reverted toward rolling mean).
    - If ``allow_short=True``, enters short (-1) when ``z_t >= entry_z`` and
      exits short when ``z_t <= exit_z``.
    """
    if window < 5:
        raise ValueError(f"window must be >= 5, got {window}")
    if entry_z <= 0.0:
        raise ValueError(f"entry_z must be > 0, got {entry_z}")
    if exit_z >= entry_z:
        raise ValueError(f"exit_z ({exit_z}) must be strictly less than entry_z ({entry_z})")

    close = df["close"].astype(float)
    rolling_mean = close.rolling(window=window, min_periods=window).mean()
    rolling_std = close.rolling(window=window, min_periods=window).std(ddof=1)
    zscore = ((close - rolling_mean) / rolling_std.replace(0.0, np.nan)).to_numpy(dtype=float)

    n = len(zscore)
    pos = np.zeros(n, dtype=float)
    current = 0.0

    for t in range(n):
        z = zscore[t]
        if np.isnan(z):
            current = 0.0
        elif current == 0.0:
            if z <= -entry_z:
                current = 1.0
            elif allow_short and z >= entry_z:
                current = -1.0
        elif current == 1.0:
            if z >= -exit_z:
                current = -1.0 if (allow_short and z >= entry_z) else 0.0
        elif current == -1.0:
            if z <= exit_z:
                current = 1.0 if z <= -entry_z else 0.0
        pos[t] = current

    return pd.Series(pos, index=df.index, name="position")


def breakout_signal(
    df: pd.DataFrame,
    entry_window: int = 20,
    exit_window: int = 10,
    vol_multiplier: float = 0.0,
    allow_short: bool = False,
) -> pd.Series:
    """Donchian channel / volatility-confirmed breakout signal.

    Compares ``close_t`` strictly against the PRIOR ``entry_window`` highs and
    PRIOR ``exit_window`` lows (shifted by 1 bar inside the rolling bound so
    today's bar never enters its own reference channel):
    - Enters long (+1) when ``close_t > prev_high + vol_multiplier * rolling_std_t``.
    - Exits long when ``close_t < prev_low``.
    """
    if entry_window < 5 or exit_window < 2:
        raise ValueError(
            f"Invalid breakout windows: entry_window={entry_window}, exit_window={exit_window}"
        )
    if exit_window > entry_window:
        raise ValueError(
            f"exit_window ({exit_window}) must be <= entry_window ({entry_window})"
        )
    if vol_multiplier < 0.0:
        raise ValueError(f"vol_multiplier must be >= 0, got {vol_multiplier}")

    close = df["close"].astype(float)
    high = df["high"].astype(float) if "high" in df.columns else close
    low = df["low"].astype(float) if "low" in df.columns else close

    prev_entry_high = high.shift(1).rolling(window=entry_window, min_periods=entry_window).max()
    prev_entry_low = low.shift(1).rolling(window=entry_window, min_periods=entry_window).min()
    prev_exit_low = low.shift(1).rolling(window=exit_window, min_periods=exit_window).min()
    prev_exit_high = high.shift(1).rolling(window=exit_window, min_periods=exit_window).max()

    # Rolling price volatility buffer (causal, up to t-1)
    price_std = close.shift(1).rolling(window=entry_window, min_periods=entry_window).std(ddof=1)
    buffer = (vol_multiplier * price_std).fillna(0.0)

    c_arr = close.to_numpy(dtype=float)
    eh_arr = (prev_entry_high + buffer).to_numpy(dtype=float)
    el_arr = (prev_entry_low - buffer).to_numpy(dtype=float)
    xl_arr = prev_exit_low.to_numpy(dtype=float)
    xh_arr = prev_exit_high.to_numpy(dtype=float)

    n = len(c_arr)
    pos = np.zeros(n, dtype=float)
    current = 0.0

    for t in range(n):
        if np.isnan(eh_arr[t]) or np.isnan(xl_arr[t]):
            current = 0.0
        elif current == 0.0:
            if c_arr[t] > eh_arr[t]:
                current = 1.0
            elif allow_short and c_arr[t] < el_arr[t]:
                current = -1.0
        elif current == 1.0:
            if c_arr[t] < xl_arr[t]:
                current = -1.0 if (allow_short and c_arr[t] < el_arr[t]) else 0.0
        elif current == -1.0:
            if c_arr[t] > xh_arr[t]:
                current = 1.0 if c_arr[t] > eh_arr[t] else 0.0
        pos[t] = current

    return pd.Series(pos, index=df.index, name="position")


def rsi_reversion_signal(
    df: pd.DataFrame,
    window: int = 14,
    oversold: float = 30.0,
    overbought: float = 70.0,
    trend_window: int = 0,
    allow_short: bool = False,
) -> pd.Series:
    """Wilder-smoothed RSI mean-reversion signal with optional SMA trend filter.

    - Enters long (+1) when ``RSI_t <= oversold`` (and, if ``trend_window > 0``,
      ``close_t > SMA(trend_window)`` to avoid catching falling knives).
    - Exits long when ``RSI_t >= overbought``.
    """
    if window < 2:
        raise ValueError(f"RSI window must be >= 2, got {window}")
    if not (5.0 <= oversold < overbought <= 95.0):
        raise ValueError(
            f"Require 5 <= oversold < overbought <= 95, got oversold={oversold}, overbought={overbought}"
        )
    if trend_window < 0:
        raise ValueError(f"trend_window must be >= 0, got {trend_window}")

    close = df["close"].astype(float)
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    # Wilder's smoothing via causal EWM (alpha = 1 / window)
    avg_gain = gain.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    # Handle zero-loss edge case (pure upward sequence -> RSI = 100)
    rsi = rsi.where(avg_loss > 0.0, np.where(avg_gain > 0.0, 100.0, 50.0))
    rsi = rsi.where(avg_gain.notna() & avg_loss.notna(), np.nan)

    trend_ok = np.ones(len(close), dtype=bool)
    if trend_window > 0:
        trend_sma = close.rolling(window=trend_window, min_periods=trend_window).mean()
        trend_ok = (close > trend_sma).fillna(False).to_numpy(dtype=bool)

    rsi_arr = rsi.to_numpy(dtype=float)
    n = len(rsi_arr)
    pos = np.zeros(n, dtype=float)
    current = 0.0

    for t in range(n):
        val = rsi_arr[t]
        if np.isnan(val):
            current = 0.0
        elif current == 0.0:
            if val <= oversold and trend_ok[t]:
                current = 1.0
            elif allow_short and val >= overbought:
                current = -1.0
        elif current == 1.0:
            if val >= overbought:
                current = -1.0 if allow_short else 0.0
        elif current == -1.0:
            if val <= oversold:
                current = 1.0 if trend_ok[t] else 0.0
        pos[t] = current

    return pd.Series(pos, index=df.index, name="position")


STRATEGY_REGISTRY = {
    "momentum": momentum_signal,
    "mean_reversion": mean_reversion_signal,
    "breakout": breakout_signal,
    "rsi_reversion": rsi_reversion_signal,
}


def generate_signal(
    df: pd.DataFrame,
    strategy_type: str,
    parameters: dict[str, Any],
    allow_short: bool = False,
) -> pd.Series:
    """Dispatch signal generation by ``strategy_type`` and parameter dictionary."""
    if strategy_type not in STRATEGY_REGISTRY:
        raise ValueError(
            f"Unknown strategy_type '{strategy_type}'. "
            f"Supported: {sorted(STRATEGY_REGISTRY.keys())}"
        )
    fn = STRATEGY_REGISTRY[strategy_type]
    return fn(df, allow_short=allow_short, **parameters)
