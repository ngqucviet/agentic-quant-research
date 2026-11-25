"""Data loading, chronological train/test splitting, and in-sample summary utilities.

All splits are strictly chronological by date so there is zero leakage between
the In-Sample (research/tuning) split and the Out-of-Sample (holdout) split.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REQUIRED_COLUMNS: tuple[str, ...] = (
    "date",
    "ticker",
    "open",
    "high",
    "low",
    "close",
    "volume",
)

VN_BANK_UNIVERSE: tuple[str, ...] = (
    "VCB.VN",
    "BID.VN",
    "CTG.VN",
    "MBB.VN",
    "ACB.VN",
)


@dataclass(frozen=True)
class SplitInfo:
    """Metadata describing a chronological In-Sample / Out-of-Sample split."""

    train_ratio: float
    in_sample_start: pd.Timestamp
    in_sample_end: pd.Timestamp
    out_of_sample_start: pd.Timestamp
    out_of_sample_end: pd.Timestamp
    n_in_sample_dates: int
    n_out_of_sample_dates: int


def load_prices(csv_path: str | Path = "data/prices.csv") -> pd.DataFrame:
    """Load and validate long-format OHLCV price data from CSV.

    Returns a DataFrame sorted by ``(ticker, date)`` with ``date`` parsed as
    ``datetime64[ns]``. Supports legacy single-asset CSVs without a ``ticker``
    column by assigning ``ticker = "ASSET"``.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Price file not found at {path}. "
            "Run `python scripts/download_data.py --years 5` (or add `--synthetic` for offline mode)."
        )

    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    if "ticker" not in df.columns:
        df["ticker"] = "ASSET"

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    df = df[list(REQUIRED_COLUMNS)].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].astype(str).str.strip()

    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["date", "ticker", "close"])
    df = df[df["close"] > 0]
    df = df.sort_values(["ticker", "date"]).drop_duplicates(subset=["ticker", "date"])
    return df.reset_index(drop=True)


def get_available_tickers(df: pd.DataFrame) -> list[str]:
    """Return sorted unique ticker symbols present in the dataset."""
    return sorted(df["ticker"].unique().tolist())


def extract_ticker_df(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Return a single ticker's OHLCV DataFrame indexed by ``date``."""
    sub = df[df["ticker"] == ticker].copy()
    if sub.empty:
        available = get_available_tickers(df)
        raise ValueError(f"Ticker '{ticker}' not found. Available tickers: {available}")
    sub = sub.sort_values("date").set_index("date")
    return sub[["open", "high", "low", "close", "volume"]]


def compute_split_info(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
) -> SplitInfo:
    """Compute a single chronological cutoff date across the dataset.

    Splitting by unique calendar/trading dates guarantees that all 5 banks
    share the exact same In-Sample / Out-of-Sample boundary.
    """
    if not (0.5 <= train_ratio <= 0.9):
        raise ValueError(f"train_ratio must be between 0.5 and 0.9, got {train_ratio}")

    unique_dates = pd.Index(sorted(df["date"].unique()))
    if len(unique_dates) < 100:
        raise ValueError(
            f"Need at least 100 trading days for a meaningful IS/OOS split, got {len(unique_dates)}."
        )

    split_idx = int(len(unique_dates) * train_ratio)
    split_idx = max(1, min(split_idx, len(unique_dates) - 1))

    is_dates = unique_dates[:split_idx]
    oos_dates = unique_dates[split_idx:]

    return SplitInfo(
        train_ratio=train_ratio,
        in_sample_start=pd.Timestamp(is_dates[0]),
        in_sample_end=pd.Timestamp(is_dates[-1]),
        out_of_sample_start=pd.Timestamp(oos_dates[0]),
        out_of_sample_end=pd.Timestamp(oos_dates[-1]),
        n_in_sample_dates=len(is_dates),
        n_out_of_sample_dates=len(oos_dates),
    )


def split_in_out_sample(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
) -> tuple[pd.DataFrame, pd.DataFrame, SplitInfo]:
    """Split long-format price DataFrame into chronological In-Sample and Out-of-Sample sets."""
    info = compute_split_info(df, train_ratio=train_ratio)
    in_sample = df[df["date"] <= info.in_sample_end].copy().reset_index(drop=True)
    out_of_sample = df[df["date"] >= info.out_of_sample_start].copy().reset_index(drop=True)
    return in_sample, out_of_sample, info


def summarize_in_sample(
    in_sample_df: pd.DataFrame,
    target: str = "ALL",
    annualization: int = 252,
) -> dict[str, Any]:
    """Compute compact descriptive statistics strictly on the In-Sample split.

    This dictionary is passed to the LLM agent during the Propose step so the
    agent understands volatility, trend, drawdown, and cross-bank correlation
    without ever seeing Out-of-Sample observations.
    """
    tickers = get_available_tickers(in_sample_df) if target == "ALL" else [target]
    per_ticker: dict[str, dict[str, float | int | str]] = {}
    return_series: dict[str, pd.Series] = {}

    for ticker in tickers:
        sub = extract_ticker_df(in_sample_df, ticker)
        rets = sub["close"].pct_change().dropna()
        return_series[ticker] = rets

        cum_ret = float((sub["close"].iloc[-1] / sub["close"].iloc[0]) - 1.0)
        ann_vol = float(rets.std(ddof=1) * np.sqrt(annualization)) if len(rets) > 1 else 0.0
        ann_ret = float(rets.mean() * annualization) if len(rets) > 0 else 0.0
        sharpe = float(ann_ret / ann_vol) if ann_vol > 1e-12 else 0.0

        equity = sub["close"] / sub["close"].iloc[0]
        max_dd = float(((equity / equity.cummax()) - 1.0).min())
        autocorr_1 = float(rets.autocorr(lag=1)) if len(rets) > 2 else 0.0

        per_ticker[ticker] = {
            "n_days": int(len(sub)),
            "start_date": sub.index.min().strftime("%Y-%m-%d"),
            "end_date": sub.index.max().strftime("%Y-%m-%d"),
            "cumulative_return": round(cum_ret, 4),
            "annualized_volatility": round(ann_vol, 4),
            "buy_and_hold_sharpe": round(sharpe, 3),
            "max_drawdown": round(max_dd, 4),
            "lag1_autocorr": round(autocorr_1, 4),
            "avg_daily_volume": int(sub["volume"].mean()),
        }

    summary: dict[str, Any] = {
        "target": target,
        "tickers": tickers,
        "n_assets": len(tickers),
        "per_ticker_in_sample": per_ticker,
    }

    if len(return_series) > 1:
        ret_df = pd.DataFrame(return_series).dropna()
        if not ret_df.empty:
            corr_matrix = ret_df.corr()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            avg_corr = float(upper.stack().mean())
            ew_rets = ret_df.mean(axis=1)
            ew_equity = (1.0 + ew_rets).cumprod()
            ew_ann_vol = float(ew_rets.std(ddof=1) * np.sqrt(annualization))
            ew_ann_ret = float(ew_rets.mean() * annualization)
            summary["equal_weight_basket_in_sample"] = {
                "n_days": int(len(ew_rets)),
                "cumulative_return": round(float(ew_equity.iloc[-1] - 1.0), 4),
                "annualized_volatility": round(ew_ann_vol, 4),
                "buy_and_hold_sharpe": round(
                    float(ew_ann_ret / ew_ann_vol) if ew_ann_vol > 1e-12 else 0.0, 3
                ),
                "max_drawdown": round(float(((ew_equity / ew_equity.cummax()) - 1.0).min()), 4),
                "avg_pairwise_correlation": round(avg_corr, 3),
            }

    return summary
