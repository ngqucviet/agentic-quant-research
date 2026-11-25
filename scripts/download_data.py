"""Download or synthesize daily OHLCV data for 5 major Vietnamese banks.

Universe (HOSE):
    - VCB.VN : Vietcombank (Big4)
    - BID.VN : BIDV (Big4)
    - CTG.VN : VietinBank (Big4)
    - MBB.VN : MB Bank
    - ACB.VN : ACB

Note: Agribank (the 4th state-owned Big4 bank) is unlisted, so the 3 listed
Big4 banks plus MBB and ACB form a clean 5-bank Vietnamese equity universe.

Usage:
    python scripts/download_data.py --years 5
    python scripts/download_data.py --synthetic --years 5 --seed 42
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

DEFAULT_TICKERS: tuple[str, ...] = (
    "VCB.VN",
    "BID.VN",
    "CTG.VN",
    "MBB.VN",
    "ACB.VN",
)

DEFAULT_OUTPUT_PATH = Path("data/prices.csv")


def download_from_yfinance(
    tickers: Sequence[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Download split/dividend-adjusted daily OHLCV data from Yahoo Finance.

    Using ``auto_adjust=True`` is essential for Vietnamese banking stocks
    because they frequently issue 15%-35% stock dividends; unadjusted close
    prices would create spurious overnight crashes in backtests.
    """
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError(
            "yfinance is not installed. Run `pip install yfinance` or use `--synthetic`."
        ) from exc

    frames: list[pd.DataFrame] = []
    for ticker in tickers:
        raw = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            interval="1d",
            auto_adjust=True,
            progress=False,
        )
        if raw is None or raw.empty:
            print(f"[WARN] No data returned for {ticker}; skipping.")
            continue

        # Handle MultiIndex columns returned by newer versions of yfinance.
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [col[0] for col in raw.columns]

        df = raw.reset_index().rename(
            columns={
                "Date": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )
        df["ticker"] = ticker
        required = ["date", "ticker", "open", "high", "low", "close", "volume"]
        df = df[required].copy()
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        df = df.dropna(subset=["close"])
        df = df[df["close"] > 0]
        frames.append(df)
        print(f"[OK] Downloaded {ticker}: {len(df)} rows ({df['date'].min()} -> {df['date'].max()})")

    if not frames:
        raise RuntimeError(
            "Failed to download data for all tickers. Check internet access or run with `--synthetic`."
        )

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["ticker", "date"]).drop_duplicates(subset=["ticker", "date"])
    return combined.reset_index(drop=True)


def generate_synthetic_bank_data(
    tickers: Sequence[str],
    years: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate realistic, correlated, regime-switching OHLCV data for offline testing.

    Models a shared Vietnamese banking sector factor plus idiosyncratic noise
    and distinct market regimes (trending bull, high-vol correction, sideways
    mean-reverting consolidation, and late-sample regime shift) so that
    in-sample vs. out-of-sample behavior mirrors real equity research.
    """
    rng = np.random.default_rng(seed)
    n_days = int(years * 252)
    dates = pd.bdate_range(end=pd.Timestamp(date.today()), periods=n_days)

    # Regime schedule (daily drift mu, daily sector vol sigma)
    # Intentionally shifts regime in the final 30% (out-of-sample) to test generalization.
    regimes = [
        (0.00065, 0.012, 0.25),   # Bull trend
        (-0.00080, 0.022, 0.15),  # Drawdown / shock
        (0.00015, 0.013, 0.30),   # Sideways / choppy recovery (end of 70% IS)
        (0.00045, 0.017, 0.15),   # OOS volatile rally
        (-0.00010, 0.016, 0.15),  # OOS mean-reverting chop
    ]

    sector_returns = np.empty(n_days, dtype=float)
    idx = 0
    for i, (mu, sigma, frac) in enumerate(regimes):
        count = n_days - idx if i == len(regimes) - 1 else int(round(n_days * frac))
        # Mild AR(1) autocorrelation in bull regimes, slight negative AR(1) in choppy regimes
        ar_coef = 0.08 if mu > 0.0003 else -0.06
        shocks = rng.normal(0.0, sigma, size=count)
        for t in range(count):
            prev = sector_returns[idx + t - 1] if (idx + t) > 0 else 0.0
            sector_returns[idx + t] = mu + ar_coef * prev + shocks[t]
        idx += count

    # Realistic starting prices (VND) and beta profiles per bank
    bank_profiles: dict[str, dict[str, float]] = {
        "VCB.VN": {"p0": 62_000.0, "beta": 0.85, "idio_vol": 0.009, "alpha": 0.00010, "base_vol": 2_200_000},
        "BID.VN": {"p0": 32_000.0, "beta": 1.05, "idio_vol": 0.012, "alpha": 0.00005, "base_vol": 3_100_000},
        "CTG.VN": {"p0": 24_000.0, "beta": 1.15, "idio_vol": 0.013, "alpha": 0.00004, "base_vol": 6_500_000},
        "MBB.VN": {"p0": 16_500.0, "beta": 1.10, "idio_vol": 0.012, "alpha": 0.00012, "base_vol": 14_000_000},
        "ACB.VN": {"p0": 17_500.0, "beta": 0.95, "idio_vol": 0.010, "alpha": 0.00011, "base_vol": 8_500_000},
    }

    frames: list[pd.DataFrame] = []
    for ticker in tickers:
        prof = bank_profiles.get(
            ticker,
            {"p0": 25_000.0, "beta": 1.0, "idio_vol": 0.012, "alpha": 0.00005, "base_vol": 5_000_000},
        )
        idio = rng.normal(prof["alpha"], prof["idio_vol"], size=n_days)
        daily_ret = prof["beta"] * sector_returns + idio
        # HOSE daily price limit is +/-7%
        daily_ret = np.clip(daily_ret, -0.07, 0.07)

        close = prof["p0"] * np.cumprod(1.0 + daily_ret)
        open_ = np.empty_like(close)
        open_[0] = prof["p0"]
        open_[1:] = close[:-1] * (1.0 + rng.normal(0.0, 0.003, size=n_days - 1))

        intraday_spread = np.abs(rng.normal(0.008, 0.005, size=n_days))
        high = np.maximum(open_, close) * (1.0 + intraday_spread)
        low = np.minimum(open_, close) * (1.0 - intraday_spread)
        low = np.maximum(low, 100.0)

        vol_mult = np.exp(rng.normal(0.0, 0.35, size=n_days)) * (1.0 + 15.0 * np.abs(daily_ret))
        volume = np.round(prof["base_vol"] * vol_mult).astype(int)

        frames.append(
            pd.DataFrame(
                {
                    "date": dates.strftime("%Y-%m-%d"),
                    "ticker": ticker,
                    "open": np.round(open_, 2),
                    "high": np.round(high, 2),
                    "low": np.round(low, 2),
                    "close": np.round(close, 2),
                    "volume": volume,
                }
            )
        )

    combined = pd.concat(frames, ignore_index=True)
    return combined.sort_values(["ticker", "date"]).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download (or synthesize) historical daily prices for 5 Vietnamese banks."
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=list(DEFAULT_TICKERS),
        help="Tickers to fetch (default: VCB.VN BID.VN CTG.VN MBB.VN ACB.VN).",
    )
    parser.add_argument(
        "--years",
        type=int,
        default=5,
        help="Number of historical years to fetch or synthesize (default: 5).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output CSV path (default: data/prices.csv).",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate synthetic correlated regime-switching data offline (no network required).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed when using --synthetic (default: 42).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    end_dt = date.today()
    start_dt = end_dt - timedelta(days=int(args.years * 365.25))

    if args.synthetic:
        print(f"Generating synthetic data ({args.years}y, seed={args.seed}) for {args.tickers}...")
        df = generate_synthetic_bank_data(args.tickers, years=args.years, seed=args.seed)
    else:
        print(f"Downloading from Yahoo Finance ({start_dt} -> {end_dt}) for {args.tickers}...")
        df = download_from_yfinance(args.tickers, start_dt.isoformat(), end_dt.isoformat())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Saved {len(df):,} rows across {df['ticker'].nunique()} tickers to {args.output}")


if __name__ == "__main__":
    main()
