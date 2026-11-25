"""Step 6: Visualize Equity Curve, Drawdown, and In-Sample vs Out-of-Sample stability.

Generates two publication-ready figures saved under ``logs/plots/``:
    1. ``equity_and_drawdown.png`` — Full-timeline Equity Curve & Underwater
       Drawdown of the selected strategy vs. Buy & Hold, clearly marking the
       chronological In-Sample (70%) / Out-of-Sample (30%) boundary.
    2. ``is_vs_oos_sharpe.png`` — Grouped bar chart comparing In-Sample vs.
       Out-of-Sample Sharpe Ratios across Top-K candidates and Buy & Hold.

Usage:
    python scripts/plot_results.py
    python scripts/plot_results.py --iteration 1 --output-dir logs/plots
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tools.backtest import DEFAULT_FEE_RATE, DEFAULT_SLIPPAGE_RATE, evaluate_strategy
from tools.data_loader import compute_split_info, load_prices


def plot_equity_and_drawdown(
    data_path: str | Path = "data/prices.csv",
    experiments_csv: str | Path = "logs/experiments.csv",
    output_dir: str | Path = "logs/plots",
    iteration: int | None = None,
    target: str | None = None,
    train_ratio: float = 0.70,
    fee_rate: float = DEFAULT_FEE_RATE,
    slippage_rate: float = DEFAULT_SLIPPAGE_RATE,
    allow_short: bool = False,
) -> Path:
    """Plot continuous Equity Curve and Drawdown with IS/OOS split marker."""
    exp_df = pd.read_csv(experiments_csv)
    if exp_df.empty:
        raise ValueError("experiments.csv is empty.")

    if iteration is not None:
        match = exp_df[exp_df["iteration"] == iteration]
        if match.empty:
            raise ValueError(f"Iteration {iteration} not found in {experiments_csv}.")
        selected = match.iloc[0]
    else:
        active = exp_df[exp_df["trade_count"] > 0]
        pool = active if not active.empty else exp_df
        selected = pool.sort_values("sharpe_ratio", ascending=False).iloc[0]

    eval_target = target or str(selected["target"])
    stype = str(selected["strategy_type"])
    params = json.loads(str(selected["parameters_json"]))
    name = str(selected["hypothesis_name"])

    full_df = load_prices(data_path)
    split_info = compute_split_info(full_df, train_ratio=train_ratio)

    full_res = evaluate_strategy(
        full_df,
        strategy_type=stype,
        parameters=params,
        target=eval_target,
        allow_short=allow_short,
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
    )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = out_dir / "equity_and_drawdown.png"

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 7.5), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.0]}
    )

    # Top panel: Equity Curve
    ax1.plot(
        full_res.equity_curve.index,
        full_res.equity_curve.values,
        label=f"Strategy: {name} ({stype})",
        color="#1f77b4",
        linewidth=2.0,
    )
    ax1.plot(
        full_res.benchmark_curve.index,
        full_res.benchmark_curve.values,
        label=f"Buy & Hold ({eval_target})",
        color="#7f7f7f",
        linestyle="--",
        linewidth=1.5,
    )
    ax1.axvline(
        split_info.out_of_sample_start,
        color="#d62728",
        linestyle=":",
        linewidth=1.8,
        label="In-Sample / Out-of-Sample Split",
    )
    ax1.axvspan(
        split_info.out_of_sample_start,
        full_res.equity_curve.index.max(),
        color="#f7d7d7",
        alpha=0.25,
        label="Out-of-Sample Region (30%)",
    )
    ax1.set_title(
        f"Equity Curve & Drawdown — {name} on {eval_target}\n"
        f"Full-Period Sharpe: {full_res.metrics['sharpe_ratio']:+.2f} "
        f"(B&H {full_res.metrics['benchmark_sharpe']:+.2f}) | "
        f"MaxDD: {full_res.metrics['max_drawdown']:.1%} | Trades: {full_res.metrics['trade_count']}",
        fontsize=11,
        fontweight="bold",
    )
    ax1.set_ylabel("Growth of 1.0 VND")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left", fontsize=9)

    # Bottom panel: Underwater Drawdown
    ax2.fill_between(
        full_res.drawdown_curve.index,
        full_res.drawdown_curve.values * 100.0,
        0.0,
        color="#1f77b4",
        alpha=0.35,
        label="Strategy Drawdown (%)",
    )
    ax2.plot(
        full_res.benchmark_drawdown.index,
        full_res.benchmark_drawdown.values * 100.0,
        color="#7f7f7f",
        linestyle="--",
        linewidth=1.2,
        label="Buy & Hold Drawdown (%)",
    )
    ax2.axvline(split_info.out_of_sample_start, color="#d62728", linestyle=":", linewidth=1.8)
    ax2.axvspan(
        split_info.out_of_sample_start,
        full_res.drawdown_curve.index.max(),
        color="#f7d7d7",
        alpha=0.25,
    )
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="lower left", fontsize=9)

    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Saved equity & drawdown chart to: {fig_path}")
    return fig_path


def plot_is_vs_oos_sharpe(
    oos_csv: str | Path = "logs/oos_evaluation.csv",
    output_dir: str | Path = "logs/plots",
) -> Path | None:
    """Plot In-Sample vs Out-of-Sample Sharpe Ratios from oos_evaluation.csv."""
    oos_path = Path(oos_csv)
    if not oos_path.exists():
        print(f"[INFO] {oos_path} not found — skipping IS vs OOS bar chart. Run `python scripts/run_oos_eval.py` first.")
        return None

    df = pd.read_csv(oos_path)
    if df.empty:
        return None

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = out_dir / "is_vs_oos_sharpe.png"

    labels = [f"#{int(r['rank_is'])} {r['hypothesis_name']}" for _, r in df.iterrows()]
    is_vals = df["is_sharpe"].to_numpy(dtype=float)
    oos_vals = df["oos_sharpe"].to_numpy(dtype=float)
    bh_oos = float(df["oos_benchmark_sharpe"].iloc[0])

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5.2))
    bars_is = ax.bar(x - width / 2, is_vals, width, label="In-Sample Sharpe (70%)", color="#2a7ab0")
    bars_oos = ax.bar(x + width / 2, oos_vals, width, label="Out-of-Sample Sharpe (30%)", color="#e07a3f")

    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.axhline(
        bh_oos,
        color="#555555",
        linestyle="--",
        linewidth=1.3,
        label=f"OOS Buy & Hold Sharpe ({bh_oos:+.2f})",
    )

    for bar in list(bars_is) + list(bars_oos):
        h = bar.get_height()
        offset = 0.04 if h >= 0 else -0.09
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + offset,
            f"{h:+.2f}",
            ha="center",
            va="bottom" if h >= 0 else "top",
            fontsize=8.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Annualized Sharpe Ratio")
    ax.set_title("Generalization Check: In-Sample vs. Out-of-Sample Sharpe Ratio", fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Saved IS vs OOS comparison chart to: {fig_path}")
    return fig_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate matplotlib charts for backtest & OOS results.")
    parser.add_argument("--data", type=Path, default=Path("data/prices.csv"))
    parser.add_argument("--experiments", type=Path, default=Path("logs/experiments.csv"))
    parser.add_argument("--oos-csv", type=Path, default=Path("logs/oos_evaluation.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("logs/plots"))
    parser.add_argument("--iteration", type=int, default=None, help="Specific iteration to plot (default: best IS Sharpe).")
    parser.add_argument("--target", type=str, default=None)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--fee-rate", type=float, default=DEFAULT_FEE_RATE)
    parser.add_argument("--slippage-rate", type=float, default=DEFAULT_SLIPPAGE_RATE)
    parser.add_argument("--allow-short", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_equity_and_drawdown(
        data_path=args.data,
        experiments_csv=args.experiments,
        output_dir=args.output_dir,
        iteration=args.iteration,
        target=args.target,
        train_ratio=args.train_ratio,
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
        allow_short=args.allow_short,
    )
    plot_is_vs_oos_sharpe(
        oos_csv=args.oos_csv,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
