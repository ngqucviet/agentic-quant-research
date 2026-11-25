"""Step 5: Independent Out-of-Sample (OOS) evaluation and overfitting audit.

Selects the Top-K strategies discovered during the In-Sample agentic loop
(from ``logs/experiments.csv``) and evaluates them on the held-out 30%
Out-of-Sample period.

Leak-free warm-up handling:
    Causal signals are computed on the full chronological price series (each
    bar ``t`` strictly uses observations ``<= t``), and performance metrics are
    sliced strictly to ``[out_of_sample_start, out_of_sample_end]``. This gives
    rolling indicators proper warm-up without leaking any future information.

Usage:
    python scripts/run_oos_eval.py --top-k 3
    python scripts/run_oos_eval.py --top-k 3 --target ALL
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from tools.backtest import DEFAULT_FEE_RATE, DEFAULT_SLIPPAGE_RATE, evaluate_strategy
from tools.data_loader import compute_split_info, load_prices


def evaluate_top_k_oos(
    data_path: str | Path = "data/prices.csv",
    experiments_csv: str | Path = "logs/experiments.csv",
    output_csv: str | Path = "logs/oos_evaluation.csv",
    top_k: int = 3,
    target: str | None = None,
    train_ratio: float = 0.70,
    fee_rate: float = DEFAULT_FEE_RATE,
    slippage_rate: float = DEFAULT_SLIPPAGE_RATE,
    allow_short: bool = False,
) -> pd.DataFrame:
    """Evaluate Top-K In-Sample candidates on the untouched Out-of-Sample window."""
    exp_path = Path(experiments_csv)
    if not exp_path.exists():
        raise FileNotFoundError(
            f"Experiment log not found at {exp_path}. Run `python -m agent.loop` first."
        )

    exp_df = pd.read_csv(exp_path)
    if exp_df.empty:
        raise ValueError(f"Experiment log at {exp_path} is empty.")

    eval_target = target or str(exp_df["target"].iloc[0])
    full_df = load_prices(data_path)
    split_info = compute_split_info(full_df, train_ratio=train_ratio)

    # Rank strictly by In-Sample Sharpe among strategies that actually traded
    active_df = exp_df[exp_df["trade_count"] > 0].copy()
    if active_df.empty:
        active_df = exp_df.copy()

    top_candidates = (
        active_df.sort_values("sharpe_ratio", ascending=False)
        .drop_duplicates(subset=["strategy_type", "parameters_json"])
        .head(top_k)
    )

    rows: list[dict[str, Any]] = []
    per_bank_tables: dict[str, dict[str, dict[str, float | int]]] = {}

    for rank, (_, rec) in enumerate(top_candidates.iterrows(), start=1):
        params = json.loads(str(rec["parameters_json"]))
        stype = str(rec["strategy_type"])
        name = str(rec["hypothesis_name"])

        # Re-verify In-Sample metrics deterministically under current cost settings
        is_res = evaluate_strategy(
            full_df[full_df["date"] <= split_info.in_sample_end].copy(),
            strategy_type=stype,
            parameters=params,
            target=eval_target,
            allow_short=allow_short,
            fee_rate=fee_rate,
            slippage_rate=slippage_rate,
        )

        # Evaluate Out-of-Sample with causal warm-up
        oos_res = evaluate_strategy(
            full_df,
            strategy_type=stype,
            parameters=params,
            target=eval_target,
            allow_short=allow_short,
            fee_rate=fee_rate,
            slippage_rate=slippage_rate,
            eval_start=split_info.out_of_sample_start,
            eval_end=split_info.out_of_sample_end,
        )

        is_sharpe = float(is_res.metrics["sharpe_ratio"])
        oos_sharpe = float(oos_res.metrics["sharpe_ratio"])
        sharpe_delta = oos_sharpe - is_sharpe
        sharpe_retention = (oos_sharpe / is_sharpe) if abs(is_sharpe) > 1e-6 else 0.0

        row: dict[str, Any] = {
            "rank_is": rank,
            "iteration": int(rec["iteration"]),
            "hypothesis_name": name,
            "strategy_type": stype,
            "parameters_json": str(rec["parameters_json"]),
            "target": eval_target,
            "is_sharpe": round(is_sharpe, 4),
            "oos_sharpe": round(oos_sharpe, 4),
            "sharpe_delta": round(sharpe_delta, 4),
            "sharpe_retention": round(sharpe_retention, 3),
            "is_return": float(is_res.metrics["cumulative_return"]),
            "oos_return": float(oos_res.metrics["cumulative_return"]),
            "oos_benchmark_return": float(oos_res.metrics["benchmark_return"]),
            "oos_benchmark_sharpe": float(oos_res.metrics["benchmark_sharpe"]),
            "is_max_dd": float(is_res.metrics["max_drawdown"]),
            "oos_max_dd": float(oos_res.metrics["max_drawdown"]),
            "oos_benchmark_max_dd": float(oos_res.metrics["benchmark_max_drawdown"]),
            "oos_trades": int(oos_res.metrics["trade_count"]),
            "oos_fee_drag": float(oos_res.metrics["fee_drag"]),
            "oos_exposure": float(oos_res.metrics["exposure"]),
            "oos_positive_sharpe_banks": oos_res.metrics.get("positive_sharpe_banks"),
            "total_banks": oos_res.metrics.get("total_banks"),
        }
        rows.append(row)
        if oos_res.per_ticker_metrics:
            per_bank_tables[name] = oos_res.per_ticker_metrics

    result_df = pd.DataFrame(rows)
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(out_path, index=False)

    # Print formatted audit report
    print("=" * 92)
    print("OUT-OF-SAMPLE GENERALIZATION & OVERFITTING AUDIT")
    print("=" * 92)
    print(f"Target       : {eval_target}")
    print(
        f"In-Sample    : {split_info.in_sample_start.date()} -> {split_info.in_sample_end.date()} "
        f"({split_info.n_in_sample_dates} days)"
    )
    print(
        f"Out-of-Sample: {split_info.out_of_sample_start.date()} -> {split_info.out_of_sample_end.date()} "
        f"({split_info.n_out_of_sample_dates} days)"
    )
    print("-" * 92)

    print("\n### Markdown Summary Table (copy into README.md)\n")
    print(
        "| Rank | Strategy | Family | IS Sharpe | OOS Sharpe | OOS B&H Sharpe | "
        "IS Ret | OOS Ret | OOS MaxDD | OOS Trades |"
    )
    print("| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for _, r in result_df.iterrows():
        print(
            f"| {int(r['rank_is'])} | {r['hypothesis_name']} | `{r['strategy_type']}` | "
            f"{r['is_sharpe']:+.2f} | {r['oos_sharpe']:+.2f} | {r['oos_benchmark_sharpe']:+.2f} | "
            f"{r['is_return']:+.1%} | {r['oos_return']:+.1%} | {r['oos_max_dd']:.1%} | {int(r['oos_trades'])} |"
        )

    if per_bank_tables and len(next(iter(per_bank_tables.values()))) > 1:
        best_name = str(result_df.iloc[0]["hypothesis_name"])
        print(f"\n### Cross-Bank OOS Breakdown for #1 Strategy: {best_name}\n")
        print("| Bank | OOS Sharpe | OOS B&H Sharpe | OOS Return | OOS MaxDD | Trades |")
        print("| :--- | :---: | :---: | :---: | :---: | :---: |")
        for ticker, m in sorted(per_bank_tables[best_name].items()):
            print(
                f"| `{ticker}` | {float(m['sharpe_ratio']):+.2f} | {float(m['benchmark_sharpe']):+.2f} | "
                f"{float(m['cumulative_return']):+.1%} | {float(m['max_drawdown']):.1%} | {int(m['trade_count'])} |"
            )

    print(f"\nSaved Out-of-Sample audit to: {out_path}")
    return result_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Top-K In-Sample strategies on Out-of-Sample data.")
    parser.add_argument("--data", type=Path, default=Path("data/prices.csv"))
    parser.add_argument("--experiments", type=Path, default=Path("logs/experiments.csv"))
    parser.add_argument("--output", type=Path, default=Path("logs/oos_evaluation.csv"))
    parser.add_argument("--top-k", type=int, default=3, help="Number of top In-Sample strategies to evaluate.")
    parser.add_argument("--target", type=str, default=None, help="Override target ticker or 'ALL'.")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--fee-rate", type=float, default=DEFAULT_FEE_RATE)
    parser.add_argument("--slippage-rate", type=float, default=DEFAULT_SLIPPAGE_RATE)
    parser.add_argument("--allow-short", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate_top_k_oos(
        data_path=args.data,
        experiments_csv=args.experiments,
        output_csv=args.output,
        top_k=args.top_k,
        target=args.target,
        train_ratio=args.train_ratio,
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
        allow_short=args.allow_short,
    )


if __name__ == "__main__":
    main()
