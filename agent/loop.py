"""Agentic Quant Research Loop: Propose -> Validate -> Execute -> Critic -> Log.

Strictly restricts the research agent to the chronological In-Sample split
(default: first 70% of trading dates). The remaining 30% Out-of-Sample split
is NEVER exposed to the agent during hypothesis generation or refinement.

Usage:
    # Run offline with built-in heuristic mock researcher across all 5 VN banks:
    python -m agent.loop --provider mock --target ALL --iterations 8

    # Run on a single bank (e.g., MB Bank):
    python -m agent.loop --provider mock --target MBB.VN --iterations 8

    # Switch to a real LLM later (after adding key to .env):
    python -m agent.loop --provider gemini --target ALL --iterations 8
    python -m agent.loop --provider openai --target ALL --iterations 8
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import ValidationError

from agent.providers import ResearchProvider, create_provider
from agent.schemas import (
    ExperimentRecord,
    StrategyHypothesis,
    validate_hypothesis_against_data,
)
from tools.backtest import DEFAULT_FEE_RATE, DEFAULT_SLIPPAGE_RATE, evaluate_strategy
from tools.data_loader import load_prices, split_in_out_sample, summarize_in_sample

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional convenience import
    def load_dotenv() -> bool:
        return False


def _save_records_csv(records: list[ExperimentRecord], output_path: Path) -> None:
    """Persist all experiment records cleanly to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [rec.model_dump() for rec in records]
    pd.DataFrame(rows).to_csv(output_path, index=False)


def _propose_with_validation(
    provider: ResearchProvider,
    dataset_summary: dict[str, Any],
    history: list[ExperimentRecord],
    n_in_sample_days: int,
    seen_signatures: set[str],
    max_retries: int = 3,
) -> tuple[StrategyHypothesis, int]:
    """Run Propose + Validate with automatic self-correction retries."""
    last_error: str | None = None

    for attempt in range(max_retries + 1):
        try:
            hypothesis = provider.propose_hypothesis(
                dataset_summary=dataset_summary,
                history=history,
                validation_error=last_error,
            )
        except (ValidationError, ValueError) as exc:
            last_error = f"Schema validation error: {exc}"
            if attempt == max_retries:
                raise RuntimeError(
                    f"Agent failed schema validation after {max_retries + 1} attempts: {last_error}"
                ) from exc
            continue

        check = validate_hypothesis_against_data(
            hypothesis=hypothesis,
            n_in_sample_days=n_in_sample_days,
            seen_signatures=seen_signatures,
        )
        if check.is_valid:
            return hypothesis, attempt

        last_error = "; ".join(check.errors)
        if attempt == max_retries:
            raise RuntimeError(
                f"Agent failed domain validation after {max_retries + 1} attempts: {last_error}"
            )

    raise RuntimeError("Unreachable validation retry state.")


def run_research_loop(
    data_path: str | Path = "data/prices.csv",
    output_csv: str | Path = "logs/experiments.csv",
    provider: ResearchProvider | None = None,
    provider_name: str = "mock",
    model: str | None = None,
    target: str = "ALL",
    iterations: int = 8,
    train_ratio: float = 0.70,
    fee_rate: float = DEFAULT_FEE_RATE,
    slippage_rate: float = DEFAULT_SLIPPAGE_RATE,
    allow_short: bool = False,
    max_retries: int = 3,
    verbose: bool = True,
) -> list[ExperimentRecord]:
    """Execute the full In-Sample Agentic Quant Research Loop."""
    if iterations < 1:
        raise ValueError("iterations must be >= 1")

    active_provider = provider or create_provider(provider_name, model=model)
    raw_df = load_prices(data_path)
    in_sample_df, _, split_info = split_in_out_sample(raw_df, train_ratio=train_ratio)
    dataset_summary = summarize_in_sample(in_sample_df, target=target)

    if verbose:
        print("=" * 84)
        print("AGENTIC QUANT RESEARCH LOOP (IN-SAMPLE ONLY)")
        print("=" * 84)
        print(f"Provider      : {active_provider.provider_name}")
        print(f"Target        : {target} ({', '.join(dataset_summary['tickers'])})")
        print(
            f"In-Sample     : {split_info.in_sample_start.date()} -> {split_info.in_sample_end.date()} "
            f"({split_info.n_in_sample_dates} trading days)"
        )
        print(
            f"Held-Out OOS  : {split_info.out_of_sample_start.date()} -> {split_info.out_of_sample_end.date()} "
            f"({split_info.n_out_of_sample_dates} trading days — STRICTLY HIDDEN)"
        )
        print(
            f"Market Rules  : {'Long/Short' if allow_short else 'Long-Only (HOSE)'} | "
            f"One-way cost: {(fee_rate + slippage_rate) * 10_000:.1f} bps"
        )
        print("-" * 84)

    history: list[ExperimentRecord] = []
    seen_signatures: set[str] = set()
    output_path = Path(output_csv)

    for idx in range(1, iterations + 1):
        # 1 & 2. Propose + Validate
        hypothesis, retries = _propose_with_validation(
            provider=active_provider,
            dataset_summary=dataset_summary,
            history=history,
            n_in_sample_days=split_info.n_in_sample_dates,
            seen_signatures=seen_signatures,
            max_retries=max_retries,
        )
        seen_signatures.add(hypothesis.signature())

        # 3. Execute deterministic Python backtest STRICTLY on In-Sample data
        result = evaluate_strategy(
            in_sample_df,
            strategy_type=hypothesis.strategy_type,
            parameters=hypothesis.parameters,
            target=target,
            allow_short=allow_short,
            fee_rate=fee_rate,
            slippage_rate=slippage_rate,
        )

        # 4. Evaluate & Critic
        critic = active_provider.critique_result(
            hypothesis=hypothesis,
            metrics=result.metrics,
            history=history,
        )

        # 5. Log incrementally to CSV
        record = ExperimentRecord.from_iteration(
            iteration=idx,
            timestamp_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            provider=active_provider.provider_name,
            target=target,
            hypothesis=hypothesis,
            validation_retries=retries,
            metrics=result.metrics,
            critic=critic,
        )
        history.append(record)
        _save_records_csv(history, output_path)

        if verbose:
            breadth = (
                f" | Banks+: {record.positive_sharpe_banks}/{record.total_banks}"
                if record.positive_sharpe_banks is not None
                else ""
            )
            print(
                f"[Iter {idx:02d}/{iterations:02d}] {record.hypothesis_name} "
                f"({record.strategy_type} {record.parameters_json})"
            )
            print(
                f"   IS Sharpe: {record.sharpe_ratio:+.2f} (B&H {record.benchmark_sharpe:+.2f}) | "
                f"Ret: {record.cumulative_return:+.1%} (B&H {record.benchmark_return:+.1%}) | "
                f"MaxDD: {record.max_drawdown:.1%} | Trades: {record.trade_count} | "
                f"FeeDrag: {record.fee_drag:.1%}{breadth}"
            )
            print(f"   Critic [{record.critic_verdict.upper()}]: {record.critic_summary}")
            print(f"   Next Step: {record.next_step_guidance}")
            print("-" * 84)

    if verbose and history:
        ranked = sorted(history, key=lambda r: r.sharpe_ratio, reverse=True)
        print("TOP 3 IN-SAMPLE CANDIDATES (Ready for Out-of-Sample Check):")
        for rank, rec in enumerate(ranked[:3], start=1):
            print(
                f"  #{rank} [Iter {rec.iteration:02d}] {rec.hypothesis_name:<36} "
                f"Sharpe={rec.sharpe_ratio:+.2f} | MaxDD={rec.max_drawdown:.1%} | "
                f"Trades={rec.trade_count} | {rec.strategy_type} {rec.parameters_json}"
            )
        print(f"\nFull experiment log saved to: {output_path}")

    return history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Agentic Quant Research Loop on the In-Sample split."
    )
    parser.add_argument("--data", type=Path, default=Path("data/prices.csv"), help="Input OHLCV CSV.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("logs/experiments.csv"),
        help="Output CSV for experiment logs.",
    )
    parser.add_argument(
        "--provider",
        choices=["mock", "gemini", "openai"],
        default="mock",
        help="Agent provider (default: mock — runs offline without API keys).",
    )
    parser.add_argument("--model", type=str, default=None, help="Optional LLM model name override.")
    parser.add_argument(
        "--target",
        type=str,
        default="ALL",
        help="Ticker symbol (e.g., MBB.VN, VCB.VN) or 'ALL' for 5-bank Equal-Weight Basket.",
    )
    parser.add_argument("--iterations", type=int, default=8, help="Number of research iterations (default: 8).")
    parser.add_argument("--train-ratio", type=float, default=0.70, help="Chronological In-Sample ratio (default: 0.70).")
    parser.add_argument("--fee-rate", type=float, default=DEFAULT_FEE_RATE, help="One-way fee rate (default: 0.0015).")
    parser.add_argument("--slippage-rate", type=float, default=DEFAULT_SLIPPAGE_RATE, help="One-way slippage (default: 0.0005).")
    parser.add_argument(
        "--allow-short",
        action="store_true",
        help="Enable short positions (-1). Default is Long-Only for HOSE equities.",
    )
    parser.add_argument("--max-retries", type=int, default=3, help="Max validation retries per iteration.")
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    run_research_loop(
        data_path=args.data,
        output_csv=args.output,
        provider_name=args.provider,
        model=args.model,
        target=args.target,
        iterations=args.iterations,
        train_ratio=args.train_ratio,
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
        allow_short=args.allow_short,
        max_retries=args.max_retries,
    )


if __name__ == "__main__":
    main()
