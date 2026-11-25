"""End-to-end smoke tests for the Agentic Loop (mock mode) and validation self-correction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from agent.loop import run_research_loop
from agent.providers import MockResearchProvider, ResearchProvider
from agent.schemas import CriticReport, ExperimentRecord, StrategyHypothesis
from scripts.download_data import DEFAULT_TICKERS, generate_synthetic_bank_data
from scripts.run_oos_eval import evaluate_top_k_oos


class _FlakyRetryProvider(ResearchProvider):
    """Test helper that intentionally triggers one domain-validation retry before succeeding."""

    provider_name = "flaky-test"

    def __init__(self) -> None:
        self._mock = MockResearchProvider()
        self.calls = 0

    def propose_hypothesis(
        self,
        dataset_summary: dict[str, Any],
        history: list[ExperimentRecord],
        validation_error: str | None = None,
    ) -> StrategyHypothesis:
        self.calls += 1
        if self.calls == 1:
            # Exceeds 35% of ~350 IS days (240 > 122) -> triggers retry
            return StrategyHypothesis(
                hypothesis_name="Too Wide Window",
                strategy_type="momentum",
                parameters={"fast_window": 50, "slow_window": 240, "min_spread_pct": 0.0},
                rationale="Intentionally oversized slow window to verify validation retry loop.",
            )
        assert validation_error is not None
        return self._mock.propose_hypothesis(dataset_summary, history, validation_error=None)

    def critique_result(
        self,
        hypothesis: StrategyHypothesis,
        metrics: dict[str, Any],
        history: list[ExperimentRecord],
    ) -> CriticReport:
        return self._mock.critique_result(hypothesis, metrics, history)


def test_end_to_end_mock_loop_and_oos_eval(tmp_path: Path) -> None:
    """Verify full offline pipeline: synthetic data -> 6-iter mock loop -> OOS evaluation."""
    data_csv = tmp_path / "prices.csv"
    exp_csv = tmp_path / "experiments.csv"
    oos_csv = tmp_path / "oos_evaluation.csv"

    df = generate_synthetic_bank_data(DEFAULT_TICKERS, years=3, seed=42)
    df.to_csv(data_csv, index=False)

    records = run_research_loop(
        data_path=data_csv,
        output_csv=exp_csv,
        provider_name="mock",
        target="ALL",
        iterations=6,
        verbose=False,
    )

    assert len(records) == 6
    assert exp_csv.exists()
    logged = pd.read_csv(exp_csv)
    assert len(logged) == 6
    # First 4 iterations should explore all 4 distinct strategy families
    assert set(logged["strategy_type"].iloc[:4]) == {
        "momentum",
        "mean_reversion",
        "breakout",
        "rsi_reversion",
    }

    oos_df = evaluate_top_k_oos(
        data_path=data_csv,
        experiments_csv=exp_csv,
        output_csv=oos_csv,
        top_k=3,
    )
    assert oos_csv.exists()
    assert len(oos_df) == 3
    assert {"is_sharpe", "oos_sharpe", "sharpe_delta", "oos_max_dd"}.issubset(oos_df.columns)


def test_validation_retry_self_correction(tmp_path: Path) -> None:
    """Verify that invalid proposals are caught and retried with error feedback."""
    data_csv = tmp_path / "prices.csv"
    exp_csv = tmp_path / "experiments.csv"
    generate_synthetic_bank_data(DEFAULT_TICKERS, years=2, seed=99).to_csv(data_csv, index=False)

    provider = _FlakyRetryProvider()
    records = run_research_loop(
        data_path=data_csv,
        output_csv=exp_csv,
        provider=provider,
        target="MBB.VN",
        iterations=1,
        verbose=False,
    )

    assert len(records) == 1
    assert records[0].validation_retries == 1
    assert provider.calls == 2
