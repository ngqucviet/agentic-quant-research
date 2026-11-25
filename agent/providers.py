"""Research Agent providers: offline MockResearchProvider + ready Gemini/OpenAI adapters.

Design principle:
    The research loop in ``agent/loop.py`` talks ONLY to the ``ResearchProvider``
    interface. You can run and verify the entire end-to-end pipeline offline
    with ``--provider mock`` (zero API key required), and switch to a real LLM
    later by setting ``GEMINI_API_KEY`` or ``OPENAI_API_KEY`` in ``.env`` and
    passing ``--provider gemini`` or ``--provider openai``.
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import Any

from agent.prompts import SYSTEM_PROMPT, build_critic_prompt, build_propose_prompt
from agent.schemas import CriticReport, ExperimentRecord, StrategyHypothesis


class ResearchProvider(ABC):
    """Abstract interface for proposing hypotheses and critiquing backtests."""

    provider_name: str = "base"

    @abstractmethod
    def propose_hypothesis(
        self,
        dataset_summary: dict[str, Any],
        history: list[ExperimentRecord],
        validation_error: str | None = None,
    ) -> StrategyHypothesis:
        """Propose the next structured trading strategy hypothesis."""

    @abstractmethod
    def critique_result(
        self,
        hypothesis: StrategyHypothesis,
        metrics: dict[str, Any],
        history: list[ExperimentRecord],
    ) -> CriticReport:
        """Critique a completed In-Sample backtest and guide the next iteration."""


class MockResearchProvider(ResearchProvider):
    """Deterministic, history-aware heuristic researcher for offline execution.

    Unlike a random generator, this mock agent:
      1. Explores all 4 strategy families cleanly in early iterations.
      2. Inspects quantitative feedback from prior runs (Sharpe, fee drag,
         trade count, exposure, drawdown) and adapts parameters accordingly.
      3. Never repeats already-tested signatures and respects validation retries.
      4. Produces grounded, metric-driven critiques in ``critique_result``.
    """

    provider_name = "mock"

    def __init__(self) -> None:
        self._rejected_signatures: set[str] = set()

        # Curated exploration & refinement roadmap across all 4 strategy families
        self._exploration_queue: list[tuple[str, str, dict[str, int | float], str]] = [
            (
                "SMA 20/60 Trend Baseline",
                "momentum",
                {"fast_window": 20, "slow_window": 60, "min_spread_pct": 0.005},
                "Baseline medium-term trend-following rule with 0.5% hysteresis band to test directional persistence in VN banking stocks.",
            ),
            (
                "Z-Score 20 Mean Reversion",
                "mean_reversion",
                {"window": 20, "entry_z": 1.5, "exit_z": 0.0},
                "Baseline short-term oversold bounce rule capturing mean reversion during sideways bank consolidations.",
            ),
            (
                "Donchian 20/10 Breakout",
                "breakout",
                {"entry_window": 20, "exit_window": 10, "vol_multiplier": 0.0},
                "Classic asymmetric Donchian breakout (20-day entry, 10-day trailing exit) to participate in strong bank rallies while cutting losers.",
            ),
            (
                "RSI-14 Trend-Filtered Reversion",
                "rsi_reversion",
                {"window": 14, "oversold": 35.0, "overbought": 65.0, "trend_window": 100},
                "Wilder RSI mean-reversion filtered by 100-day SMA to buy pullbacks only within primary uptrends.",
            ),
        ]

        # Targeted refinements keyed by strategy family
        self._refinements: dict[str, list[tuple[str, dict[str, int | float], str]]] = {
            "momentum": [
                (
                    "SMA 15/50 Low-Whipsaw Momentum",
                    {"fast_window": 15, "slow_window": 50, "min_spread_pct": 0.015},
                    "Increasing hysteresis deadband to 1.5% to suppress whipsaw trades and reduce transaction fee drag.",
                ),
                (
                    "SMA 30/90 Slow Regime Momentum",
                    {"fast_window": 30, "slow_window": 90, "min_spread_pct": 0.01},
                    "Lengthening moving average windows to filter short-term noise and lower turnover.",
                ),
                (
                    "SMA 10/40 Responsive Momentum",
                    {"fast_window": 10, "slow_window": 40, "min_spread_pct": 0.01},
                    "Testing faster trend capture while preserving a 1.0% deadband against choppy reversals.",
                ),
            ],
            "mean_reversion": [
                (
                    "Z-Score 30 High-Conviction Reversion",
                    {"window": 30, "entry_z": 1.8, "exit_z": -0.2},
                    "Widening lookback to 30 days and raising entry_z to 1.8 to reduce trade frequency and improve per-trade edge.",
                ),
                (
                    "Z-Score 40 Patient Mean Reversion",
                    {"window": 40, "entry_z": 1.5, "exit_z": -0.5},
                    "Holding rebounds past the rolling mean (exit_z=-0.5) to increase average gain relative to round-trip costs.",
                ),
                (
                    "Z-Score 15 Fast Pullback Reversion",
                    {"window": 15, "entry_z": 1.3, "exit_z": 0.0},
                    "Testing a tighter 15-day window with lower entry threshold to increase capital exposure.",
                ),
            ],
            "breakout": [
                (
                    "Donchian 30/15 Vol-Buffered Breakout",
                    {"entry_window": 30, "exit_window": 15, "vol_multiplier": 0.25},
                    "Adding 0.25x rolling-std buffer and wider channel to avoid false breakouts and trim fee drag.",
                ),
                (
                    "Donchian 40/20 Slow Channel Breakout",
                    {"entry_window": 40, "exit_window": 20, "vol_multiplier": 0.0},
                    "Extending breakout horizon to 40/20 days to capture multi-month banking sector trends with minimal turnover.",
                ),
                (
                    "Donchian 15/10 Tactical Breakout",
                    {"entry_window": 15, "exit_window": 10, "vol_multiplier": 0.15},
                    "Testing quicker breakout entry with light volatility confirmation.",
                ),
            ],
            "rsi_reversion": [
                (
                    "RSI-14 Strict Oversold + SMA-120 Filter",
                    {"window": 14, "oversold": 32.0, "overbought": 72.0, "trend_window": 120},
                    "Tightening entry to RSI<=32 and letting winners run to RSI>=72 under a 120-day trend regime filter.",
                ),
                (
                    "RSI-10 Unfiltered Swing Reversion",
                    {"window": 10, "oversold": 30.0, "overbought": 68.0, "trend_window": 0},
                    "Testing faster RSI-10 without trend filter to evaluate whether higher exposure improves risk-adjusted return.",
                ),
                (
                    "RSI-20 Smooth Trend Pullback",
                    {"window": 20, "oversold": 40.0, "overbought": 70.0, "trend_window": 80},
                    "Smoothing RSI to 20 days to reduce signal noise and fee drag while maintaining trend alignment.",
                ),
            ],
        }

    def _score_record(self, rec: ExperimentRecord) -> float:
        """Composite research score prioritizing Sharpe while penalizing fee drag and empty runs."""
        if rec.trade_count == 0:
            return -10.0
        return rec.sharpe_ratio - 2.0 * max(0.0, rec.fee_drag) - 0.5 * abs(rec.max_drawdown)

    def propose_hypothesis(
        self,
        dataset_summary: dict[str, Any],
        history: list[ExperimentRecord],
        validation_error: str | None = None,
    ) -> StrategyHypothesis:
        tested_sigs = {
            f"{rec.strategy_type}:{rec.parameters_json}" for rec in history
        } | self._rejected_signatures

        # Phase 1: Explore all 4 strategy families first
        for name, stype, params, rationale in self._exploration_queue:
            cand = StrategyHypothesis(
                hypothesis_name=name,
                strategy_type=stype,  # type: ignore[arg-type]
                parameters=params,
                rationale=rationale,
            )
            if cand.signature() not in tested_sigs:
                if validation_error:
                    self._rejected_signatures.add(cand.signature())
                return cand

        # Phase 2: Rank strategy families by best observed In-Sample risk-adjusted score
        family_best: dict[str, float] = {}
        for rec in history:
            score = self._score_record(rec)
            if rec.strategy_type not in family_best or score > family_best[rec.strategy_type]:
                family_best[rec.strategy_type] = score

        ranked_families = sorted(
            self._refinements.keys(),
            key=lambda fam: family_best.get(fam, -999.0),
            reverse=True,
        )

        # Pick the highest-priority untested refinement from the best-performing families
        last_rec = history[-1] if history else None
        for fam in ranked_families:
            for name, params, base_rationale in self._refinements[fam]:
                rationale = base_rationale
                if last_rec is not None:
                    rationale = (
                        f"After iter {last_rec.iteration} ({last_rec.hypothesis_name}, "
                        f"Sharpe={last_rec.sharpe_ratio:.2f}, fee_drag={last_rec.fee_drag:.2%}), "
                        f"{base_rationale[0].lower()}{base_rationale[1:]}"
                    )
                cand = StrategyHypothesis(
                    hypothesis_name=name,
                    strategy_type=fam,  # type: ignore[arg-type]
                    parameters=params,
                    rationale=rationale,
                )
                if cand.signature() not in tested_sigs:
                    return cand

        # Fallback: systematically vary momentum windows if user runs > 16 iterations
        step = len(history) + 1
        fast = min(5 + (step * 3) % 40, 45)
        slow = min(fast + 25 + (step * 5) % 80, 150)
        return StrategyHypothesis(
            hypothesis_name=f"Adaptive SMA {fast}/{slow} Exploration",
            strategy_type="momentum",
            parameters={"fast_window": fast, "slow_window": slow, "min_spread_pct": 0.01},
            rationale=(
                f"Systematic parameter sweep iteration {step} testing SMA({fast}, {slow}) "
                "with 1% deadband after exhausting curated refinements."
            ),
        )

    def critique_result(
        self,
        hypothesis: StrategyHypothesis,
        metrics: dict[str, Any],
        history: list[ExperimentRecord],
    ) -> CriticReport:
        sharpe = float(metrics["sharpe_ratio"])
        bench_sharpe = float(metrics["benchmark_sharpe"])
        cum_ret = float(metrics["cumulative_return"])
        bench_ret = float(metrics["benchmark_return"])
        fee_drag = float(metrics["fee_drag"])
        max_dd = float(metrics["max_drawdown"])
        bench_dd = float(metrics["benchmark_max_drawdown"])
        trades = int(metrics["trade_count"])
        exposure = float(metrics["exposure"])

        breadth_note = ""
        if "positive_sharpe_banks" in metrics and "total_banks" in metrics:
            breadth_note = (
                f" Positive Sharpe on {metrics['positive_sharpe_banks']}/{metrics['total_banks']} banks."
            )

        if trades == 0 or exposure < 0.01:
            return CriticReport(
                verdict="no_trades",
                summary=(
                    f"{hypothesis.hypothesis_name} generated {trades} trades (exposure={exposure:.1%}), "
                    "providing no statistical evidence."
                ),
                key_risk="Entry thresholds are too restrictive for the observed volatility regime.",
                next_step_guidance="Loosen entry thresholds or shorten lookback windows to generate active signals.",
            )

        if fee_drag > 0.08 and fee_drag > max(0.02, 0.35 * abs(float(metrics["gross_cumulative_return"]))):
            return CriticReport(
                verdict="overtrading",
                summary=(
                    f"Sharpe={sharpe:.2f} (vs B&H {bench_sharpe:.2f}), net return={cum_ret:.1%}, "
                    f"but {trades} trades incurred {fee_drag:.1%} fee drag.{breadth_note}"
                ),
                key_risk="High turnover allows transaction fees and slippage to erode gross edge.",
                next_step_guidance="Increase hysteresis/buffer or lengthen lookback windows to cut trade frequency.",
            )

        if sharpe >= max(0.6, bench_sharpe) and abs(max_dd) <= abs(bench_dd) + 0.03:
            return CriticReport(
                verdict="promising",
                summary=(
                    f"Strong risk-adjusted profile: Sharpe={sharpe:.2f} (B&H {bench_sharpe:.2f}), "
                    f"net return={cum_ret:.1%} (B&H {bench_ret:.1%}), MaxDD={max_dd:.1%} (B&H {bench_dd:.1%}), "
                    f"trades={trades}, fee_drag={fee_drag:.1%}.{breadth_note}"
                ),
                key_risk="Potential In-Sample parameter sensitivity; must verify stability on unseen Out-of-Sample split.",
                next_step_guidance="Test neighboring parameter values to verify robustness rather than chasing a sharp peak.",
            )

        if sharpe > 0.25 or abs(max_dd) < abs(bench_dd) * 0.75:
            return CriticReport(
                verdict="mixed",
                summary=(
                    f"Moderate performance: Sharpe={sharpe:.2f} (B&H {bench_sharpe:.2f}), "
                    f"MaxDD={max_dd:.1%} vs B&H {bench_dd:.1%}, trades={trades}, fee_drag={fee_drag:.1%}.{breadth_note}"
                ),
                key_risk="Risk reduction is visible, but upside participation lags Buy & Hold or costs remain noticeable.",
                next_step_guidance="Refine exit rules or trend filter to keep drawdown protection while improving upside capture.",
            )

        return CriticReport(
            verdict="underperforming",
            summary=(
                f"Weak In-Sample edge: Sharpe={sharpe:.2f} (B&H {bench_sharpe:.2f}), "
                f"net return={cum_ret:.1%} vs B&H {bench_ret:.1%}, MaxDD={max_dd:.1%}, "
                f"trades={trades}.{breadth_note}"
            ),
            key_risk="Signal dynamics do not align well with the dominant In-Sample banking market regime.",
            next_step_guidance="Pivot to an alternative strategy family or add a regime/trend filter.",
        )


class GeminiProvider(ResearchProvider):
    """Google Gemini API provider using Structured Outputs (``google-genai`` SDK).

    Requires:
        - ``pip install google-genai``
        - ``GEMINI_API_KEY`` environment variable (or in ``.env``)
    """

    provider_name = "gemini"

    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Add it to .env or pass --provider mock to run offline."
            )
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise RuntimeError(
                "google-genai SDK is not installed. Run `pip install google-genai` "
                "or use `--provider mock`."
            ) from exc

        self._types = types
        self._client = genai.Client(api_key=self.api_key)

    def _call_structured(self, prompt: str, schema_cls: type[StrategyHypothesis] | type[CriticReport]) -> Any:
        schema_hint = json.dumps(schema_cls.model_json_schema(), indent=2)
        full_prompt = (
            prompt
            + "\n\nRespond ONLY with a valid JSON object matching this JSON Schema:\n"
            + schema_hint
        )
        response = self._client.models.generate_content(
            model=self.model,
            contents=full_prompt,
            config=self._types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                response_mime_type="application/json",
                temperature=0.3,
            ),
        )
        raw_text = response.text
        if not raw_text:
            raise RuntimeError("Gemini returned an empty response.")
        return schema_cls.model_validate_json(raw_text)

    def propose_hypothesis(
        self,
        dataset_summary: dict[str, Any],
        history: list[ExperimentRecord],
        validation_error: str | None = None,
    ) -> StrategyHypothesis:
        prompt = build_propose_prompt(dataset_summary, history, validation_error)
        return self._call_structured(prompt, StrategyHypothesis)

    def critique_result(
        self,
        hypothesis: StrategyHypothesis,
        metrics: dict[str, Any],
        history: list[ExperimentRecord],
    ) -> CriticReport:
        prompt = build_critic_prompt(hypothesis, metrics, history)
        return self._call_structured(prompt, CriticReport)


class OpenAIProvider(ResearchProvider):
    """OpenAI (or OpenAI-compatible: OpenRouter, Groq, Ollama) provider.

    Requires:
        - ``pip install openai``
        - ``OPENAI_API_KEY`` environment variable (and optional ``OPENAI_BASE_URL``)
    """

    provider_name = "openai"

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        if not self.api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Add it to .env or pass --provider mock to run offline."
            )
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "openai SDK is not installed. Run `pip install openai` or use `--provider mock`."
            ) from exc

        self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def _call_structured(self, prompt: str, schema_cls: type[StrategyHypothesis] | type[CriticReport]) -> Any:
        schema_hint = json.dumps(schema_cls.model_json_schema(), indent=2)
        completion = self._client.chat.completions.create(
            model=self.model,
            temperature=0.3,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT + "\n\nRespond ONLY with JSON matching this schema:\n" + schema_hint,
                },
                {"role": "user", "content": prompt},
            ],
        )
        content = completion.choices[0].message.content
        if not content:
            raise RuntimeError("OpenAI-compatible provider returned an empty response.")
        return schema_cls.model_validate_json(content)

    def propose_hypothesis(
        self,
        dataset_summary: dict[str, Any],
        history: list[ExperimentRecord],
        validation_error: str | None = None,
    ) -> StrategyHypothesis:
        prompt = build_propose_prompt(dataset_summary, history, validation_error)
        return self._call_structured(prompt, StrategyHypothesis)

    def critique_result(
        self,
        hypothesis: StrategyHypothesis,
        metrics: dict[str, Any],
        history: list[ExperimentRecord],
    ) -> CriticReport:
        prompt = build_critic_prompt(hypothesis, metrics, history)
        return self._call_structured(prompt, CriticReport)


def create_provider(name: str = "mock", model: str | None = None) -> ResearchProvider:
    """Instantiate a ResearchProvider by name ('mock', 'gemini', or 'openai')."""
    normalized = name.strip().lower()
    if normalized == "mock":
        return MockResearchProvider()
    if normalized == "gemini":
        return GeminiProvider(model=model)
    if normalized == "openai":
        return OpenAIProvider(model=model)
    raise ValueError(f"Unsupported provider '{name}'. Choose from: mock, gemini, openai.")
