"""Models, cost profiles and runtime settings.

Everything that a maintainer may want to tune lives here. Values can be
overridden with environment variables (see ``Settings.from_env``) so that the
GitHub workflow can change behaviour through repository *variables* without a
code change.

Model IDs and prices were verified on https://openrouter.ai on 2026-09-23
(standard tier, USD per million tokens). Only OpenAI, Anthropic and Google
models are used because the free tournament credits only cover those three
providers.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone

# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ModelSpec:
    """One OpenRouter model plus the settings used when calling it."""

    model: str  # OpenRouter model id, e.g. "anthropic/claude-opus-5.5"
    runs: int = 1  # independent samples per question (forecasters only)
    max_tokens: int = 16_000  # includes reasoning tokens
    reasoning_effort: str | None = "high"
    price_in: float = 0.0  # USD per 1M prompt tokens
    price_out: float = 0.0  # USD per 1M completion tokens (reasoning included)
    search_price: float = 0.01  # USD per native web search
    timeout_s: float = 300.0

    def worst_case_cost(self, prompt_tokens: int, searches: int = 0) -> float:
        """Upper bound of the cost of one call (used to reserve budget)."""
        tokens_cost = (
            prompt_tokens * self.price_in + self.max_tokens * self.price_out
        ) / 1_000_000
        return tokens_cost + searches * self.search_price

    def with_runs(self, runs: int) -> "ModelSpec":
        return replace(self, runs=runs)


# Forecasters (reasoning effort high).
CLAUDE_FABLE = ModelSpec(
    "anthropic/claude-fable-5.1", max_tokens=16_000, price_in=10.0, price_out=50.0
)
CLAUDE_OPUS = ModelSpec(
    "anthropic/claude-opus-5.5", max_tokens=16_000, price_in=4.0, price_out=20.0
)
GPT_ASTRA = ModelSpec(
    "openai/gpt-6-astra", max_tokens=20_000, price_in=10.0, price_out=50.0
)
GPT_SOL = ModelSpec("openai/gpt-6-sol", max_tokens=20_000, price_in=2.0, price_out=10.0)
GEMINI_PRO = ModelSpec(
    "google/gemini-3.1-pro-preview",
    max_tokens=20_000,
    price_in=2.0,
    price_out=12.0,
    search_price=0.014,
)
GEMINI_FLASH = ModelSpec(
    "google/gemini-3.8-flash",
    max_tokens=20_000,
    price_in=0.75,
    price_out=3.75,
    search_price=0.014,
)

# Researchers: same models with web search and low reasoning effort (the
# search does the heavy lifting; we want facts, not long deliberation).
RESEARCH_SOL = replace(GPT_SOL, max_tokens=8_000, reasoning_effort="low", timeout_s=240)
RESEARCH_FLASH = replace(
    GEMINI_FLASH, max_tokens=8_000, reasoning_effort="low", timeout_s=240
)
RESEARCH_OPUS = replace(
    CLAUDE_OPUS, max_tokens=8_000, reasoning_effort="low", timeout_s=240
)

# Parser used only when the regex/JSON parsers cannot read an answer.
PARSER = replace(GEMINI_FLASH, max_tokens=3_000, reasoning_effort="low", timeout_s=90)

# --------------------------------------------------------------------------- #
# Profiles
# --------------------------------------------------------------------------- #

ASKNEWS_LATEST = "latest news"  # last 48 h, 1 AskNews call
ASKNEWS_KNOWLEDGE = "news knowledge"  # archive, counted as several calls


@dataclass(frozen=True)
class Profile:
    """A complete recipe for one question: who researches and who forecasts."""

    name: str
    forecasters: tuple[ModelSpec, ...]
    web_researchers: tuple[ModelSpec, ...]
    asknews_strategies: tuple[str, ...]
    web_max_searches: int
    cap_per_question: float  # hard cap in USD (worst-case reservations)
    expected_cost: float  # typical cost in USD, used for budget planning

    @property
    def planned_members(self) -> int:
        return sum(spec.runs for spec in self.forecasters)

    def fast(self) -> "Profile":
        """Degraded recipe for questions that close soon."""
        return replace(
            self,
            name=f"{self.name}-fast",
            forecasters=tuple(spec.with_runs(1) for spec in self.forecasters),
            web_researchers=self.web_researchers[:1],
            asknews_strategies=(ASKNEWS_LATEST,) if self.asknews_strategies else (),
            web_max_searches=min(self.web_max_searches, 2),
        )


PROFILES: dict[str, Profile] = {
    # Flagship model of each provider, 2 runs each (≈1.7 USD/question).
    "premium": Profile(
        name="premium",
        forecasters=(
            CLAUDE_FABLE.with_runs(2),
            GPT_ASTRA.with_runs(2),
            GEMINI_PRO.with_runs(2),
        ),
        web_researchers=(RESEARCH_SOL, RESEARCH_FLASH),
        asknews_strategies=(ASKNEWS_LATEST, ASKNEWS_KNOWLEDGE),
        web_max_searches=4,
        cap_per_question=6.0,
        expected_cost=1.70,
    ),
    # Default: strong model of each provider, 3 runs each (≈0.85 USD/question).
    "standard": Profile(
        name="standard",
        forecasters=(
            CLAUDE_OPUS.with_runs(3),
            GPT_SOL.with_runs(3),
            GEMINI_FLASH.with_runs(3),
        ),
        web_researchers=(RESEARCH_SOL, RESEARCH_FLASH),
        asknews_strategies=(ASKNEWS_LATEST, ASKNEWS_KNOWLEDGE),
        web_max_searches=4,
        cap_per_question=2.5,
        expected_cost=0.85,
    ),
    # Cheap: one or two runs per model, one web researcher plus AskNews, or two
    # web researchers when AskNews is not configured (≈0.32-0.40 USD/question).
    "economy": Profile(
        name="economy",
        forecasters=(
            CLAUDE_OPUS.with_runs(1),
            GPT_SOL.with_runs(1),
            GEMINI_FLASH.with_runs(2),
        ),
        web_researchers=(RESEARCH_FLASH,),
        asknews_strategies=(ASKNEWS_LATEST,),
        web_max_searches=3,
        cap_per_question=1.1,
        expected_cost=0.38,
    ),
}

PROFILE_ORDER = ("premium", "standard", "economy")  # most to least expensive


def all_model_specs() -> list[ModelSpec]:
    """Every distinct model used anywhere (for the `models` health check)."""
    seen: dict[str, ModelSpec] = {}
    for profile in PROFILES.values():
        for spec in (*profile.forecasters, *profile.web_researchers):
            seen.setdefault(spec.model, spec)
    seen.setdefault(PARSER.model, PARSER)
    return list(seen.values())


# --------------------------------------------------------------------------- #
# Runtime settings
# --------------------------------------------------------------------------- #


def _env_str(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip()


def _env_float(name: str, default: float) -> float:
    value = _env_str(name)
    return float(value) if value is not None else default


def _env_int(name: str, default: int) -> int:
    value = _env_str(name)
    return int(value) if value is not None else default


def _env_bool(name: str, default: bool) -> bool:
    value = _env_str(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "y", "on"}


@dataclass
class Settings:
    """Runtime settings. Defaults are tuned for the Fall 2026 season."""

    # Where to forecast.
    seasonal_tournament: str = "fall-futureeval-2026"
    minibench_tournament: str = "minibench"
    test_tournament: str = "bot-testing-area"
    season_start: datetime = datetime(2026, 9, 28, tzinfo=timezone.utc)
    season_end: datetime = datetime(2027, 1, 6, tzinfo=timezone.utc)

    # Budget planning.
    expected_seasonal_questions: int = 400
    expected_minibench_per_week: float = 30.0
    profile_override: str | None = None  # BOT_PROFILE (seasonal tournament)
    minibench_profile: str = "economy"
    test_profile: str = "economy"
    credit_reserve_usd: float = 5.0
    max_cost_per_run_usd: float = 40.0

    # Throughput and timing.
    max_concurrent_questions: int = 3
    max_concurrent_llm_calls: int = 8
    min_minutes_before_close: float = 3.0  # skip if closing sooner than this
    fast_path_minutes: float = 15.0  # use the degraded profile below this
    max_minutes_per_question: float = 25.0
    min_success_fraction: float = 0.34  # of planned forecasters

    # Aggregation and calibration.
    trim_fraction: float = 0.2
    binary_clip_low: float = 0.02
    binary_clip_high: float = 0.98
    platt_a: float = 1.0  # identity until fitted on resolved questions
    platt_b: float = 0.0
    mc_floor: float = 0.01

    # Research.
    asknews_articles: int = 8
    asknews_min_interval_s: float = 11.0  # free tier: 1 call per 10 s
    research_chars_per_source: int = 12_000

    # Logging. Public GitHub logs must not reveal forecasts on open questions.
    log_forecasts: bool = False

    extra: dict = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "Settings":
        s = cls()
        s.seasonal_tournament = _env_str("SEASONAL_TOURNAMENT", s.seasonal_tournament)
        s.minibench_tournament = _env_str(
            "MINIBENCH_TOURNAMENT", s.minibench_tournament
        )
        s.test_tournament = _env_str("TEST_TOURNAMENT", s.test_tournament)
        s.expected_seasonal_questions = _env_int(
            "EXPECTED_SEASONAL_QUESTIONS", s.expected_seasonal_questions
        )
        s.expected_minibench_per_week = _env_float(
            "EXPECTED_MINIBENCH_PER_WEEK", s.expected_minibench_per_week
        )
        s.profile_override = _env_str("BOT_PROFILE", None)
        s.minibench_profile = _env_str("MINIBENCH_PROFILE", s.minibench_profile)
        s.test_profile = _env_str("TEST_PROFILE", s.test_profile)
        s.credit_reserve_usd = _env_float("CREDIT_RESERVE_USD", s.credit_reserve_usd)
        s.max_cost_per_run_usd = _env_float(
            "MAX_COST_PER_RUN_USD", s.max_cost_per_run_usd
        )
        s.max_concurrent_questions = _env_int(
            "MAX_CONCURRENT_QUESTIONS", s.max_concurrent_questions
        )
        s.max_concurrent_llm_calls = _env_int(
            "MAX_CONCURRENT_LLM_CALLS", s.max_concurrent_llm_calls
        )
        s.trim_fraction = _env_float("TRIM_FRACTION", s.trim_fraction)
        s.binary_clip_low = _env_float("BINARY_CLIP_LOW", s.binary_clip_low)
        s.binary_clip_high = _env_float("BINARY_CLIP_HIGH", s.binary_clip_high)
        s.platt_a = _env_float("PLATT_A", s.platt_a)
        s.platt_b = _env_float("PLATT_B", s.platt_b)
        s.log_forecasts = _env_bool("LOG_FORECASTS", s.log_forecasts)
        for name in (s.profile_override, s.minibench_profile, s.test_profile):
            if name is not None and name not in PROFILES:
                raise ValueError(
                    f"Unknown profile '{name}'. Valid: {', '.join(PROFILES)}"
                )
        return s
