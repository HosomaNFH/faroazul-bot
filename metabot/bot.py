"""ForecastBot subclass: research -> ensemble -> parse -> aggregate -> publish.

The parent class (forecasting-tools) handles fetching questions, skipping
already-forecast questions, error collection, the report/comment format and
publishing (forecast + private comment). We override:

* ``run_research``: multi-source research (research.py);
* ``_research_and_make_predictions``: one research pass, then every ensemble
  member (model x run) in parallel under a per-question budget and deadline;
* ``_aggregate_predictions``: our robust aggregation (aggregation.py);
* ``forecast_questions``: skip questions that close too soon, earliest first.

Conditional questions (not used in the tournament, present in the testing
area) reuse the logic of forecasting-tools' template bot (MIT licence).
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
import math
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
from forecasting_tools import (
    BinaryQuestion,
    ConditionalPrediction,
    ConditionalQuestion,
    DataOrganizer,
    ForecastBot,
    MetaculusQuestion,
    MonetaryCostManager,
    MultipleChoiceQuestion,
    NumericDistribution,
    Percentile,
    PredictedOption,
    PredictedOptionList,
    PredictionAffirmed,
    QuestionState,
    ReasonedPrediction,
)
from forecasting_tools.data_models.forecast_report import ResearchWithPredictions

from .aggregation import (
    aggregate_binary,
    aggregate_cdfs,
    aggregate_multiple_choice,
    normalize_with_floor,
)
from .budget import Budget
from .cdf import Scale, build_cdf, plausibility_issue, validate_cdf
from .config import PARSER, PROFILES, ModelSpec, Profile, Settings
from .llm import LlmError, OpenRouterClient
from .parsing import extract_json_objects, parse_binary, parse_multiple_choice, parse_percentiles, to_float
from .prompts import (
    SYSTEM_FORECASTER,
    SYSTEM_PARSER,
    binary_prompt,
    multiple_choice_prompt,
    numeric_prompt,
    parser_prompt,
)
from .qctx import QCtx, ctx_from_question
from .research import AskNewsClient, SourceResult, gather_research

logger = logging.getLogger("metabot")


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _aware(dt: datetime | None) -> datetime | None:
    if dt is None or dt.tzinfo is not None:
        return dt
    return dt.replace(tzinfo=timezone.utc)


class ForecastFailure(RuntimeError):
    """A question could not be forecast with enough confidence to publish."""


@dataclass
class QState:
    ctx: QCtx
    profile: Profile
    budget: Budget
    deadline: float  # time.monotonic() value
    fast: bool
    started: float = field(default_factory=time.monotonic)
    sources: list[SourceResult] = field(default_factory=list)
    members_ok: int = 0
    members_planned: int = 0


_CURRENT_STATE: contextvars.ContextVar[QState | None] = contextvars.ContextVar(
    "metabot_qstate", default=None
)


def report_cost_to_forecasting_tools(cost: float) -> None:
    """OpenRouterClient callback: add real costs to the per-question manager
    that ForecastBot opens, so reports and comments show actual spend."""
    MonetaryCostManager.increase_current_usage_in_parent_managers(cost)


class EnsembleBot(ForecastBot):
    """Multi-model ensemble bot for Metaculus FutureEval."""

    def __init__(
        self,
        *,
        settings: Settings,
        client: OpenRouterClient,
        run_budget: Budget,
        dry_run: bool,
        kind_profiles: dict[str, str] | None = None,
        save_folder: str | None = None,
        skip_previously_forecasted: bool = True,
    ) -> None:
        # Attributes used by overridden methods called from the parent __init__.
        self.settings = settings
        self.client = client
        self.run_budget = run_budget
        self.dry_run = dry_run
        self.kind_profiles: dict[str, str] = dict(kind_profiles or {})
        self.current_kind = "seasonal"
        self.credits_exhausted = False
        self.states: dict[int, QState] = {}
        self.last_selected: list = []
        self._question_semaphore = asyncio.Semaphore(settings.max_concurrent_questions)
        super().__init__(
            research_reports_per_question=1,
            predictions_per_research_report=1,
            use_research_summary_to_forecast=False,
            publish_reports_to_metaculus=not dry_run,
            folder_to_save_reports_to=save_folder,
            skip_previously_forecasted_questions=skip_previously_forecasted,
            llms=self._llm_config_defaults(),
            enable_summarize_research=False,
            extra_metadata_in_explanation=True,
            required_successful_predictions=0.0,  # we enforce our own minimum
        )

    # ------------------------------------------------------------------ config
    @classmethod
    def _llm_config_defaults(cls) -> dict[str, Any]:
        # Informational only: calls go through OpenRouterClient, not GeneralLlm.
        return {
            "default": "openrouter/ensemble (see make_llm_dict)",
            "summarizer": "disabled",
            "researcher": "asknews + openrouter:web_search",
            "parser": f"openrouter/{PARSER.model}",
        }

    def make_llm_dict(self) -> dict[str, Any]:  # shown in the private comment
        info: dict[str, Any] = {}
        for kind, name in getattr(self, "kind_profiles", {}).items():
            profile = PROFILES[name]
            info[kind] = {
                "profile": name,
                "forecasters": [f"{s.model} x{s.runs}" for s in profile.forecasters],
                "web_research": [s.model for s in profile.web_researchers],
                "asknews": list(profile.asknews_strategies),
            }
        return info

    def _profile_for(self, question: MetaculusQuestion) -> tuple[Profile, bool]:
        name = self.kind_profiles.get(self.current_kind, "standard")
        profile = PROFILES[name]
        minutes_left = None
        close_time = _aware(question.close_time)
        if close_time is not None:
            minutes_left = (close_time - utcnow()).total_seconds() / 60
        fast = minutes_left is not None and minutes_left < self.settings.fast_path_minutes
        return (profile.fast() if fast else profile), fast

    def _new_state(self, question: MetaculusQuestion) -> QState:
        if isinstance(question, ConditionalQuestion):
            ctx = ctx_from_question(question.child)
            ctx.title = (
                f"Conditional question. Parent: {question.parent.question_text} | "
                f"Child: {question.child.question_text}"
            )
        else:
            ctx = ctx_from_question(question)
        profile, fast = self._profile_for(question)
        budget = Budget(profile.cap_per_question, parent=self.run_budget, name=str(question.id_of_question))
        seconds = self.settings.max_minutes_per_question * 60
        close_time = _aware(question.close_time)
        if close_time is not None:
            to_close = (close_time - utcnow()).total_seconds() - 120  # publish reserve
            seconds = min(seconds, to_close)
        seconds = max(seconds, 60.0)
        state = QState(
            ctx=ctx,
            profile=profile,
            budget=budget,
            deadline=time.monotonic() + seconds,
            fast=fast,
            members_planned=profile.planned_members,
        )
        if question.id_of_question is not None:
            self.states[question.id_of_question] = state
        return state

    async def _initialize_notepad(self, question: MetaculusQuestion):  # type: ignore[override]
        notepad = await super()._initialize_notepad(question)
        notepad.note_entries["qstate"] = self._new_state(question)
        return notepad

    async def _state_for(self, question: MetaculusQuestion) -> QState:
        current = _CURRENT_STATE.get()
        if current is not None:
            return current
        notepad = await self._get_notepad(question)
        return notepad.note_entries["qstate"]

    # ---------------------------------------------------------------- filters
    def select_questions(self, questions) -> list:
        """Questions this run will forecast, earliest close first.

        Deterministic and idempotent: the runner zips ``last_selected`` with
        the reports returned by ``forecast_questions``.
        """
        now = utcnow()
        keep = []
        for question in questions:
            if self.skip_previously_forecasted_questions and question.already_forecasted:
                continue
            if not self.dry_run and question.state not in (None, QuestionState.OPEN):
                logger.info("Skip %s: not open (%s)", question.page_url, question.state)
                continue
            close_time = _aware(question.close_time)
            if close_time is not None and not self.dry_run:
                minutes = (close_time - now).total_seconds() / 60
                if minutes < self.settings.min_minutes_before_close:
                    logger.warning("Skip %s: closes in %.1f min", question.page_url, minutes)
                    continue
            keep.append(question)
        far = now + timedelta(days=36500)
        keep.sort(key=lambda q: _aware(q.close_time) or far)
        return keep

    async def forecast_questions(  # type: ignore[override]
        self, questions, return_exceptions: bool = False
    ):
        selected = self.select_questions(questions)
        self.last_selected = selected
        # Already-forecast questions were removed above, so the parent's own
        # skip filter keeps this order and the results align with `selected`.
        return await super().forecast_questions(selected, return_exceptions)

    # --------------------------------------------------------------- research
    async def run_research(self, question: MetaculusQuestion) -> str:
        state = await self._state_for(question)
        fallback = replace(state.profile.forecasters[0], reasoning_effort="low", max_tokens=4000)
        text, sources = await gather_research(
            state.ctx,
            state.profile,
            self.client,
            state.budget,
            state.deadline,
            utcnow(),
            self.settings,
            fallback_spec=fallback,
        )
        state.sources = sources
        ok = [s.name for s in sources if s.ok]
        logger.info(
            "Research for %s: %d source(s) ok (%s)",
            question.page_url,
            len(ok),
            ", ".join(ok) or "none",
        )
        return text

    # ------------------------------------------------------------ forecasting
    async def _research_and_make_predictions(  # type: ignore[override]
        self, question: MetaculusQuestion
    ) -> ResearchWithPredictions:
        notepad = await self._get_notepad(question)
        notepad.total_research_reports_attempted += 1
        state: QState = notepad.note_entries["qstate"]
        token = _CURRENT_STATE.set(state)
        try:
            async with self._question_semaphore:
                if self.credits_exhausted:
                    raise ForecastFailure("OpenRouter credits exhausted; not forecasting")
                research = await self.run_research(question)
                if isinstance(question, ConditionalQuestion):
                    predictions = [await self._run_forecast_on_conditional(question, research)]
                    errors: list[str] = []
                    planned = 1
                else:
                    members = [
                        (spec, run) for spec in state.profile.forecasters for run in range(spec.runs)
                    ]
                    planned = len(members)
                    tasks = [self._forecast_member(question, research, spec, run, state) for spec, run in members]
                    predictions, errors, _ = await self._gather_results_and_exceptions(tasks)
                state.members_ok = len(predictions)
                needed = max(1, math.ceil(self.settings.min_success_fraction * planned))
                if planned >= 3:
                    needed = max(needed, 2)
                if len(predictions) < needed:
                    raise ForecastFailure(
                        f"Only {len(predictions)} of {planned} forecasters succeeded "
                        f"(need {needed}). Errors: {errors[:4]}"
                    )
        finally:
            _CURRENT_STATE.reset(token)

        summary = self._summary_line(state, errors)
        return ResearchWithPredictions(
            research_report=research,
            summary_report=summary,
            errors=errors,
            predictions=predictions,
        )

    def _summary_line(self, state: QState, errors: list[str]) -> str:
        sources = ", ".join(f"{s.name} [{'ok' if s.ok else 'failed'}]" for s in state.sources)
        return (
            f"Profile: {state.profile.name}. Forecasters succeeded: {state.members_ok}/"
            f"{state.members_planned}. Research sources: {sources or 'none'}. "
            f"Aggregation: trimmed mean ({self.settings.trim_fraction:.0%} each side)."
            + (f" Member errors: {len(errors)}." if errors else "")
        )

    def _prompt_for(self, ctx: QCtx, research: str) -> str:
        now = utcnow()
        if ctx.kind == "binary":
            return binary_prompt(ctx, research, now)
        if ctx.kind == "multiple_choice":
            return multiple_choice_prompt(ctx, research, now)
        return numeric_prompt(ctx, research, now)

    @staticmethod
    def _parse(kind: str, text: str, ctx: QCtx):
        if kind == "binary":
            return parse_binary(text)
        if kind == "multiple_choice":
            return parse_multiple_choice(text, ctx.options)
        return parse_percentiles(text, is_date=kind == "date")

    async def _parse_with_llm(self, kind: str, text: str, ctx: QCtx, state: QState):
        try:
            result = await self.client.complete(
                PARSER,
                parser_prompt(kind, text, ctx),
                system=SYSTEM_PARSER,
                budget=state.budget,
                deadline=state.deadline,
                json_mode=True,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Fallback parser failed: %s", type(exc).__name__)
            return None
        if kind == "binary":
            for obj in reversed(extract_json_objects(result.text)):
                if isinstance(obj, dict) and "probability" in obj:
                    value = to_float(obj["probability"])  # percent by contract
                    if value is not None and 0 <= value <= 100:
                        return value / 100.0
            return None
        return self._parse(kind, result.text, ctx)

    async def _forecast_member(
        self,
        question: MetaculusQuestion,
        research: str,
        spec: ModelSpec,
        run: int,
        state: QState,
    ) -> ReasonedPrediction:
        ctx = ctx_from_question(question)
        prompt = self._prompt_for(ctx, research)
        try:
            result = await self.client.complete(
                spec, prompt, system=SYSTEM_FORECASTER, budget=state.budget, deadline=state.deadline
            )
        except LlmError as exc:
            if exc.status == 402:
                self.credits_exhausted = True
            raise
        value = self._parse(ctx.kind, result.text, ctx)
        if value is None:
            value = await self._parse_with_llm(ctx.kind, result.text, ctx, state)
        if value is None:
            raise ForecastFailure(f"Could not read a final answer from {spec.model}")
        prediction = self._to_prediction(ctx, value, question)
        reasoning = f"Model: {spec.model} (run {run + 1})\n\n{result.text[:8000]}"
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    def _to_prediction(self, ctx: QCtx, value: Any, question: MetaculusQuestion):
        if ctx.kind == "binary":
            return float(min(max(value, 0.001), 0.999))
        if ctx.kind == "multiple_choice":
            probs = normalize_with_floor(value, self.settings.mc_floor)
            return PredictedOptionList(
                predicted_options=[
                    PredictedOption(option_name=o, probability=probs[o]) for o in ctx.options
                ]
            )
        scale = Scale.from_ctx(ctx)
        issue = plausibility_issue(value, scale)
        if issue:
            raise ForecastFailure(f"Implausible distribution: {issue}")
        cdf = build_cdf(value, scale)
        validate_cdf(cdf, scale)
        return self._distribution_from_cdf(cdf, scale, question, standardize=False)

    @staticmethod
    def _distribution_from_cdf(
        cdf: np.ndarray, scale: Scale, question: MetaculusQuestion, standardize: bool | None
    ) -> NumericDistribution:
        values = scale.grid_values()
        points = [
            Percentile(percentile=float(p), value=float(v)) for v, p in zip(values, np.asarray(cdf))
        ]
        return NumericDistribution.from_question(points, question, standardize_cdf=standardize)

    # Single-member entry points (used for conditional sub-questions).
    def _single_spec(self) -> ModelSpec:
        state = _CURRENT_STATE.get()
        profile = state.profile if state else PROFILES["economy"]
        return profile.forecasters[0].with_runs(1)

    async def _single(self, question: MetaculusQuestion, research: str) -> ReasonedPrediction:
        state = _CURRENT_STATE.get()
        if state is None:
            state = self._new_state(question)
        return await self._forecast_member(question, research, self._single_spec(), 0, state)

    async def _run_forecast_on_binary(self, question: BinaryQuestion, research: str):  # type: ignore[override]
        return await self._single(question, research)

    async def _run_forecast_on_multiple_choice(self, question: MultipleChoiceQuestion, research: str):  # type: ignore[override]
        return await self._single(question, research)

    async def _run_forecast_on_numeric(self, question, research: str):  # type: ignore[override]
        return await self._single(question, research)

    async def _run_forecast_on_date(self, question, research: str):  # type: ignore[override]
        return await self._single(question, research)

    # ------------------------------------------------------------ conditional
    # Adapted from forecasting-tools' template_bot_2026_fall.py (MIT licence).
    async def _run_forecast_on_conditional(self, question: ConditionalQuestion, research: str):  # type: ignore[override]
        parent_info, full_research = await self._get_question_prediction_info(question.parent, research, "parent")
        child_info, full_research = await self._get_question_prediction_info(question.child, research, "child")
        yes_info, full_research = await self._get_question_prediction_info(question.question_yes, full_research, "yes")
        no_info, full_research = await self._get_question_prediction_info(question.question_no, full_research, "no")
        full_reasoning = (
            f"## Parent Question Reasoning\n{parent_info.reasoning}\n"
            f"## Child Question Reasoning\n{child_info.reasoning}\n"
            f"## Yes Question Reasoning\n{yes_info.reasoning}\n"
            f"## No Question Reasoning\n{no_info.reasoning}"
        )
        prediction = ConditionalPrediction(
            parent=parent_info.prediction_value,  # type: ignore[arg-type]
            child=child_info.prediction_value,  # type: ignore[arg-type]
            prediction_yes=yes_info.prediction_value,  # type: ignore[arg-type]
            prediction_no=no_info.prediction_value,  # type: ignore[arg-type]
        )
        return ReasonedPrediction(reasoning=full_reasoning, prediction_value=prediction)

    async def _get_question_prediction_info(
        self, question: MetaculusQuestion, research: str, question_type: str
    ):
        previous = question.previous_forecasts
        if (
            question_type in ["parent", "child"]
            and previous
            and question_type not in self.force_reforecast_in_conditional
        ):
            last = previous[-1]
            if last.timestamp_end is None or last.timestamp_end > utcnow():
                pretty = DataOrganizer.get_readable_prediction(last)  # type: ignore[arg-type]
                return (
                    ReasonedPrediction(
                        prediction_value=PredictionAffirmed(),
                        reasoning=f"Already existing forecast reaffirmed at {pretty}.",
                    ),
                    research,
                )
        info = await self._single(question, research)
        readable = DataOrganizer.get_readable_prediction(info.prediction_value)
        label = question_type.title()
        full_research = (
            f"{research}\n---\n## {label} Question Information\n"
            f"You have previously forecasted the {label} Question to the value: {readable}\n"
            "This is relevant information for your current forecast, but it is NOT your "
            "current forecast.\nThe reasoning for that question was:\n```\n"
            f"{info.reasoning}\n```\nDo NOT use this reasoning to re-forecast the {label} question."
        )
        return info, full_research

    # ------------------------------------------------------------ aggregation
    async def _aggregate_predictions(self, predictions, question: MetaculusQuestion):  # type: ignore[override]
        if isinstance(question, ConditionalQuestion) or any(
            isinstance(p, ConditionalPrediction) for p in predictions
        ):
            return await super()._aggregate_predictions(predictions, question)
        s = self.settings
        ctx = ctx_from_question(question)
        if ctx.kind == "binary":
            return aggregate_binary(
                [float(p) for p in predictions],
                trim=s.trim_fraction,
                clip_low=s.binary_clip_low,
                clip_high=s.binary_clip_high,
                platt_a=s.platt_a,
                platt_b=s.platt_b,
            )
        if ctx.kind == "multiple_choice":
            pooled = aggregate_multiple_choice(
                [p.to_dict() for p in predictions], ctx.options, trim=s.trim_fraction, floor=s.mc_floor
            )
            return PredictedOptionList(
                predicted_options=[PredictedOption(option_name=o, probability=pooled[o]) for o in ctx.options]
            )
        scale = Scale.from_ctx(ctx)
        cdfs = [
            np.asarray([pt.percentile for pt in p.declared_percentiles], dtype=float)
            for p in predictions
            if len(p.declared_percentiles) == scale.cdf_size
        ]
        if not cdfs:
            raise ForecastFailure("No member CDF matches the question grid")
        pooled = aggregate_cdfs(cdfs, trim=s.trim_fraction)
        # Default standardisation here: forecasting-tools' final safety net.
        return self._distribution_from_cdf(pooled, scale, question, standardize=None)


def asknews_available() -> bool:
    return AskNewsClient.configured()


def safe_prediction_text(report: Any) -> str:
    """Readable final prediction (only printed when LOG_FORECASTS is on)."""
    try:
        return type(report).make_readable_prediction(report.prediction).strip()
    except Exception:  # noqa: BLE001
        return "?"


async def close_quietly(client: OpenRouterClient) -> None:
    try:
        await client.aclose()
    except Exception:  # noqa: BLE001
        pass


__all__ = [
    "EnsembleBot",
    "ForecastFailure",
    "QState",
    "asknews_available",
    "close_quietly",
    "report_cost_to_forecasting_tools",
    "safe_prediction_text",
    "utcnow",
]
