"""Run modes used by main.py.

Everything printed here is safe for public GitHub logs: question URLs,
counts, costs and errors, but never forecasts or reasoning (unless
LOG_FORECASTS=true, meant for local dry runs only).
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from .budget import Budget, CreditInfo, choose_profile, expected_remaining_questions
from .config import PROFILES, Settings, all_model_specs
from .llm import OpenRouterClient
from .state import RunState

logger = logging.getLogger("metabot")
BANNER = "=" * 78


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class TargetResult:
    kind: str
    tournament: str
    profile: str
    fetched: int = 0
    selected: int = 0
    ok: int = 0
    failed: int = 0
    skipped_attempts: int = 0
    cost: float = 0.0
    failures: list[str] = field(default_factory=list)
    error: str | None = None


def _metaculus_client():
    from forecasting_tools import MetaculusClient

    return MetaculusClient()


async def fetch_open_questions(tournament: str) -> list:
    client = _metaculus_client()
    return await asyncio.to_thread(client.get_all_open_questions_from_tournament, tournament)


async def fetch_urls(urls: list[str]) -> list:
    client = _metaculus_client()
    questions: list = []
    for url in urls:
        found = await asyncio.to_thread(client.get_question_by_url, url, "unpack_subquestions")
        questions.extend(found if isinstance(found, list) else [found])
    return questions


def diverse_subset(questions: list, n: int) -> list:
    """Pick up to n questions, one of each type first (for smoke tests)."""
    by_type: dict[str, list] = {}
    for q in questions:
        by_type.setdefault(type(q).__name__, []).append(q)
    picked: list = []
    while len(picked) < n and any(by_type.values()):
        for bucket in by_type.values():
            if bucket and len(picked) < n:
                picked.append(bucket.pop(0))
    return picked


def _short(exc: BaseException, limit: int = 240) -> str:
    text = f"{type(exc).__name__}: {exc}"
    return text if len(text) <= limit else text[:limit] + "..."


async def forecast_targets(
    targets: list[tuple[str, str, list | None]],
    settings: Settings,
    *,
    dry_run: bool,
    profile_override: str | None = None,
    max_questions: int | None = None,
    diverse: bool = False,
    save_folder: str | None = None,
    skip_previously_forecasted: bool = True,
    state: RunState | None = None,
) -> tuple[list[TargetResult], float, CreditInfo | None]:
    """Forecast every (kind, tournament, questions-or-None) target."""
    from forecasting_tools import ForecastReport

    from .bot import EnsembleBot, report_cost_to_forecasting_tools, safe_prediction_text

    state = state or RunState()
    client = OpenRouterClient(
        os.environ.get("OPENROUTER_API_KEY", ""),
        max_concurrency=settings.max_concurrent_llm_calls,
        on_cost=report_cost_to_forecasting_tools,
    )
    results: list[TargetResult] = []
    credit: CreditInfo | None = None
    try:
        credit = CreditInfo.from_api(await client.key_info())
        remaining = credit.limit_remaining if credit else None
        run_budget = Budget(settings.max_cost_per_run_usd, name="run")
        bot = EnsembleBot(
            settings=settings,
            client=client,
            run_budget=run_budget,
            dry_run=dry_run,
            save_folder=save_folder,
            skip_previously_forecasted=skip_previously_forecasted,
        )
        for kind, tournament, questions in targets:
            profile = profile_override or choose_profile(kind, remaining, utcnow(), settings)
            result = TargetResult(kind=kind, tournament=tournament, profile=profile)
            results.append(result)
            if profile == "stop":
                result.error = (
                    f"OpenRouter credit below the reserve (${settings.credit_reserve_usd:.0f}); "
                    "not forecasting"
                )
                continue
            try:
                if questions is None:
                    questions = await fetch_open_questions(tournament)
            except Exception as exc:  # noqa: BLE001
                result.error = "could not fetch questions: " + _short(exc)
                continue
            result.fetched = len(questions)

            pending = []
            for q in questions:
                if not dry_run and not q.already_forecasted and state.exhausted(q.id_of_question):
                    result.skipped_attempts += 1
                    continue
                pending.append(q)
            if max_questions is not None:
                pending = diverse_subset(pending, max_questions) if diverse else pending[:max_questions]

            bot.kind_profiles[kind] = profile
            bot.current_kind = kind
            spent_before = client.total_cost
            reports = await bot.forecast_questions(pending, return_exceptions=True)
            selected = bot.last_selected
            result.selected = len(selected)
            result.cost = client.total_cost - spent_before

            for question, report in zip(selected, reports):
                qid = question.id_of_question
                qstate = bot.states.get(qid) if qid is not None else None
                if isinstance(report, ForecastReport):
                    result.ok += 1
                    cost = report.price_estimate or 0.0
                    state.record_success(qid, cost, kind, profile)
                    members = f"{qstate.members_ok}/{qstate.members_planned}" if qstate else "?"
                    line = (
                        f"OK   {question.page_url} [{kind}/{qstate.profile.name if qstate else profile}] "
                        f"cost ${cost:.3f}, {report.minutes_taken or 0:.1f} min, forecasters {members}"
                    )
                    if settings.log_forecasts:
                        line += f" -> {safe_prediction_text(report)}"
                    print(line)
                else:
                    result.failed += 1
                    state.record_failure(qid)
                    message = f"{question.page_url}: {_short(report)}"
                    result.failures.append(message)
                    print(f"FAIL {message}")
            if bot.credits_exhausted:
                logger.error("OpenRouter returned 402 (no credit). Stopping this run.")
                break
    finally:
        await client.aclose()
        state.save()
    return results, client.total_cost, credit


def print_summary(
    mode: str,
    publish: bool,
    results: list[TargetResult],
    total_cost: float,
    credit: CreditInfo | None,
) -> int:
    """Print a safe summary; return the process exit code."""
    print()
    print(BANNER)
    print(f"Run summary: mode={mode}, publish={'yes' if publish else 'no (dry run)'}")
    exit_code = 0
    for r in results:
        if r.error:
            exit_code = 1
            print(f"- {r.kind} ({r.tournament}): ERROR {r.error}")
            continue
        print(
            f"- {r.kind} ({r.tournament}): profile={r.profile}, open={r.fetched}, "
            f"forecast now={r.selected}, ok={r.ok}, failed={r.failed}, "
            f"gave up after retries={r.skipped_attempts}, cost=${r.cost:.2f}"
        )
        if r.failed:
            exit_code = 1
    print(f"Total OpenRouter cost this run: ${total_cost:.3f}")
    if credit and credit.limit_remaining is not None:
        print(f"OpenRouter credit remaining (before this run): ${credit.limit_remaining:.2f}")
    print(BANNER)
    return exit_code


async def coverage_report(settings: Settings, days: int) -> int:
    """How many recently closed tournament questions did the bot forecast?"""
    from forecasting_tools import ApiFilter

    client = _metaculus_client()
    since = utcnow() - timedelta(days=days)
    worst = 1.0
    print(BANNER)
    print(f"Coverage of questions closed in the last {days} day(s)")
    for kind, tournament in (
        ("seasonal", settings.seasonal_tournament),
        ("minibench", settings.minibench_tournament),
    ):
        api_filter = ApiFilter(
            allowed_tournaments=[tournament],
            allowed_statuses=["closed", "resolved"],
            close_time_gt=since,
            group_question_mode="unpack_subquestions",
        )
        try:
            questions = await client.get_questions_matching_filter(
                api_filter, num_questions=500, error_if_question_target_missed=False
            )
        except Exception as exc:  # noqa: BLE001
            print(f"- {kind}: could not fetch ({_short(exc)})")
            continue
        total = len(questions)
        done = sum(1 for q in questions if q.already_forecasted)
        share = done / total if total else 1.0
        worst = min(worst, share)
        print(f"- {kind} ({tournament}): forecast {done}/{total} ({share:.0%})")
        for q in questions:
            if not q.already_forecasted:
                print(f"    missed: {q.page_url} (closed {q.close_time})")
    print(BANNER)
    return 0 if worst >= 0.9 else 1


async def budget_report(settings: Settings) -> int:
    client = OpenRouterClient(os.environ.get("OPENROUTER_API_KEY", ""))
    try:
        credit = CreditInfo.from_api(await client.key_info())
    finally:
        await client.aclose()
    now = utcnow()
    seasonal_left, minibench_left = expected_remaining_questions(now, settings)
    print(BANNER)
    if credit is None:
        print("Could not read the OpenRouter key (is OPENROUTER_API_KEY valid?)")
        print(BANNER)
        return 1
    print(f"OpenRouter key: limit={credit.limit}, remaining={credit.limit_remaining}, used={credit.usage}")
    print(f"Expected questions left: seasonal≈{seasonal_left:.0f}, minibench≈{minibench_left:.0f}")
    for kind in ("seasonal", "minibench", "test"):
        name = choose_profile(kind, credit.limit_remaining, now, settings)
        cost = PROFILES[name].expected_cost if name in PROFILES else 0
        print(f"- {kind}: profile={name} (≈${cost:.2f}/question)")
    print(BANNER)
    return 0


async def models_check(settings: Settings) -> int:
    """Ping every configured model (cost: a few cents)."""
    client = OpenRouterClient(os.environ.get("OPENROUTER_API_KEY", ""))
    budget = Budget(3.0, name="models-check")
    failures = 0
    print(BANNER)
    try:
        for spec in all_model_specs():
            try:
                res = await client.complete(
                    spec,
                    "Reply with the single word OK.",
                    reasoning_effort="low",
                    max_tokens=1500,
                    budget=budget,
                )
                print(f"OK   {spec.model:<34} answered by {res.model} (${res.cost:.4f})")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"FAIL {spec.model:<34} {_short(exc)}")
    finally:
        await client.aclose()
    print(f"Total cost: ${client.total_cost:.4f}")
    print(BANNER)
    return 1 if failures else 0


def describe_config(settings: Settings) -> dict[str, Any]:
    return {
        "seasonal": settings.seasonal_tournament,
        "minibench": settings.minibench_tournament,
        "profiles": {k: [f"{s.model} x{s.runs}" for s in p.forecasters] for k, p in PROFILES.items()},
    }
