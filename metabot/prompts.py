"""Prompt templates.

Design notes (see Diseño.md for sources):
* Every prompt states today's date and the question's timeline.
* Resolution criteria and fine print are quoted verbatim and the model must
  restate the resolution mechanics before forecasting.
* Explicit outside view (reference classes and a numeric base rate) before
  the inside view; a short pre-mortem; guidance against extreme tails.
* The question is still open: an explicit guard against the common failure of
  treating an open question as already resolved.
* No "Bayesian" framing (it underperformed in Metaculus' prompt experiments).
"""

from __future__ import annotations

from datetime import datetime, timezone

from .cdf import STANDARD_PERCENTILES
from .qctx import QCtx

SYSTEM_FORECASTER = (
    "You are a meticulous, well-calibrated forecaster competing in a tournament "
    "scored with a proper log scoring rule relative to other forecasters. You read "
    "resolution criteria literally, start from base rates, weigh evidence by its "
    "reliability and recency, and express uncertainty honestly. Confident mistakes "
    "are punished severely, and so is timid hedging when the evidence is strong."
)

SYSTEM_RESEARCHER = (
    "You are a careful research assistant for a professional forecaster. You search "
    "the web, report facts with sources and dates, flag uncertainty, and never give "
    "your own forecast."
)

SYSTEM_PARSER = (
    "You convert a forecaster's written answer into JSON. You copy the forecaster's "
    "final numbers exactly and never invent or change them."
)


# --------------------------------------------------------------------------- #
# formatting helpers
# --------------------------------------------------------------------------- #


def fmt_num(x: float | None) -> str:
    if x is None:
        return "?"
    x = float(x)
    if x != 0 and (abs(x) >= 1e15 or abs(x) < 1e-4):
        return repr(x)
    text = f"{x:.6f}".rstrip("0").rstrip(".")
    return text if text not in {"", "-0"} else "0"


def fmt_dt(dt: datetime | None) -> str:
    if dt is None:
        return "unknown"
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def fmt_date_from_ts(ts: float | None) -> str:
    if ts is None:
        return "?"
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")


def _hours_between(a: datetime, b: datetime) -> float:
    return (b - a).total_seconds() / 3600.0


def timeline_block(ctx: QCtx, now: datetime) -> str:
    lines = [f"- Today: {fmt_dt(now)}"]
    if ctx.open_time:
        lines.append(f"- Question opened: {fmt_dt(ctx.open_time)}")
    if ctx.close_time:
        hours = _hours_between(now, ctx.close_time)
        lines.append(f"- Forecasting closes: {fmt_dt(ctx.close_time)} (in {hours:.1f} hours)")
    if ctx.resolve_time:
        days = _hours_between(now, ctx.resolve_time) / 24.0
        lines.append(
            f"- Scheduled resolution: {fmt_dt(ctx.resolve_time)} ({days:.1f} days from today)"
        )
    return "\n".join(lines)


def question_block(ctx: QCtx) -> str:
    parts = [f"QUESTION\n{ctx.title.strip()}"]
    if ctx.group_option:
        parts.append(
            "This is one sub-question of a group of related questions. It is "
            f'specifically about: "{ctx.group_option}". Forecast only this sub-question.'
        )
    if ctx.conditional_type in {"yes", "no"}:
        parts.append(
            "This is the child question of a conditional pair. Forecast only the child "
            "question, assuming the parent question's resolution stated in the text. "
            "Never re-forecast the parent question."
        )
    parts.append(
        "RESOLUTION CRITERIA (these decide the outcome; read them literally)\n"
        + (ctx.criteria.strip() or "(not provided)")
    )
    parts.append("FINE PRINT\n" + (ctx.fine_print.strip() or "(none)"))
    parts.append("BACKGROUND\n" + (ctx.background.strip() or "(none)"))
    return "\n\n".join(parts)


def research_block(research: str) -> str:
    research = research.strip() or "(no research available: rely on your own knowledge)"
    return (
        "RESEARCH (gathered minutes ago by automated assistants; it can contain "
        "errors, stale or irrelevant items; check dates and sources)\n" + research
    )


def range_block(ctx: QCtx) -> str:
    if ctx.kind == "date":
        lo, hi = fmt_date_from_ts(ctx.lower), fmt_date_from_ts(ctx.upper)
        unit = ""
    else:
        lo_val = ctx.nominal_lower if ctx.nominal_lower is not None else ctx.lower
        hi_val = ctx.nominal_upper if ctx.nominal_upper is not None else ctx.upper
        lo, hi = fmt_num(lo_val), fmt_num(hi_val)
        unit = f" {ctx.unit}".rstrip() if ctx.unit else ""
    lines = []
    if ctx.kind != "date":
        lines.append(
            "UNITS: " + (ctx.unit if ctx.unit else "not stated; infer them from the question")
        )
    lines.append(f"RANGE shown on Metaculus: {lo}{unit} to {hi}{unit}.")
    if ctx.open_lower:
        lines.append(
            f"- The lower end is OPEN: outcomes below {lo}{unit} are possible. If you think "
            "that is plausible, place your low percentiles below it."
        )
    else:
        lines.append(f"- The lower end is CLOSED: the outcome cannot be below {lo}{unit}.")
    if ctx.open_upper:
        lines.append(
            f"- The upper end is OPEN: outcomes above {hi}{unit} are possible. If you think "
            "that is plausible, place your high percentiles above it."
        )
    else:
        lines.append(f"- The upper end is CLOSED: the outcome cannot be above {hi}{unit}.")
    step = ctx.discrete_step()
    if step:
        lines.append(
            f"- This is a DISCRETE question: possible outcomes go from {lo} to {hi} in "
            f"steps of {fmt_num(step)}."
        )
    if ctx.zero_point is not None and ctx.kind != "date":
        lines.append("- The range is displayed on a logarithmic scale.")
    return "\n".join(lines)


def _percentile_keys() -> list[str]:
    return [f"{p * 100:g}" for p in STANDARD_PERCENTILES]


# --------------------------------------------------------------------------- #
# research prompts
# --------------------------------------------------------------------------- #


def research_prompt(ctx: QCtx, now: datetime) -> str:
    extra = ""
    if ctx.is_continuous_like:
        extra = (
            "\n6. For this quantitative question: the latest value of the underlying "
            "quantity with its date and source, its recent trend, and how much it has "
            "varied historically over a comparable horizon.\n\n" + range_block(ctx)
        )
    if ctx.kind == "multiple_choice":
        extra = "\n6. Evidence that bears on each of these options: " + "; ".join(ctx.options)
    return f"""Gather the most decision-relevant, up-to-date facts for the forecasting question below. Use web search. Do NOT give a forecast or a probability.

TIMELINE
{timeline_block(ctx, now)}

{question_block(ctx)}

Search for, and report with sources and dates:
1. The current status of whatever the resolution depends on. If the criteria name a resolution source, check that source first.
2. The most important recent developments (last 1-4 weeks) and any scheduled events before the resolution date.
3. Base rates: how often comparable events have happened historically; similar past cases and how they turned out.
4. What others expect: prediction markets (e.g. Polymarket, Kalshi, Manifold), expert or analyst forecasts, polls, official projections.
5. Anything suggesting the question could resolve early, or that the criteria might be met or missed on a technicality.{extra}

Write a factual brief of at most about 600 words, in bullet points, each with (source, date). Say explicitly when you could not find something. Distinguish confirmed facts from reports or rumours."""


def knowledge_prompt(ctx: QCtx, now: datetime) -> str:
    return f"""Live web search is unavailable. From your own knowledge, write a factual brief (at most 400 words) for the forecasting question below: relevant background, base rates, key actors and scheduled events. State your knowledge cutoff and flag anything that may have changed since. Do NOT give a forecast.

TIMELINE
{timeline_block(ctx, now)}

{question_block(ctx)}"""


# --------------------------------------------------------------------------- #
# forecasting prompts
# --------------------------------------------------------------------------- #

_OPEN_QUESTION_GUARD = (
    "This question is still open on Metaculus, so it has NOT been resolved yet. If "
    "something in the research looks like it already satisfies the criteria, check its "
    "date against the question window and the exact wording before relying on it; "
    "events before the question opened, or reports from sources other than the named "
    "resolution source, often do not count."
)


def binary_prompt(ctx: QCtx, research: str, now: datetime) -> str:
    return f"""TIMELINE
{timeline_block(ctx, now)}

{question_block(ctx)}

{research_block(research)}

INSTRUCTIONS
Work through these steps, with concise notes for each:
1. Resolution mechanics: restate exactly what must happen, by when, and according to which source, for the question to resolve YES. Note edge cases, ambiguities, and anything in the fine print that changes the naive reading.
2. Current status: where things stand today relative to those criteria. {_OPEN_QUESTION_GUARD}
3. Outside view: name one or more reference classes and give a numeric base rate (for example, how often events like this happen within a window of this length, or how often the status quo persists over such a window).
4. Inside view: adjust from the base rate for the specific evidence, trends, scheduled events and the time remaining. The status quo usually persists over short windows, but surprises do happen.
5. Pre-mortem: the most plausible way your forecast turns out wrong, in each direction.
6. Final answer. Use values below 3% or above 97% only when the outcome is essentially locked in by the rules and the timeline.

End your response with a line in exactly this format:
Probability: XX%"""


def multiple_choice_prompt(ctx: QCtx, research: str, now: datetime) -> str:
    options = "\n".join(f"{i + 1}. {o}" for i, o in enumerate(ctx.options))
    example = ", ".join(f'"{o}": ...' for o in ctx.options[:3])
    return f"""TIMELINE
{timeline_block(ctx, now)}

{question_block(ctx)}

OPTIONS (use these names exactly)
{options}

{research_block(research)}

INSTRUCTIONS
Work through these steps, with concise notes for each:
1. Resolution mechanics: restate how the winning option will be determined, by when and from which source. Note edge cases and ties.
2. Current status of each option. {_OPEN_QUESTION_GUARD}
3. Outside view: base rates or reference classes relevant to the options (for example, how often the current leader stays ahead over a window of this length).
4. Inside view: adjust for the specific evidence, trends, scheduled events and time remaining.
5. Pre-mortem: which option could surprise, and how.
6. Final answer: a probability for every option, summing to 100. Give every option at least 1% unless the resolution criteria make it logically impossible.

End your response with a JSON block containing every option name exactly as written above, with probabilities in percent:
```json
{{"probabilities": {{{example}}}}}
```"""


def numeric_prompt(ctx: QCtx, research: str, now: datetime) -> str:
    keys = _percentile_keys()
    if ctx.kind == "date":
        value_rules = (
            "- Values are dates in the format YYYY-MM-DD (UTC).\n"
            "- Dates must be in chronological order: the 1st percentile is the earliest date."
        )
        example_value = '"YYYY-MM-DD"'
        what = "date"
    else:
        value_rules = (
            f"- Values are plain numbers in the question's units ({ctx.unit or 'as inferred'}): "
            "no thousands separators, no words like 'million', no scientific notation "
            "(write 1250000, not 1.25M or 1.25e6). If the units are for example 'billion $', "
            "write 0.5 for 500 million dollars.\n"
            "- Values must strictly increase from the 1st to the 99th percentile."
        )
        example_value = "..."
        what = "value"
    example = ", ".join(f'"{k}": {example_value}' for k in keys)
    return f"""TIMELINE
{timeline_block(ctx, now)}

{question_block(ctx)}

{range_block(ctx)}

{research_block(research)}

INSTRUCTIONS
Work through these steps, with concise notes for each:
1. Resolution mechanics: restate exactly which {what} resolves the question, from which source, measured how and when (units, rounding, revisions, definitions).
2. Current status: the latest known data point and its date. {_OPEN_QUESTION_GUARD}
3. Outside view: what would happen if nothing changed; what the historical variability over a comparable horizon implies; base rates from similar cases.
4. Inside view: trends, scheduled events, expert and market expectations; how they shift the distribution.
5. Tails: describe a plausible scenario for a surprisingly low and a surprisingly high outcome. Your 1st and 99th percentiles should be values you would be genuinely surprised (about 1 in 100 on each side) to see exceeded.
6. Final answer: your percentiles.

Formatting rules:
{value_rules}

End your response with a JSON block with exactly these percentile keys:
```json
{{"percentiles": {{{example}}}}}
```"""


def parser_prompt(kind: str, answer_text: str, ctx: QCtx) -> str:
    if kind == "binary":
        schema = '{"probability": <number between 0 and 100>}'
        extra = "Use the forecaster's FINAL probability."
    elif kind == "multiple_choice":
        names = ", ".join(f'"{o}"' for o in ctx.options)
        schema = '{"probabilities": {"<option name>": <percent>, ...}}'
        extra = f"Use exactly these option names: {names}. Include every option."
    else:
        keys = ", ".join(f'"{k}"' for k in _percentile_keys())
        if kind == "date":
            schema = '{"percentiles": {"<percentile>": "YYYY-MM-DD", ...}}'
        else:
            schema = '{"percentiles": {"<percentile>": <number>, ...}}'
        extra = (
            f"Percentile keys: {keys} (use the ones the forecaster gave). Values must be in "
            f"these units: {ctx.unit or 'the units used by the forecaster'}; convert words like "
            "'million' into plain numbers in those units."
        )
    return f"""Extract the forecaster's final answer from the text below and return ONLY JSON of the form {schema}. {extra}
If the text does not contain a final answer, or it treats the question as already resolved instead of forecasting it, return {{"error": "missing"}}.

TEXT:
<<<
{answer_text[-12000:]}
>>>"""
