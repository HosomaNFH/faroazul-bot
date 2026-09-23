"""Library-independent view of a Metaculus question.

``QCtx`` holds everything prompts, parsers and the CDF builder need. It is
built from a forecasting-tools question by duck typing, so the pure modules
(and their tests) never import forecasting-tools.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

KINDS = ("binary", "multiple_choice", "numeric", "discrete", "date")


def _ts(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.timestamp()
    return float(value)


def _dt(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    return None


@dataclass
class QCtx:
    kind: str
    title: str
    background: str = ""
    criteria: str = ""
    fine_print: str = ""
    url: str = ""
    options: list[str] = field(default_factory=list)
    unit: str = ""
    # Numeric/date range. For dates these are UNIX timestamps (seconds).
    lower: float | None = None
    upper: float | None = None
    nominal_lower: float | None = None
    nominal_upper: float | None = None
    open_lower: bool = False
    open_upper: bool = False
    zero_point: float | None = None
    cdf_size: int = 201
    open_time: datetime | None = None
    close_time: datetime | None = None
    resolve_time: datetime | None = None
    group_option: str | None = None
    conditional_type: str | None = None  # "yes"/"no" for conditional children

    @property
    def is_continuous_like(self) -> bool:
        return self.kind in {"numeric", "discrete", "date"}

    @property
    def range_width(self) -> float:
        if self.lower is None or self.upper is None:
            return 0.0
        return float(self.upper) - float(self.lower)

    def discrete_step(self) -> float | None:
        """Spacing between possible outcomes for discrete questions."""
        if self.kind != "discrete" or self.cdf_size < 3:
            return None
        if self.nominal_lower is not None and self.nominal_upper is not None:
            return (self.nominal_upper - self.nominal_lower) / (self.cdf_size - 2)
        return self.range_width / (self.cdf_size - 1)


def kind_of(question: Any) -> str:
    """Map a forecasting-tools question object to one of KINDS."""
    q_type = getattr(question, "question_type", None)
    if q_type in KINDS:
        return q_type
    name = type(question).__name__
    mapping = {
        "BinaryQuestion": "binary",
        "MultipleChoiceQuestion": "multiple_choice",
        "NumericQuestion": "numeric",
        "DiscreteQuestion": "discrete",
        "DateQuestion": "date",
    }
    if name in mapping:
        return mapping[name]
    raise ValueError(f"Unsupported question type: {name} ({q_type})")


def ctx_from_question(question: Any) -> QCtx:
    kind = kind_of(question)
    ctx = QCtx(
        kind=kind,
        title=getattr(question, "question_text", "") or "",
        background=getattr(question, "background_info", "") or "",
        criteria=getattr(question, "resolution_criteria", "") or "",
        fine_print=getattr(question, "fine_print", "") or "",
        url=getattr(question, "page_url", "") or "",
        unit=getattr(question, "unit_of_measure", "") or "",
        open_time=_dt(getattr(question, "open_time", None)),
        close_time=_dt(getattr(question, "close_time", None)),
        resolve_time=_dt(getattr(question, "scheduled_resolution_time", None)),
        group_option=getattr(question, "group_question_option", None),
        conditional_type=getattr(question, "conditional_type", None),
    )
    if kind == "multiple_choice":
        ctx.options = list(getattr(question, "options", []) or [])
    if kind in {"numeric", "discrete", "date"}:
        ctx.lower = _ts(getattr(question, "lower_bound", None))
        ctx.upper = _ts(getattr(question, "upper_bound", None))
        ctx.open_lower = bool(getattr(question, "open_lower_bound", False))
        ctx.open_upper = bool(getattr(question, "open_upper_bound", False))
        ctx.zero_point = getattr(question, "zero_point", None)
        ctx.cdf_size = int(getattr(question, "cdf_size", 201) or 201)
        if kind != "date":
            ctx.nominal_lower = getattr(question, "nominal_lower_bound", None)
            ctx.nominal_upper = getattr(question, "nominal_upper_bound", None)
    return ctx
