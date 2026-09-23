"""Cost control.

Two mechanisms:

1. ``Budget``: reserve-then-settle accounting. Before each LLM call we reserve
   its *worst-case* cost; after the call we release the reservation and record
   the *actual* cost reported by OpenRouter. Budgets can be chained
   (question -> run), so neither the per-question cap nor the per-run cap can
   be exceeded even when many calls run concurrently. asyncio runs on a single
   thread and these methods never await, so no lock is needed.

2. ``choose_profile``: picks the most accurate profile that the remaining
   OpenRouter credit can sustain until the end of the season.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from .config import PROFILE_ORDER, PROFILES, Settings


class BudgetExceeded(RuntimeError):
    """Raised when a call cannot be afforded within the active caps."""


class Budget:
    def __init__(
        self, cap_usd: float | None, parent: "Budget | None" = None, name: str = ""
    ) -> None:
        if cap_usd is not None and cap_usd < 0:
            raise ValueError("cap_usd must be >= 0")
        self.cap = cap_usd
        self.parent = parent
        self.name = name
        self.spent = 0.0
        self.reserved = 0.0

    def _chain(self) -> list["Budget"]:
        chain: list[Budget] = []
        node: Budget | None = self
        while node is not None:
            chain.append(node)
            node = node.parent
        return chain

    def can_afford(self, amount: float) -> bool:
        for node in self._chain():
            if node.cap is not None and node.spent + node.reserved + amount > node.cap + 1e-12:
                return False
        return True

    def try_reserve(self, amount: float) -> bool:
        if amount < 0:
            raise ValueError("amount must be >= 0")
        if not self.can_afford(amount):
            return False
        for node in self._chain():
            node.reserved += amount
        return True

    def settle(self, reserved: float, actual: float) -> None:
        for node in self._chain():
            node.reserved = max(0.0, node.reserved - reserved)
            node.spent += max(0.0, actual)

    @property
    def remaining(self) -> float | None:
        if self.cap is None:
            return None
        return self.cap - self.spent - self.reserved


@dataclass(frozen=True)
class CreditInfo:
    """Subset of OpenRouter's GET /api/v1/key response."""

    limit: float | None
    limit_remaining: float | None
    usage: float | None

    @classmethod
    def from_api(cls, payload: dict | None) -> "CreditInfo | None":
        if not payload:
            return None
        data = payload.get("data", payload)

        def _num(key: str) -> float | None:
            value = data.get(key)
            return float(value) if isinstance(value, (int, float)) else None

        return cls(
            limit=_num("limit"),
            limit_remaining=_num("limit_remaining"),
            usage=_num("usage"),
        )


def expected_remaining_questions(now: datetime, settings: Settings) -> tuple[float, float]:
    """(seasonal, minibench) questions still expected this season."""
    start, end = settings.season_start, settings.season_end
    if now >= end:
        return 0.0, 0.0
    effective_now = max(now, start)
    fraction = (end - effective_now) / (end - start)
    seasonal = settings.expected_seasonal_questions * fraction
    weeks_left = (end - effective_now).total_seconds() / (7 * 24 * 3600)
    minibench = settings.expected_minibench_per_week * weeks_left
    return seasonal, minibench


def _cheaper(name: str) -> str | None:
    index = PROFILE_ORDER.index(name)
    return PROFILE_ORDER[index + 1] if index + 1 < len(PROFILE_ORDER) else None


def choose_profile(
    kind: str,
    limit_remaining: float | None,
    now: datetime,
    settings: Settings,
    safety_factor: float = 1.1,
) -> str:
    """Return a profile name, or "stop" when credit is below the reserve.

    kind: "seasonal", "minibench", "test" or "other".
    """
    if limit_remaining is not None and limit_remaining < settings.credit_reserve_usd:
        return "stop"

    if kind == "minibench":
        requested: str | None = settings.minibench_profile
    elif kind in {"test", "other"}:
        requested = settings.test_profile
    else:
        requested = settings.profile_override

    if limit_remaining is None:  # key without a spending limit
        return requested or "standard"

    seasonal_left, minibench_left = expected_remaining_questions(now, settings)
    available = limit_remaining - settings.credit_reserve_usd
    minibench_cost = PROFILES[settings.minibench_profile].expected_cost

    if kind == "seasonal" and requested is None:
        per_question = (available - minibench_left * minibench_cost) / max(
            seasonal_left, 25.0
        )
        for name in PROFILE_ORDER:
            if per_question >= PROFILES[name].expected_cost * safety_factor:
                return name
        return PROFILE_ORDER[-1]

    # Explicitly requested profile: downgrade only if clearly unaffordable.
    name = requested or "standard"
    per_question_all = available / max(seasonal_left + minibench_left, 25.0)
    while (
        PROFILES[name].expected_cost > per_question_all * 1.5
        and _cheaper(name) is not None
    ):
        name = _cheaper(name)  # type: ignore[assignment]
    return name
