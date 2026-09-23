"""Minimal async OpenRouter client.

Why not litellm/GeneralLlm: we need OpenRouter-specific features (reasoning
effort, the ``openrouter:web_search`` server tool, the real per-call cost in
``usage.cost``) and a hard per-question budget. A thin httpx client gives
exact control and exact costs.

Every call:
1. reserves its worst-case cost in the active ``Budget`` (question -> run);
2. retries transient failures (timeouts, 429, 5xx) with exponential backoff,
   never past the question deadline;
3. settles the reservation with the actual cost reported by OpenRouter and
   forwards that cost to an optional callback (forecasting-tools' cost
   manager), so reports show real spend.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import httpx

from .budget import Budget, BudgetExceeded
from .config import ModelSpec

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524, 529}


class LlmError(RuntimeError):
    def __init__(self, message: str, *, status: int | None = None, retryable: bool = False):
        super().__init__(message)
        self.status = status
        self.retryable = retryable


@dataclass
class LlmResult:
    text: str
    model: str
    cost: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    finish_reason: str | None = None
    citations: list[dict] = field(default_factory=list)
    elapsed_s: float = 0.0


def estimate_tokens(*texts: str | None) -> int:
    chars = sum(len(t) for t in texts if t)
    return int(chars / 3.5) + 50


def web_search_tool(max_uses: int, max_results: int = 5, context: str = "medium") -> list[dict]:
    return [
        {
            "type": "openrouter:web_search",
            "parameters": {
                "engine": "native",
                "max_results": max_results,
                "max_uses": max_uses,
                "search_context_size": context,
            },
        }
    ]


def web_search_plugin(max_results: int = 5) -> list[dict]:
    return [{"id": "web", "engine": "native", "max_results": max_results}]


def _message_text(message: dict) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                text = part.get("text") or part.get("content")
                if isinstance(text, str):
                    parts.append(text)
            elif isinstance(part, str):
                parts.append(part)
        return "\n".join(parts)
    return ""


def _citations(message: dict) -> list[dict]:
    out = []
    for ann in message.get("annotations") or []:
        if not isinstance(ann, dict):
            continue
        cit = ann.get("url_citation") or {}
        url = cit.get("url")
        if url:
            out.append({"url": url, "title": cit.get("title") or ""})
    return out


def parse_completion(data: dict, spec: ModelSpec, prompt_tokens_est: int) -> LlmResult:
    """Turn an OpenRouter JSON response into an LlmResult (raises LlmError)."""
    if "error" in data and data["error"]:
        err = data["error"]
        code = err.get("code") if isinstance(err, dict) else None
        message = err.get("message") if isinstance(err, dict) else str(err)
        status = int(code) if isinstance(code, int) or (isinstance(code, str) and code.isdigit()) else None
        raise LlmError(
            f"OpenRouter error ({code}): {message}",
            status=status,
            retryable=status in RETRYABLE_STATUS if status else True,
        )
    choices = data.get("choices") or []
    if not choices:
        raise LlmError("Response without choices", retryable=True)
    choice = choices[0]
    message = choice.get("message") or {}
    usage = data.get("usage") or {}
    cost = usage.get("cost")
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    if not isinstance(cost, (int, float)):
        # Fallback estimate from token counts and list prices.
        cost = (
            (prompt_tokens or prompt_tokens_est) * spec.price_in
            + completion_tokens * spec.price_out
        ) / 1_000_000
    details = usage.get("completion_tokens_details") or {}
    return LlmResult(
        text=_message_text(message).strip(),
        model=data.get("model") or spec.model,
        cost=float(cost),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        reasoning_tokens=int(details.get("reasoning_tokens") or 0),
        finish_reason=choice.get("finish_reason"),
        citations=_citations(message),
    )


class OpenRouterClient:
    def __init__(
        self,
        api_key: str,
        *,
        max_concurrency: int = 8,
        max_attempts: int = 3,
        app_title: str = "FutureEval forecasting bot",
        on_cost: Callable[[float], None] | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is missing")
        self._http = httpx.AsyncClient(
            base_url=OPENROUTER_BASE_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "X-Title": app_title,
            },
            timeout=httpx.Timeout(300.0, connect=20.0),
            transport=transport,
        )
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._max_attempts = max_attempts
        self._on_cost = on_cost
        self.total_cost = 0.0
        self.calls = 0
        self.failures = 0

    async def aclose(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> "OpenRouterClient":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    async def key_info(self) -> dict | None:
        try:
            resp = await self._http.get("/key", timeout=20.0)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:  # noqa: BLE001 - informational only
            logger.warning("Could not read OpenRouter key info: %s", type(exc).__name__)
            return None

    def _record_cost(self, cost: float) -> None:
        self.total_cost += cost
        if self._on_cost and cost > 0:
            try:
                self._on_cost(cost)
            except Exception:  # noqa: BLE001 - cost reporting must never break a call
                logger.debug("Cost callback failed", exc_info=True)

    async def complete(
        self,
        spec: ModelSpec,
        prompt: str,
        *,
        system: str | None = None,
        budget: Budget | None = None,
        tools: list[dict] | None = None,
        plugins: list[dict] | None = None,
        max_searches: int = 0,
        json_mode: bool = False,
        deadline: float | None = None,
        reasoning_effort: str | None = "__spec__",
        max_tokens: int | None = None,
    ) -> LlmResult:
        """One chat completion. ``deadline`` is a time.monotonic() value."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        effort = spec.reasoning_effort if reasoning_effort == "__spec__" else reasoning_effort
        payload: dict[str, Any] = {
            "model": spec.model,
            "messages": messages,
            "max_tokens": max_tokens or spec.max_tokens,
        }
        if effort:
            payload["reasoning"] = {"effort": effort, "exclude": True}
        if tools:
            payload["tools"] = tools
        if plugins:
            payload["plugins"] = plugins
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        prompt_tokens_est = estimate_tokens(system, prompt)
        # Web results are injected into the prompt: budget ~4k tokens per search.
        worst = spec.worst_case_cost(prompt_tokens_est + 4_000 * max_searches, max_searches)
        if budget is not None and not budget.try_reserve(worst):
            raise BudgetExceeded(
                f"Cannot afford {spec.model} (worst case ${worst:.3f}; "
                f"remaining ${budget.remaining if budget.remaining is not None else float('inf'):.3f})"
            )

        actual = 0.0
        last_error: Exception | None = None
        try:
            for attempt in range(1, self._max_attempts + 1):
                timeout = spec.timeout_s
                if deadline is not None:
                    timeout = min(timeout, deadline - time.monotonic())
                if timeout < 15:
                    raise LlmError("No time left before the question deadline")
                started = time.monotonic()
                try:
                    async with self._semaphore:
                        self.calls += 1
                        resp = await self._http.post(
                            "/chat/completions", json=payload, timeout=timeout
                        )
                    if resp.status_code >= 400:
                        text = resp.text[:300]
                        raise LlmError(
                            f"HTTP {resp.status_code} from OpenRouter for {spec.model}: {text}",
                            status=resp.status_code,
                            retryable=resp.status_code in RETRYABLE_STATUS,
                        )
                    result = parse_completion(resp.json(), spec, prompt_tokens_est)
                    actual += result.cost
                    self._record_cost(result.cost)
                    result.elapsed_s = time.monotonic() - started
                    if not result.text:
                        raise LlmError(
                            f"Empty answer from {spec.model} (finish_reason={result.finish_reason})",
                            retryable=result.finish_reason != "length",
                        )
                    return result
                except (httpx.TimeoutException, httpx.TransportError) as exc:
                    last_error = LlmError(f"{type(exc).__name__} calling {spec.model}", retryable=True)
                except LlmError as exc:
                    last_error = exc
                    if not exc.retryable:
                        break
                except ValueError as exc:  # invalid JSON body
                    last_error = LlmError(f"Invalid JSON from OpenRouter: {exc}", retryable=True)
                if attempt < self._max_attempts:
                    wait = min(30.0, 3.0 * 2 ** (attempt - 1)) + random.uniform(0, 2)
                    if deadline is not None and time.monotonic() + wait + 20 > deadline:
                        break
                    logger.info("Retrying %s after %s (attempt %d)", spec.model, type(last_error).__name__, attempt)
                    await asyncio.sleep(wait)
            self.failures += 1
            raise last_error or LlmError(f"Unknown failure calling {spec.model}")
        finally:
            if budget is not None:
                budget.settle(worst, actual)
