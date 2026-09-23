"""Multi-source research with graceful degradation.

Sources run concurrently:
* AskNews (free for tournament bots): latest news (48 h) and, in the richer
  profiles, the news archive. Calls are globally spaced to respect the free
  tier rate limit (1 call per 10 s).
* One or two web-search researchers through OpenRouter's
  ``openrouter:web_search`` server tool (the model may search several times:
  iterative search beat one-shot retrieval in published analyses).

Fallbacks: if every web researcher fails we retry once with the older
``plugins: web`` interface; if there is still nothing we ask a model for a
knowledge-only brief, clearly labelled as such. Research never raises: the
forecasters always get *something* plus a note of what failed.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from .budget import Budget
from .config import RESEARCH_FLASH, RESEARCH_SOL, ModelSpec, Profile, Settings
from .llm import LlmError, OpenRouterClient, web_search_plugin, web_search_tool
from .prompts import SYSTEM_RESEARCHER, knowledge_prompt, research_prompt
from .qctx import QCtx

logger = logging.getLogger(__name__)


@dataclass
class SourceResult:
    name: str
    ok: bool
    text: str = ""
    error: str | None = None
    cost: float = 0.0


def _truncate(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit].rsplit("\n", 1)[0] + "\n[... truncated ...]"


# --------------------------------------------------------------------------- #
# AskNews
# --------------------------------------------------------------------------- #


class AskNewsClient:
    """Thin wrapper over the official asknews SDK with a global rate limit."""

    _lock: asyncio.Lock | None = None
    _last_call: float = 0.0

    def __init__(self, min_interval_s: float = 11.0, n_articles: int = 8) -> None:
        self.client_id = os.getenv("ASKNEWS_CLIENT_ID") or None
        self.client_secret = os.getenv("ASKNEWS_SECRET") or None
        self.api_key = os.getenv("ASKNEWS_API_KEY") or None
        if self.client_id and self.client_secret:
            self.api_key = None  # prefer OAuth, like forecasting-tools
        self.min_interval_s = min_interval_s
        self.n_articles = n_articles

    @staticmethod
    def configured() -> bool:
        has_oauth = bool(os.getenv("ASKNEWS_CLIENT_ID") and os.getenv("ASKNEWS_SECRET"))
        return has_oauth or bool(os.getenv("ASKNEWS_API_KEY"))

    @classmethod
    def _get_lock(cls) -> asyncio.Lock:
        # One lock per event loop run (main.py uses a single asyncio.run).
        if cls._lock is None:
            cls._lock = asyncio.Lock()
        return cls._lock

    async def _wait_turn(self) -> None:
        async with self._get_lock():
            wait = AskNewsClient._last_call + self.min_interval_s - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
            AskNewsClient._last_call = time.monotonic()

    @staticmethod
    def _format_articles(articles: list[Any]) -> str:
        lines = []
        for art in articles:
            get = art.get if isinstance(art, dict) else (lambda k, a=art: getattr(a, k, None))
            title = get("eng_title") or get("title") or "(untitled)"
            summary = get("summary") or ""
            date = get("pub_date") or ""
            source = get("source_id") or ""
            url = get("article_url") or get("url") or ""
            lines.append(f"- **{title}** ({source}, {date})\n  {summary}\n  {url}")
        return "\n".join(lines)

    async def search(self, query: str, strategies: tuple[str, ...]) -> str:
        from asknews_sdk import AsyncAskNewsSDK  # imported lazily

        chunks = []
        async with AsyncAskNewsSDK(
            client_id=self.client_id,
            client_secret=self.client_secret,
            api_key=self.api_key,
            scopes={"news"},
        ) as ask:
            for strategy in strategies:
                await self._wait_turn()
                response = await ask.news.search_news(
                    query=query,
                    n_articles=self.n_articles,
                    return_type="both",
                    strategy=strategy,
                )
                text = getattr(response, "as_string", None)
                if not text:
                    text = self._format_articles(list(getattr(response, "as_dicts", None) or []))
                label = "Latest news (last 48 hours)" if strategy == "latest news" else "News archive"
                chunks.append(f"### {label}\n{text.strip() or 'No articles found.'}")
        return "\n\n".join(chunks)


# --------------------------------------------------------------------------- #
# Web research through OpenRouter
# --------------------------------------------------------------------------- #


def _with_citations(text: str, citations: list[dict], limit: int = 12) -> str:
    if not citations:
        return text
    seen = []
    for c in citations:
        if c["url"] not in [s["url"] for s in seen]:
            seen.append(c)
    refs = "\n".join(f"- {c['title'] or c['url']}: {c['url']}" for c in seen[:limit])
    return f"{text}\n\nSources cited:\n{refs}"


async def web_research(
    ctx: QCtx,
    spec: ModelSpec,
    client: OpenRouterClient,
    budget: Budget,
    deadline: float,
    now: datetime,
    max_searches: int,
    use_plugin: bool = False,
) -> SourceResult:
    name = f"Web search ({spec.model})"
    try:
        kwargs: dict[str, Any] = {"max_searches": max_searches}
        if use_plugin:
            kwargs["plugins"] = web_search_plugin()
        else:
            kwargs["tools"] = web_search_tool(max_uses=max_searches)
        result = await client.complete(
            spec,
            research_prompt(ctx, now),
            system=SYSTEM_RESEARCHER,
            budget=budget,
            deadline=deadline,
            **kwargs,
        )
        return SourceResult(name, True, _with_citations(result.text, result.citations), cost=result.cost)
    except Exception as exc:  # noqa: BLE001 - research must degrade, not fail
        return SourceResult(name, False, error=f"{type(exc).__name__}: {str(exc)[:200]}")


async def knowledge_brief(
    ctx: QCtx, spec: ModelSpec, client: OpenRouterClient, budget: Budget, deadline: float, now: datetime
) -> SourceResult:
    name = f"Model knowledge only, no live search ({spec.model})"
    try:
        result = await client.complete(
            spec, knowledge_prompt(ctx, now), system=SYSTEM_RESEARCHER, budget=budget, deadline=deadline
        )
        return SourceResult(name, True, result.text, cost=result.cost)
    except Exception as exc:  # noqa: BLE001
        return SourceResult(name, False, error=f"{type(exc).__name__}: {str(exc)[:200]}")


async def asknews_research(ctx: QCtx, strategies: tuple[str, ...], settings: Settings) -> SourceResult:
    name = "AskNews"
    if not strategies:
        return SourceResult(name, False, error="disabled in this profile")
    if not AskNewsClient.configured():
        return SourceResult(name, False, error="no AskNews credentials")
    try:
        query = ctx.title.strip()
        if ctx.group_option:
            query = f"{query} ({ctx.group_option})"
        text = await AskNewsClient(settings.asknews_min_interval_s, settings.asknews_articles).search(
            query[:350], strategies
        )
        return SourceResult(name, True, text)
    except Exception as exc:  # noqa: BLE001
        return SourceResult(name, False, error=f"{type(exc).__name__}: {str(exc)[:200]}")


async def gather_research(
    ctx: QCtx,
    profile: Profile,
    client: OpenRouterClient,
    budget: Budget,
    deadline: float,
    now: datetime,
    settings: Settings,
    fallback_spec: ModelSpec,
) -> tuple[str, list[SourceResult]]:
    # AskNews is used only when its secrets exist (it is optional).
    use_asknews = bool(profile.asknews_strategies) and AskNewsClient.configured()
    web_specs = list(profile.web_researchers)
    if not use_asknews:
        # Without AskNews keep at least two independent web sources.
        for extra in (RESEARCH_SOL, RESEARCH_FLASH):
            if len(web_specs) >= 2:
                break
            if extra.model not in {s.model for s in web_specs}:
                web_specs.append(extra)

    tasks = [asknews_research(ctx, profile.asknews_strategies, settings)] if use_asknews else []
    tasks += [
        web_research(ctx, spec, client, budget, deadline, now, profile.web_max_searches)
        for spec in web_specs
    ]
    results: list[SourceResult] = list(await asyncio.gather(*tasks))
    web_results = results[1:] if use_asknews else results

    if web_specs and not any(r.ok for r in web_results):
        logger.warning("All web researchers failed; retrying with the web plugin")
        results.append(
            await web_research(
                ctx, web_specs[0], client, budget, deadline, now,
                profile.web_max_searches, use_plugin=True,
            )
        )
    if not any(r.ok for r in results):
        # Last resort; if even this fails the forecasters work without research.
        results.append(await knowledge_brief(ctx, fallback_spec, client, budget, deadline, now))

    sections = []
    for index, res in enumerate([r for r in results if r.ok], start=1):
        body = _truncate(res.text, settings.research_chars_per_source)
        sections.append(f"## Research source {index}: {res.name}\n{body}")
    failed = [f"{r.name}: {r.error}" for r in results if not r.ok and r.error != "disabled in this profile"]
    if failed:
        sections.append("## Research notes\nUnavailable sources: " + "; ".join(failed))
    return "\n\n".join(sections), results
