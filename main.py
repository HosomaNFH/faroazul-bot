"""Entry point of the FutureEval forecasting bot.

Modes
  tournament  seasonal FutureEval tournament + MiniBench (default; scheduled)
  seasonal    only the seasonal tournament
  minibench   only MiniBench
  test        a few questions of the bot-testing-area (one per type first)
  urls        specific questions given with --urls (dry run unless --publish)
  coverage    share of recently closed tournament questions the bot forecast
  budget      OpenRouter credit and the profile each tournament would use
  models      ping every configured model (a few cents)

Examples
  python main.py --mode test --dry-run --max-questions 4
  python main.py --mode urls --urls https://www.metaculus.com/questions/578/ --dry-run
  python main.py --mode tournament
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys

from dotenv import load_dotenv

from metabot.config import PROFILES, Settings


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    noisy = ("httpx", "httpcore", "LiteLLM", "litellm", "asknews_sdk", "urllib3")
    for name in noisy:
        logging.getLogger(name).setLevel(logging.WARNING)
    # forecasting-tools logs are fine locally; in public CI logs keep them to
    # warnings so that nothing about open-question forecasts leaks.
    logging.getLogger("forecasting_tools").setLevel(logging.INFO if verbose else logging.WARNING)


def check_environment(mode: str) -> list[str]:
    problems = []
    needs_metaculus = mode not in {"budget", "models"}
    if needs_metaculus and not os.getenv("METACULUS_TOKEN"):
        problems.append("METACULUS_TOKEN is missing")
    if mode != "coverage" and not os.getenv("OPENROUTER_API_KEY"):
        problems.append("OPENROUTER_API_KEY is missing")
    return problems


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FutureEval ensemble forecasting bot")
    parser.add_argument(
        "--mode",
        default="tournament",
        choices=["tournament", "seasonal", "minibench", "test", "urls", "coverage", "budget", "models"],
    )
    parser.add_argument("--dry-run", action="store_true", help="never publish forecasts")
    parser.add_argument("--publish", action="store_true", help="allow publishing in urls mode")
    parser.add_argument("--urls", nargs="*", default=[], help="question URLs for --mode urls")
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--profile", choices=sorted(PROFILES), default=None)
    parser.add_argument("--save-reports", default=None, help="folder for JSON reports (local use)")
    parser.add_argument("--days", type=int, default=7, help="window for --mode coverage")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


async def run(args: argparse.Namespace, settings: Settings) -> int:
    from metabot import runner
    from metabot.research import AskNewsClient

    AskNewsClient._lock = None  # fresh lock for this event loop

    if args.mode == "coverage":
        return await runner.coverage_report(settings, args.days)
    if args.mode == "budget":
        return await runner.budget_report(settings)
    if args.mode == "models":
        return await runner.models_check(settings)

    skip_previous = True
    dry_run = args.dry_run
    max_questions = args.max_questions
    diverse = False
    if args.mode == "tournament":
        targets = [
            ("seasonal", settings.seasonal_tournament, None),
            ("minibench", settings.minibench_tournament, None),
        ]
    elif args.mode == "seasonal":
        targets = [("seasonal", settings.seasonal_tournament, None)]
    elif args.mode == "minibench":
        targets = [("minibench", settings.minibench_tournament, None)]
    elif args.mode == "test":
        targets = [("test", settings.test_tournament, None)]
        skip_previous = False
        diverse = True
        max_questions = max_questions or 4
    else:  # urls
        if not args.urls:
            print("--mode urls needs --urls")
            return 2
        questions = await runner.fetch_urls(args.urls)
        targets = [("other", "urls", questions)]
        skip_previous = False
        dry_run = not args.publish or args.dry_run

    results, total_cost, credit = await runner.forecast_targets(
        targets,
        settings,
        dry_run=dry_run,
        profile_override=args.profile,
        max_questions=max_questions,
        diverse=diverse,
        save_folder=args.save_reports,
        skip_previously_forecasted=skip_previous,
    )
    return runner.print_summary(args.mode, not dry_run, results, total_cost, credit)


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = parse_args(argv)
    configure_logging(args.verbose)
    settings = Settings.from_env()
    problems = check_environment(args.mode)
    if problems:
        if args.mode in {"tournament", "seasonal", "minibench"}:
            # Deployed before every key exists (e.g. credits not granted yet):
            # keep the scheduled workflow green but visible in the run summary.
            for problem in problems:
                print(f"::warning::{problem}; skipping this scheduled run")
            return 0
        print("Setup problems:\n  - " + "\n  - ".join(problems))
        return 1
    mode_publish = "no (dry run)" if args.dry_run or (args.mode == "urls" and not args.publish) else "yes"
    print(f"Bot starting: mode={args.mode}, publish={mode_publish}")
    return asyncio.run(run(args, settings))


if __name__ == "__main__":
    sys.exit(main())
