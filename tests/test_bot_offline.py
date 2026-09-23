"""End-to-end dry run of the bot with a fake OpenRouter (no keys, no network).

Builds one question of each type with forecasting-tools, answers every LLM
call from a mocked HTTP transport, and checks that the aggregated predictions
are valid Metaculus submissions. Skipped when forecasting-tools is missing.
"""

import asyncio
import json
import os
import re
import unittest
from datetime import datetime, timedelta, timezone

try:
    import httpx
    from forecasting_tools import (
        BinaryQuestion,
        DateQuestion,
        DiscreteQuestion,
        ForecastReport,
        MultipleChoiceQuestion,
        NumericQuestion,
        QuestionState,
    )

    from metabot.bot import EnsembleBot, report_cost_to_forecasting_tools
    from metabot.budget import Budget
    from metabot.cdf import Scale, validate_cdf
    from metabot.config import Settings
    from metabot.llm import OpenRouterClient
    from metabot.qctx import ctx_from_question

    AVAILABLE = True
except ImportError:
    AVAILABLE = False

NOW = datetime.now(timezone.utc)
KEYS = ["1", "2.5", "5", "10", "20", "40", "50", "60", "80", "90", "95", "97.5", "99"]


def fake_answer(prompt: str) -> str:
    if "Gather the most decision-relevant" in prompt:
        return "- Status: nothing decisive yet (Example News, 2026-09-20)."
    if "Probability: XX%" in prompt:
        return "Steps...\nProbability: 30%"
    if '"probabilities"' in prompt and "OPTIONS (use these names exactly)" in prompt:
        block = prompt.split("OPTIONS (use these names exactly)\n", 1)[1].split("\n\n", 1)[0]
        names = [re.sub(r"^\d+\. ", "", line) for line in block.splitlines() if line.strip()]
        share = round(100 / len(names), 4)
        body = ", ".join(f'"{n}": {share}' for n in names)
        return f'```json\n{{"probabilities": {{{body}}}}}\n```'
    if "YYYY-MM-DD" in prompt:
        dates = [(NOW + timedelta(days=20 + 15 * i)).strftime("%Y-%m-%d") for i in range(13)]
        body = ", ".join(f'"{k}": "{d}"' for k, d in zip(KEYS, dates))
        return f'```json\n{{"percentiles": {{{body}}}}}\n```'
    if "steps of 1" in prompt:  # discrete 0..10
        values = [0, 0, 1, 1, 2, 3, 3, 3, 4, 5, 6, 7, 8]
    else:  # numeric 0..100
        values = [5, 8, 12, 20, 30, 42, 48, 54, 66, 76, 84, 90, 97]
    body = ", ".join(f'"{k}": {v}' for k, v in zip(KEYS, values))
    return f'```json\n{{"percentiles": {{{body}}}}}\n```'


def handler(request):
    payload = json.loads(request.content)
    prompt = payload["messages"][-1]["content"]
    return httpx.Response(
        200,
        json={
            "model": payload["model"],
            "choices": [{"message": {"content": fake_answer(prompt)}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 500, "cost": 0.001},
        },
    )


def common(i):
    return dict(
        question_text=f"Test question {i}?",
        id_of_post=900 + i,
        id_of_question=9000 + i,
        page_url=f"https://www.metaculus.com/questions/{900 + i}/",
        resolution_criteria="Resolves according to the test source.",
        fine_print="",
        background_info="",
        close_time=NOW + timedelta(hours=1),
        open_time=NOW - timedelta(minutes=10),
        scheduled_resolution_time=NOW + timedelta(days=90),
        state=QuestionState.OPEN,
        already_forecasted=False,
    )


@unittest.skipUnless(AVAILABLE, "forecasting-tools / httpx not installed")
class OfflineBotTests(unittest.TestCase):
    def setUp(self):
        for var in ("ASKNEWS_CLIENT_ID", "ASKNEWS_SECRET", "ASKNEWS_API_KEY"):
            os.environ.pop(var, None)

    def test_every_question_type(self):
        questions = [
            BinaryQuestion(**common(1)),
            MultipleChoiceQuestion(**common(2), options=["Alpha", "Beta", "Gamma (other)"]),
            NumericQuestion(**common(3), lower_bound=0.0, upper_bound=100.0,
                            open_lower_bound=False, open_upper_bound=True, unit_of_measure="units"),
            DiscreteQuestion(**common(4), lower_bound=-0.5, upper_bound=10.5, open_lower_bound=False,
                             open_upper_bound=False, cdf_size=12, nominal_lower_bound=0.0,
                             nominal_upper_bound=10.0),
            DateQuestion(**common(5), lower_bound=NOW, upper_bound=NOW + timedelta(days=365),
                         open_lower_bound=False, open_upper_bound=True),
        ]

        async def go():
            client = OpenRouterClient("test", transport=httpx.MockTransport(handler),
                                      on_cost=report_cost_to_forecasting_tools)
            bot = EnsembleBot(settings=Settings(), client=client, run_budget=Budget(20.0),
                              dry_run=True, kind_profiles={"seasonal": "standard"})
            try:
                return await bot.forecast_questions(questions, return_exceptions=True)
            finally:
                await client.aclose()

        reports = asyncio.run(go())
        self.assertEqual(len(reports), 5)
        for question, report in zip(questions, reports):
            self.assertIsInstance(report, ForecastReport, msg=f"{type(question).__name__}: {report}")
            # forecasting-tools prefixes the markdown report with a newline.
            self.assertTrue(report.explanation.lstrip().startswith("#"))
            self.assertGreater(report.price_estimate or 0, 0)
        binary, mc, numeric, discrete, date = (r.prediction for r in reports)
        self.assertTrue(0.02 <= binary <= 0.98)
        self.assertAlmostEqual(sum(o.probability for o in mc.predicted_options), 1.0, places=6)
        for question, dist in ((questions[2], numeric), (questions[3], discrete), (questions[4], date)):
            scale = Scale.from_ctx(ctx_from_question(question))
            cdf = [p.percentile for p in dist.get_cdf()]
            self.assertEqual(len(cdf), question.cdf_size)
            validate_cdf(cdf, scale)


if __name__ == "__main__":
    unittest.main()
