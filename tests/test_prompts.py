"""Prompts contain what the forecaster needs, for every question type."""

import unittest
from datetime import datetime, timedelta, timezone

from metabot.prompts import (
    binary_prompt,
    fmt_num,
    multiple_choice_prompt,
    numeric_prompt,
    parser_prompt,
    research_prompt,
)
from metabot.qctx import QCtx

NOW = datetime(2026, 10, 5, 12, 0, tzinfo=timezone.utc)


def ctx(kind, **kw):
    base = dict(
        kind=kind,
        title="Will X happen?",
        criteria="Resolves YES if X happens before 2026-12-31 per source S.",
        fine_print="Announcements do not count.",
        background="Some background.",
        open_time=NOW - timedelta(hours=1),
        close_time=NOW + timedelta(minutes=80),
        resolve_time=NOW + timedelta(days=60),
    )
    base.update(kw)
    return QCtx(**base)


class PromptTests(unittest.TestCase):
    def test_binary(self):
        text = binary_prompt(ctx("binary"), "research here", NOW)
        for needle in ("2026-10-05", "Announcements do not count.", "Probability: XX%", "base rate",
                       "NOT been resolved", "research here", "60.0 days"):
            self.assertIn(needle, text)
        self.assertNotIn("Bayesian", text)

    def test_multiple_choice(self):
        c = ctx("multiple_choice", options=["Red", "Green", "Blue team"])
        text = multiple_choice_prompt(c, "", NOW)
        self.assertIn("3. Blue team", text)
        self.assertIn('"probabilities"', text)
        self.assertIn("no research available", text)

    def test_numeric_bounds_and_keys(self):
        c = ctx("numeric", lower=0, upper=500, open_lower=False, open_upper=True, unit="GW")
        text = numeric_prompt(c, "r", NOW)
        self.assertIn("lower end is CLOSED", text)
        self.assertIn("upper end is OPEN", text)
        for key in ("\"1\"", "\"2.5\"", "\"50\"", "\"97.5\"", "\"99\""):
            self.assertIn(key, text)
        self.assertIn("GW", text)

    def test_discrete_step(self):
        c = ctx("discrete", lower=-0.5, upper=10.5, nominal_lower=0, nominal_upper=10, cdf_size=12)
        self.assertIn("steps of 1", numeric_prompt(c, "r", NOW))

    def test_date(self):
        lo = datetime(2026, 10, 1, tzinfo=timezone.utc).timestamp()
        hi = datetime(2027, 12, 31, tzinfo=timezone.utc).timestamp()
        c = ctx("date", lower=lo, upper=hi, open_lower=False, open_upper=True)
        text = numeric_prompt(c, "r", NOW)
        self.assertIn("2027-12-31", text)
        self.assertIn("YYYY-MM-DD", text)

    def test_group_and_research_prompts(self):
        c = ctx("binary", group_option="Spain")
        self.assertIn('specifically about: "Spain"', binary_prompt(c, "", NOW))
        self.assertIn("Do NOT give a forecast", research_prompt(c, NOW))

    def test_parser_prompt(self):
        c = ctx("multiple_choice", options=["A", "B"])
        self.assertIn('"A", "B"', parser_prompt("multiple_choice", "text", c))

    def test_fmt_num(self):
        self.assertEqual(fmt_num(1234567.0), "1234567")
        self.assertEqual(fmt_num(0.5), "0.5")
        self.assertEqual(fmt_num(-0.0), "0")


if __name__ == "__main__":
    unittest.main()
