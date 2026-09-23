"""Parsers: the formats models actually produce."""

import unittest
from datetime import datetime, timezone

from metabot.parsing import (
    parse_binary,
    parse_multiple_choice,
    parse_percentiles,
    to_float,
)


class NumberTests(unittest.TestCase):
    def test_to_float(self):
        self.assertEqual(to_float("1,234,567"), 1234567.0)
        self.assertEqual(to_float("-3.5e6"), -3.5e6)
        self.assertEqual(to_float("12%"), 12.0)
        self.assertEqual(to_float("−4"), -4.0)
        self.assertIsNone(to_float("1.2 million"))
        self.assertIsNone(to_float("abc"))
        self.assertIsNone(to_float(True))


class BinaryTests(unittest.TestCase):
    def test_final_line(self):
        self.assertAlmostEqual(parse_binary("Reasoning...\nProbability: 23%"), 0.23)

    def test_markdown_and_decimals(self):
        self.assertAlmostEqual(parse_binary("**Probability:** 7.5 %"), 0.075)

    def test_prefers_final_line_over_earlier_mentions(self):
        text = (
            "The base rate probability of 40% seems high.\n"
            "Probability: 12%\n"
            "Note: a probability of 5% would be too low."
        )
        self.assertAlmostEqual(parse_binary(text), 0.12)

    def test_json(self):
        self.assertAlmostEqual(parse_binary('```json\n{"probability": 0.3}\n```'), 0.3)
        self.assertAlmostEqual(parse_binary('{"probability": 35}'), 0.35)

    def test_missing(self):
        self.assertIsNone(parse_binary("I cannot say."))


class MultipleChoiceTests(unittest.TestCase):
    options = ["Donald Trump", "Kamala Harris", "Other (anyone else)"]

    def test_json_exact_names(self):
        text = 'blah\n```json\n{"probabilities": {"Donald Trump": 55, "Kamala Harris": 40, "Other (anyone else)": 5}}\n```'
        out = parse_multiple_choice(text, self.options)
        self.assertAlmostEqual(out["Donald Trump"], 0.55)
        self.assertAlmostEqual(sum(out.values()), 1.0)

    def test_json_fuzzy_names(self):
        text = '{"probabilities": {"donald trump": 0.5, "Option 2": 0.3, "other": 0.2}}'
        out = parse_multiple_choice(text, self.options)
        self.assertAlmostEqual(out["Kamala Harris"], 0.3)
        self.assertAlmostEqual(out["Other (anyone else)"], 0.2)

    def test_lines(self):
        text = "Donald Trump: 60%\nKamala Harris: 35%\nOther (anyone else): 5%"
        out = parse_multiple_choice(text, self.options)
        self.assertAlmostEqual(out["Donald Trump"], 0.60)

    def test_missing_option_filled_when_sum_complete(self):
        text = '{"probabilities": {"Donald Trump": 70, "Kamala Harris": 30}}'
        out = parse_multiple_choice(text, self.options)
        self.assertEqual(out["Other (anyone else)"], 0.0)

    def test_bad_sum_rejected(self):
        text = '{"probabilities": {"Donald Trump": 70, "Kamala Harris": 70, "Other (anyone else)": 70}}'
        self.assertIsNone(parse_multiple_choice(text, self.options))


class PercentileTests(unittest.TestCase):
    keys = ["1", "2.5", "5", "10", "20", "40", "50", "60", "80", "90", "95", "97.5", "99"]

    def test_json(self):
        values = [1, 2, 3, 5, 8, 12, 14, 16, 20, 25, 30, 34, 40]
        body = ", ".join(f'"{k}": {v}' for k, v in zip(self.keys, values))
        out = parse_percentiles(f'```json\n{{"percentiles": {{{body}}}}}\n```')
        self.assertEqual(len(out), 13)
        self.assertEqual(out[0], (0.01, 1.0))
        self.assertEqual(out[-1], (0.99, 40.0))
        self.assertEqual(dict(out)[0.025], 2.0)

    def test_lines_with_commas_and_scientific(self):
        text = "\n".join(
            [
                "Percentile 10: 1,000",
                "Percentile 20: 2,000",
                "Percentile 40: 3,500",
                "Percentile 60: 4,500",
                "Percentile 80: 6e3",
                "Percentile 90: 8,000",
            ]
        )
        out = parse_percentiles(text)
        self.assertEqual(dict(out)[0.8], 6000.0)

    def test_ordinal_lines(self):
        text = "10th percentile: 5\n20th percentile: 6\n50th percentile: 8\n80th percentile: 10\n90th percentile: 12"
        out = parse_percentiles(text)
        self.assertEqual(dict(out)[0.5], 8.0)

    def test_dates(self):
        dates = ["2026-11-01", "2026-11-15", "2026-12-01", "2027-01-01", "2027-02-01", "2027-03-01",
                 "2027-03-15", "2027-04-01", "2027-05-01", "2027-06-01", "2027-07-01", "2027-08-01", "2027-09-01"]
        body = ", ".join(f'"{k}": "{v}"' for k, v in zip(self.keys, dates))
        out = parse_percentiles(f'{{"percentiles": {{{body}}}}}', is_date=True)
        expected = datetime(2027, 3, 15, tzinfo=timezone.utc).timestamp()
        self.assertEqual(dict(out)[0.5], expected)

    def test_fraction_keys(self):
        text = '{"percentiles": {"0.1": 1, "0.25": 2, "0.5": 3, "0.75": 4, "0.9": 5}}'
        out = parse_percentiles(text)
        self.assertEqual(dict(out)[0.5], 3.0)

    def test_insufficient(self):
        self.assertIsNone(parse_percentiles("Percentile 50: 10"))


if __name__ == "__main__":
    unittest.main()
