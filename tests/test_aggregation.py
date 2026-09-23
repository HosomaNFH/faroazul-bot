"""Aggregation: robustness, clipping, calibration, validity of pooled CDFs."""

import unittest
from statistics import NormalDist

import numpy as np

from metabot.aggregation import (
    aggregate_binary,
    aggregate_cdfs,
    aggregate_multiple_choice,
    normalize_with_floor,
    trim_count,
    trimmed_mean,
)
from metabot.cdf import STANDARD_PERCENTILES, Scale, build_cdf, validate_cdf


class TrimTests(unittest.TestCase):
    def test_trim_count(self):
        self.assertEqual(trim_count(1, 0.2), 0)
        self.assertEqual(trim_count(2, 0.2), 0)
        self.assertEqual(trim_count(3, 0.2), 1)  # median
        self.assertEqual(trim_count(4, 0.2), 1)
        self.assertEqual(trim_count(9, 0.2), 1)
        self.assertEqual(trim_count(10, 0.2), 2)

    def test_trimmed_mean_ignores_outliers(self):
        self.assertAlmostEqual(trimmed_mean([1, 2, 3, 100], 0.2), 2.5)


class BinaryTests(unittest.TestCase):
    def test_robust_to_one_outlier(self):
        p = aggregate_binary([0.30, 0.32, 0.35, 0.31, 0.99])
        self.assertGreater(p, 0.30)
        self.assertLess(p, 0.36)

    def test_clipping(self):
        self.assertEqual(aggregate_binary([0.999] * 5), 0.98)
        self.assertEqual(aggregate_binary([0.0001] * 5), 0.02)

    def test_symmetry(self):
        a = aggregate_binary([0.2, 0.3, 0.4])
        b = aggregate_binary([0.8, 0.7, 0.6])
        self.assertAlmostEqual(a, 1 - b, places=12)

    def test_platt_identity_and_extremizing(self):
        base = aggregate_binary([0.7, 0.72, 0.75])
        self.assertAlmostEqual(base, aggregate_binary([0.7, 0.72, 0.75], platt_a=1.0, platt_b=0.0))
        self.assertGreater(aggregate_binary([0.7, 0.72, 0.75], platt_a=1.5), base)

    def test_geometric_mean_of_odds_for_two(self):
        # Two members: plain mean of log-odds (odds 1/9 and 1) -> odds 1/3 -> p=0.25.
        self.assertAlmostEqual(aggregate_binary([0.1, 0.5], clip_low=0.001), 0.25, places=9)


class MultipleChoiceTests(unittest.TestCase):
    def test_floor_and_sum(self):
        probs = normalize_with_floor({"a": 0.995, "b": 0.005, "c": 0.0}, 0.01)
        self.assertAlmostEqual(sum(probs.values()), 1.0)
        self.assertGreaterEqual(min(probs.values()), 0.01 - 1e-12)

    def test_floor_too_high_gives_uniform(self):
        probs = normalize_with_floor({o: 1.0 for o in "abcde"}, 0.3)
        self.assertAlmostEqual(probs["a"], 0.2)

    def test_pool(self):
        options = ["A", "B", "C"]
        members = [
            {"A": 0.6, "B": 0.3, "C": 0.1},
            {"A": 0.5, "B": 0.4, "C": 0.1},
            {"A": 0.7, "B": 0.2, "C": 0.1},
            {"A": 0.05, "B": 0.05, "C": 0.9},  # outlier
        ]
        pooled = aggregate_multiple_choice(members, options)
        self.assertEqual(list(pooled), options)
        self.assertAlmostEqual(sum(pooled.values()), 1.0)
        self.assertGreater(pooled["A"], pooled["C"])


class CdfPoolTests(unittest.TestCase):
    def test_pooled_cdf_is_valid(self):
        for open_lower, open_upper in ((True, True), (False, False), (True, False), (False, True)):
            s = Scale(0, 100, open_lower, open_upper)
            cdfs = []
            for mu, sigma in ((40, 8), (45, 12), (55, 6), (90, 3), (48, 20)):
                dist = NormalDist(mu, sigma)
                declared = [(p, dist.inv_cdf(p)) for p in STANDARD_PERCENTILES]
                cdfs.append(build_cdf(declared, s))
            pooled = aggregate_cdfs(cdfs)
            validate_cdf(pooled, s)

    def test_length_mismatch(self):
        with self.assertRaises(ValueError):
            aggregate_cdfs([np.linspace(0, 1, 5), np.linspace(0, 1, 6)])


if __name__ == "__main__":
    unittest.main()
