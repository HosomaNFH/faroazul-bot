"""CDF construction: validity for every bound/scale/type combination."""

import unittest
from datetime import datetime, timezone
from statistics import NormalDist

import numpy as np

from metabot.cdf import (
    STANDARD_PERCENTILES,
    DistributionError,
    Scale,
    build_cdf,
    cdf_quantile,
    pchip_eval,
    pchip_slopes,
    plausibility_issue,
    standardize_cdf,
    validate_cdf,
)


def normal_declared(mu, sigma, percentiles=STANDARD_PERCENTILES):
    dist = NormalDist(mu, sigma)
    return [(p, dist.inv_cdf(p)) for p in percentiles]


class ScaleTests(unittest.TestCase):
    def test_linear_round_trip(self):
        s = Scale(0, 100, False, False)
        xs = np.array([-20.0, 0.0, 12.5, 100.0, 180.0])
        np.testing.assert_allclose(s.from_loc(s.to_loc(xs)), xs, rtol=1e-12, atol=1e-9)
        self.assertAlmostEqual(float(s.to_loc(50)), 0.5)

    def test_log_round_trip(self):
        s = Scale(1, 1000, True, True, zero_point=0.0)
        xs = np.array([1.0, 10.0, 100.0, 1000.0, 5000.0])
        np.testing.assert_allclose(s.from_loc(s.to_loc(xs)), xs, rtol=1e-9)
        self.assertAlmostEqual(float(s.to_loc(1)), 0.0)
        self.assertAlmostEqual(float(s.to_loc(1000)), 1.0)

    def test_invalid_zero_point_falls_back_to_linear(self):
        s = Scale(10, 20, False, False, zero_point=10)
        self.assertIsNone(s.zero_point)

    def test_step_limits(self):
        s = Scale(0, 1, False, False, cdf_size=201)
        self.assertAlmostEqual(s.min_step, 5e-5)
        self.assertAlmostEqual(s.max_step, 0.2)
        d = Scale(0, 1, False, False, cdf_size=12)
        self.assertEqual(d.max_step, 1.0)


class BuildCdfTests(unittest.TestCase):
    def check(self, cdf, scale):
        validate_cdf(cdf, scale)
        self.assertTrue(np.all(np.diff(cdf) > 0))

    def test_open_bounds_normal(self):
        s = Scale(0, 100, True, True)
        cdf = build_cdf(normal_declared(50, 10), s)
        self.check(cdf, s)
        self.assertAlmostEqual(cdf_quantile(cdf, s, 0.5), 50, delta=1.5)
        self.assertGreaterEqual(cdf[0], 0.001)
        self.assertLessEqual(cdf[-1], 0.999)

    def test_closed_bounds(self):
        s = Scale(0, 100, False, False)
        cdf = build_cdf(normal_declared(30, 8), s)
        self.check(cdf, s)
        self.assertEqual(cdf[0], 0.0)
        self.assertAlmostEqual(cdf[-1], 1.0, places=9)

    def test_mixed_bounds(self):
        for open_lower, open_upper in ((True, False), (False, True)):
            s = Scale(0, 100, open_lower, open_upper)
            cdf = build_cdf(normal_declared(60, 15), s)
            self.check(cdf, s)

    def test_thin_tail_beyond_open_upper(self):
        s = Scale(0, 100, True, True)
        cdf = build_cdf(normal_declared(40, 5), s)  # p99 ~ 51.6
        self.assertLess(1 - cdf[-1], 0.02)

    def test_mass_beyond_open_upper_is_kept(self):
        s = Scale(0, 100, False, True)
        # The forecaster thinks ~20% is above the upper bound.
        declared = normal_declared(85, 20)
        cdf = build_cdf(declared, s)
        self.check(cdf, s)
        expected_above = 1 - NormalDist(85, 20).cdf(100)
        self.assertAlmostEqual(1 - cdf[-1], expected_above, delta=0.04)

    def test_values_below_closed_lower_bound_are_clamped(self):
        s = Scale(0, 50, False, False)
        declared = [(p, v) for p, v in normal_declared(2, 5)]  # many values < 0
        cdf = build_cdf(declared, s)
        self.check(cdf, s)

    def test_log_scaled_question(self):
        s = Scale(1, 10_000, True, True, zero_point=0.0)
        declared = [(p, float(np.exp(NormalDist(np.log(300), 1.0).inv_cdf(p)))) for p in STANDARD_PERCENTILES]
        cdf = build_cdf(declared, s)
        self.check(cdf, s)
        self.assertAlmostEqual(cdf_quantile(cdf, s, 0.5), 300, delta=60)

    def test_discrete_question_with_ties(self):
        # Outcomes 0..10, Metaculus-style half-integer bounds, 11 buckets.
        s = Scale(-0.5, 10.5, False, False, cdf_size=12, discrete=True)
        declared = [(0.01, 0), (0.025, 0), (0.05, 1), (0.1, 1), (0.2, 2), (0.4, 3),
                    (0.5, 3), (0.6, 3), (0.8, 4), (0.9, 5), (0.95, 6), (0.975, 7), (0.99, 8)]
        cdf = build_cdf(declared, s)
        self.check(cdf, s)
        pmf = np.diff(cdf)
        self.assertEqual(int(np.argmax(pmf)), 3)  # bucket for outcome 3
        self.assertGreater(pmf[3], 0.25)

    def test_discrete_all_same_value(self):
        s = Scale(-0.5, 10.5, False, False, cdf_size=12, discrete=True)
        declared = [(p, 4) for p in STANDARD_PERCENTILES]
        cdf = build_cdf(declared, s)
        self.check(cdf, s)
        self.assertGreater(np.diff(cdf)[4], 0.8)

    def test_date_like_timestamps(self):
        lo = datetime(2026, 10, 1, tzinfo=timezone.utc).timestamp()
        hi = datetime(2027, 12, 31, tzinfo=timezone.utc).timestamp()
        s = Scale(lo, hi, False, True)
        mid = datetime(2027, 3, 1, tzinfo=timezone.utc).timestamp()
        declared = normal_declared(mid, 60 * 86400)
        cdf = build_cdf(declared, s)
        self.check(cdf, s)

    def test_out_of_order_values_are_sorted(self):
        s = Scale(0, 100, True, True)
        declared = normal_declared(50, 10)
        declared[5], declared[6] = (declared[5][0], declared[6][1]), (declared[6][0], declared[5][1])
        cdf = build_cdf(declared, s)
        self.check(cdf, s)

    def test_identical_values_rejected_for_continuous(self):
        s = Scale(0, 100, True, True)
        with self.assertRaises(DistributionError):
            build_cdf([(p, 42.0) for p in STANDARD_PERCENTILES], s)

    def test_few_percentiles_like_the_template(self):
        s = Scale(0, 100, True, True)
        declared = normal_declared(50, 12, (0.1, 0.2, 0.4, 0.6, 0.8, 0.9))
        cdf = build_cdf(declared, s)
        self.check(cdf, s)

    def test_narrow_distribution_respects_max_step(self):
        s = Scale(0, 1000, True, True)
        cdf = build_cdf(normal_declared(500, 0.5), s)  # far narrower than one bin
        self.check(cdf, s)


class PchipTests(unittest.TestCase):
    def test_monotone_and_interpolating(self):
        rng = np.random.default_rng(0)
        for _ in range(50):
            x = np.sort(rng.uniform(0, 1, 8))
            x = np.unique(x)
            y = np.sort(rng.uniform(0, 1, len(x)))
            y = y + np.arange(len(x)) * 1e-6
            d = pchip_slopes(x, y)
            xq = np.linspace(x[0], x[-1], 400)
            yq = pchip_eval(x, y, d, xq)
            self.assertTrue(np.all(np.diff(yq) >= -1e-12))
            np.testing.assert_allclose(pchip_eval(x, y, d, x), y, atol=1e-12)


class StandardizeTests(unittest.TestCase):
    def test_flat_raw_cdf_gets_min_step_with_margin(self):
        for open_lower in (True, False):
            for open_upper in (True, False):
                raw = np.concatenate([np.zeros(100), np.ones(101)])
                cdf = standardize_cdf(raw, open_lower, open_upper)
                s = Scale(0, 1, open_lower, open_upper)
                validate_cdf(cdf, s)
                self.assertGreater(np.min(np.diff(cdf)), 5e-5 * 1.01)
                self.assertLessEqual(np.max(np.diff(cdf)), 0.2)


class PlausibilityTests(unittest.TestCase):
    def test_units_error_detected(self):
        s = Scale(0, 100, True, True)
        declared = [(p, v * 1_000_000) for p, v in normal_declared(50, 10)]
        self.assertIsNotNone(plausibility_issue(declared, s))

    def test_normal_case_ok(self):
        s = Scale(0, 100, True, True)
        self.assertIsNone(plausibility_issue(normal_declared(50, 10), s))


if __name__ == "__main__":
    unittest.main()
