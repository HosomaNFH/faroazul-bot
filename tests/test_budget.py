"""Budget reservations and adaptive profile choice."""

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from metabot.budget import Budget, CreditInfo, choose_profile, expected_remaining_questions
from metabot.config import PROFILES, Settings
from metabot.state import RunState

SEASON_START = datetime(2026, 9, 28, tzinfo=timezone.utc)
MID_SEASON = datetime(2026, 11, 17, tzinfo=timezone.utc)


class BudgetTests(unittest.TestCase):
    def test_reserve_and_settle(self):
        run = Budget(10.0, name="run")
        q = Budget(2.0, parent=run, name="q")
        self.assertTrue(q.try_reserve(1.5))
        self.assertFalse(q.try_reserve(1.0))  # question cap
        q.settle(1.5, 0.4)
        self.assertAlmostEqual(q.spent, 0.4)
        self.assertAlmostEqual(run.spent, 0.4)
        self.assertAlmostEqual(q.reserved, 0.0)
        self.assertTrue(q.try_reserve(1.0))

    def test_parent_cap_applies(self):
        run = Budget(1.0)
        a = Budget(5.0, parent=run)
        b = Budget(5.0, parent=run)
        self.assertTrue(a.try_reserve(0.8))
        self.assertFalse(b.try_reserve(0.5))

    def test_no_cap(self):
        self.assertTrue(Budget(None).try_reserve(1e9))

    def test_credit_info(self):
        info = CreditInfo.from_api({"data": {"limit": 100, "limit_remaining": 74.5, "usage": 25.5}})
        self.assertEqual(info.limit_remaining, 74.5)
        self.assertIsNone(CreditInfo.from_api(None))


class ProfileChoiceTests(unittest.TestCase):
    def setUp(self):
        self.settings = Settings()

    def test_unlimited_key(self):
        self.assertEqual(choose_profile("seasonal", None, SEASON_START, self.settings), "standard")

    def test_requested_credit_600(self):
        self.assertEqual(choose_profile("seasonal", 600, SEASON_START, self.settings), "standard")
        # A few test runs before the season must not push the season to economy.
        self.assertEqual(choose_profile("seasonal", 570, SEASON_START, self.settings), "standard")

    def test_generous_credit(self):
        self.assertEqual(choose_profile("seasonal", 3000, SEASON_START, self.settings), "premium")

    def test_tight_credit(self):
        self.assertEqual(choose_profile("seasonal", 150, SEASON_START, self.settings), "economy")

    def test_below_reserve_stops(self):
        self.assertEqual(choose_profile("seasonal", 3, SEASON_START, self.settings), "stop")

    def test_minibench_uses_its_profile(self):
        self.assertEqual(choose_profile("minibench", 600, SEASON_START, self.settings), "economy")

    def test_override_is_downgraded_when_unaffordable(self):
        self.settings.profile_override = "premium"
        self.assertEqual(choose_profile("seasonal", 100, SEASON_START, self.settings), "economy")

    def test_remaining_questions_decrease(self):
        early = expected_remaining_questions(SEASON_START, self.settings)
        mid = expected_remaining_questions(MID_SEASON, self.settings)
        self.assertGreater(early[0], mid[0])
        self.assertAlmostEqual(early[0], self.settings.expected_seasonal_questions)

    def test_profiles_are_consistent(self):
        for name, profile in PROFILES.items():
            self.assertGreater(profile.cap_per_question, profile.expected_cost, name)
            fast = profile.fast()
            self.assertLessEqual(fast.planned_members, profile.planned_members)
            providers = {spec.model.split("/")[0] for spec in profile.forecasters}
            self.assertEqual(providers, {"anthropic", "openai", "google"}, name)


class StateTests(unittest.TestCase):
    def test_attempts_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            state = RunState(path, max_attempts=2)
            state.record_failure(7)
            state.record_failure(7)
            self.assertTrue(state.exhausted(7))
            state.save()
            again = RunState(path, max_attempts=2)
            self.assertTrue(again.exhausted(7))
            again.record_success(7, 0.5, "seasonal", "standard")
            self.assertFalse(again.exhausted(7))

    def test_corrupt_file_is_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            path.write_text("{not json", encoding="utf-8")
            self.assertEqual(RunState(path).attempts(1), 0)


if __name__ == "__main__":
    unittest.main()
