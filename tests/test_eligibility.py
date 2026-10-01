import unittest
from datetime import date
from temporal_eligibility import add_months, eligibility


def material(public_by=None, first_public=None, verified=False):
    return {"public_by": public_by, "first_public": first_public,
            "first_appearance_verified": verified, "evidence_urls": ["https://example.org/evidence"]}


class EligibilityTests(unittest.TestCase):
    def setUp(self):
        self.original = {"analysis": material("2024-01-01")}
        self.twin = {"analysis": material("2026-01-01", "2026-01-01", True),
                     "data": material("2026-01-01", "2026-01-01", True)}

    def test_known_valid_pair(self):
        self.assertEqual(eligibility(self.original, self.twin, "2025-01-01")["status"], "eligible")

    def test_missing_cutoff(self):
        self.assertEqual(eligibility(self.original, self.twin, None)["status"], "unknown")

    def test_paper_date_is_not_first_appearance(self):
        self.twin["analysis"]["first_appearance_verified"] = False
        self.assertEqual(eligibility(self.original, self.twin, "2025-01-01")["status"], "unknown")

    def test_old_data_cannot_pass_as_new(self):
        self.twin["data"] = material("2020-04-22")
        self.assertEqual(eligibility(self.original, self.twin, "2025-01-01")["status"], "ineligible")
        self.assertEqual(eligibility(self.original, self.twin, "2025-01-01", novelty="analysis_novel")["status"], "eligible")

    def test_calendar_month_margin(self):
        self.assertEqual(add_months(date(2024, 1, 31), 1), date(2024, 2, 29))

    def test_boundary_is_strict(self):
        self.twin["analysis"] = material("2025-04-01", "2025-04-01", True)
        self.assertEqual(eligibility(self.original, self.twin, "2025-01-01")["status"], "ineligible")

    def test_original_known_to_be_late(self):
        self.original["analysis"] = material("2025-02-01", "2025-02-01", True)
        self.assertEqual(eligibility(self.original, self.twin, "2025-01-01")["status"], "ineligible")

    def test_contradictory_dates_rejected(self):
        self.twin["data"] = material("2020-01-01", "2026-01-01", True)
        with self.assertRaises(ValueError):
            eligibility(self.original, self.twin, "2025-01-01")


if __name__ == "__main__":
    unittest.main()
