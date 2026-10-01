import json
import random
import unittest
from pathlib import Path

from mcq_audit import audit, build_items, coerce_list, format_like, repair_row
from option_artifacts import numeric_rank, parse_number
from text_artifacts import RULES, cross_validated, load_items, pick

AUDIT = Path("results/mcq_audit.json")


class FormatPreservationTests(unittest.TestCase):
    """A repair that reformats options trades a numeric leak for a surface one."""

    def test_decimal_places_are_copied(self):
        self.assertEqual(format_like("0.0500", 0.0637), "0.0637")
        self.assertEqual(format_like("0.05", 0.0637), "0.06")

    def test_scientific_notation_and_exponent_width_are_copied(self):
        rendered = format_like("7.820659E-05", 1.2345e-4)
        self.assertIn("E-", rendered)
        self.assertEqual(rendered.split("E")[0].split(".")[1].__len__(), 6)
        self.assertAlmostEqual(parse_number(rendered), 1.2345e-4, places=9)

    def test_exponent_is_zero_padded_like_the_template(self):
        self.assertTrue(format_like("3.23E-08", 5e-8).endswith("E-08"))

    def test_percent_and_grouping_are_preserved(self):
        self.assertTrue(format_like("35%", 42.0).endswith("%"))
        self.assertEqual(format_like("3,556.0", 4201.0), "4,201.0")

    def test_integers_stay_integers(self):
        self.assertEqual(format_like("120", 137.4), "137")


class RepairTests(unittest.TestCase):
    def test_key_value_is_never_changed(self):
        rng = random.Random(1)
        options = ["0.05", "0.015", "0.075", "0.105"]
        for _ in range(50):
            repaired = repair_row(options, rng)
            if repaired:
                self.assertEqual(repaired[0], options[0])

    def test_repaired_options_stay_distinct_and_numeric(self):
        rng = random.Random(2)
        for options in (["0.05", "0.015", "0.075", "0.105"],
                        ["1.9E-05", "3.23E-08", "4.56E-04", "2.15E-06"],
                        ["3556.0", "1247.5", "2584.0", "4891.0"]):
            for _ in range(20):
                repaired = repair_row(options, rng)
                if repaired is None:
                    continue
                values = [parse_number(o) for o in repaired]
                self.assertNotIn(None, values)
                self.assertEqual(len(set(values)), 4)

    def test_achieved_rank_matches_the_drawn_rank(self):
        """Rendering rounds; the repair must verify placement, not assume it."""
        rng = random.Random(3)
        ranks = []
        for _ in range(400):
            repaired = repair_row(["0.05", "0.015", "0.075", "0.105"], rng)
            if repaired:
                ranks.append(numeric_rank(repaired, 0))
        self.assertGreater(len(ranks), 300)
        for rank in range(4):
            share = ranks.count(rank) / len(ranks)
            self.assertGreater(share, 0.15, f"rank {rank} under-represented: {share:.2f}")
            self.assertLess(share, 0.35, f"rank {rank} over-represented: {share:.2f}")

    def test_repair_declines_rather_than_emitting_a_bad_item(self):
        # one decimal place cannot separate four values this close together
        rng = random.Random(4)
        self.assertIsNone(repair_row(["0.0", "0.0", "0.0", "0.0"], rng))


class SurfaceFamilyTests(unittest.TestCase):
    def test_every_rule_returns_a_valid_option_index(self):
        options = ["alpha", "a much longer option here", "beta", "12"]
        for name, rule in RULES.items():
            self.assertIn(pick(rule, options, "what is alpha?"), range(4), name)

    def test_cross_validation_reports_chance_on_random_keys(self):
        """A family with no generalising member must not beat chance."""
        rng = random.Random(7)
        items = []
        for i in range(200):
            opts = [f"option-{rng.randrange(1000)}" for _ in range(4)]
            items.append({"options": opts, "key": 0, "question": "q",
                          "cluster": f"c{i % 40}", "id": str(i), "row_index": i})
        cv = cross_validated(items, reps=400)
        self.assertLess(cv["accuracy"], 0.40)

    def test_non_numeric_bixbench_items_show_no_confirmed_leak(self):
        path = Path("results/text_artifacts.json")
        if not path.exists():
            self.skipTest("text artifacts not generated")
        rec = json.loads(path.read_text())["subsets"]["non_numeric"]["cross_validated"]
        self.assertFalse(rec["above_chance"],
                         "a surface leak is now detected; the manuscript says none was")


class EndToEndRepairTests(unittest.TestCase):
    def setUp(self):
        if not AUDIT.exists():
            self.skipTest("mcq_audit not run")
        self.doc = json.loads(AUDIT.read_text())

    def test_released_file_is_flagged(self):
        self.assertIn("numeric rule family", self.doc["before"]["coordinates_firing"])
        self.assertTrue(self.doc["before"]["numeric_family"]["leaks"])

    def test_repair_closes_every_coordinate(self):
        after = self.doc["after_repair"]
        self.assertEqual(after["coordinates_firing"], [])
        self.assertLessEqual(after["numeric_family"]["cv_ci95"][0], 0.25)
        self.assertLessEqual(after["surface_family"]["cv_ci95"][0], 0.25)

    def test_repair_does_not_create_a_surface_leak(self):
        """The first repair we wrote reformatted options and did exactly this."""
        self.assertFalse(self.doc["after_repair"]["surface_family"]["leaks"])

    def test_repair_covers_every_numeric_item(self):
        self.assertEqual(self.doc["n_repaired"],
                         self.doc["before"]["numeric_family"]["n_items"])

    def test_repaired_file_keeps_every_keyed_answer(self):
        repaired = Path("results/repaired/bixbench_v15_repaired.jsonl")
        if not repaired.exists():
            self.skipTest("repaired file not written")
        with Path("data/bixbench.jsonl").open(encoding="utf-8") as handle:
            original = {json.loads(l)["question_id"]: json.loads(l)["ideal"]
                        for l in handle if l.strip()}
        with repaired.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    self.assertEqual(row["ideal"], original[row["question_id"]],
                                     f"{row['question_id']}: repair altered a keyed answer")


class ReachabilityTests(unittest.TestCase):
    """The repair can only draw a rank the arithmetic allows."""

    def test_reachability_falls_towards_the_extreme_ranks(self):
        from mcq_audit import reachability
        path = Path("results/mcq_audit.json")
        if not path.exists():
            self.skipTest("audit not run")
        shares = json.loads(path.read_text())["reachability"]["reachable_share_by_rank"]
        self.assertGreater(shares["0"], shares["3"])
        self.assertGreater(shares["0"], 0.95)

    def test_ten_options_are_harder_to_repair_than_four(self):
        narrow = Path("results/mcq_audit.json")
        wide = Path("results/mcq_audit_mmlu_pro.json")
        if not (narrow.exists() and wide.exists()):
            self.skipTest("audits not run")
        a = json.loads(narrow.read_text())["reachability"]["mean_reachable_share"]
        b = json.loads(wide.read_text())["reachability"]["mean_reachable_share"]
        self.assertGreater(a, b)
        self.assertGreater(a, 0.9)
        self.assertLess(b, 0.9)

    def test_a_small_integer_key_cannot_reach_a_high_rank(self):
        """The structural reason, in one item: there is nothing below a 1."""
        from mcq_audit import redraw_row
        rng = random.Random(0)
        self.assertIsNone(redraw_row(["1", "2", "3", "4"], rng, target=3))
        self.assertIsNotNone(redraw_row(["1", "2", "3", "4"], rng, target=0))


class LoaderTests(unittest.TestCase):
    def test_distractors_parse_from_a_stringified_list(self):
        self.assertEqual(coerce_list("['a', 'b']"), ["a", "b"])
        self.assertEqual(coerce_list(["a"]), ["a"])
        self.assertEqual(coerce_list(None), [])

    def test_a_requested_option_count_keeps_only_matching_items(self):
        rows = [{"ideal": "1", "distractors": ["2", "3"]},
                {"ideal": "1", "distractors": ["2", "3", "4"]}]
        items, k = build_items(rows, "ideal", "distractors", None, "question", 4)
        self.assertEqual((len(items), k), (1, 4))

    def test_the_modal_option_count_is_detected(self):
        rows = ([{"ideal": "1", "distractors": ["2", "3", "4", "5", "6"]}] * 3
                + [{"ideal": "1", "distractors": ["2", "3", "4"]}])
        items, k = build_items(rows, "ideal", "distractors", None, "question")
        self.assertEqual((len(items), k), (3, 6))

    def test_a_tie_in_option_count_breaks_towards_the_larger(self):
        rows = [{"ideal": "1", "distractors": ["2", "3"]},
                {"ideal": "1", "distractors": ["2", "3", "4"]}]
        _, k = build_items(rows, "ideal", "distractors", None, "question")
        self.assertEqual(k, 4)


if __name__ == "__main__":
    unittest.main()
