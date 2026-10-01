"""The four orderings no repair targets, and the coupling that predicts transfer."""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mcq_audit
from held_out_orderings import (HELD_OUT, coupling, decimal_place_scores,
                                digit_sum_scores, leading_digit_scores,
                                string_order_scores)
from mcq_audit import feature_rank

RESULT = Path("results/held_out_orderings.json")


class ScoreTests(unittest.TestCase):
    def test_leading_digit_is_the_first_significant_one(self):
        self.assertEqual(leading_digit_scores(["0.00345", "12", "9", "700"]),
                         [3.0, 1.0, 9.0, 7.0])

    def test_leading_digit_ignores_the_sign_however_it_is_written(self):
        """U+2212 is accepted by parse_number, so it must not become a digit."""
        self.assertEqual(leading_digit_scores(["−12.5", "-12.5", "12.5", "3"]),
                         [1.0, 1.0, 1.0, 3.0])

    def test_decimal_places_is_not_significant_digits(self):
        self.assertEqual(decimal_place_scores(["0.0004", "1.50", "12", "3.1"]),
                         [4.0, 2.0, 0.0, 1.0])
        # the same four under roundness: 0.0004 is the *roundest*, not the longest
        self.assertEqual(mcq_audit.significant_digits("0.0004"), 1.0)

    def test_digit_sum_adds_the_digits(self):
        self.assertEqual(digit_sum_scores(["123", "1.23", "900", "0.05"]),
                         [6.0, 6.0, 9.0, 5.0])

    def test_string_order_is_the_text_tie_break(self):
        """A constant score leaves feature_rank sorting on the text alone."""
        options = ["9.5", "12.7", "100", "3"]
        self.assertEqual(string_order_scores(options), [0.0] * 4)
        ranks = [feature_rank(options, string_order_scores(options), i) for i in range(4)]
        # lexicographic: "100" < "12.7" < "3" < "9.5"
        self.assertEqual(ranks, [3, 1, 0, 2])

    def test_every_held_out_score_declines_non_numeric_options(self):
        for name, fn in HELD_OUT.items():
            self.assertIsNone(fn(["alpha", "beta", "gamma", "delta"]), name)

    def test_none_of_them_is_an_ordering_the_audit_reports(self):
        audited = {"sorted value", "lexical isolation", "written length",
                   "significant digits"}
        self.assertEqual(audited, set(mcq_audit.SURFACE_ORDERINGS) | {"sorted value"},
                         "the audited orderings changed; the held-out set may overlap")
        self.assertFalse(set(HELD_OUT) & audited,
                         "a held-out ordering is also an audited one")


class CouplingTests(unittest.TestCase):
    def _items(self, option_sets):
        return [{"options": opts, "rank": 0, "cluster": "c0", "key": opts[0]}
                for opts in option_sets]

    def test_same_digit_width_makes_text_order_the_value_order(self):
        items = self._items([["11", "22", "33", "44"], ["1.5", "2.5", "3.5", "9.5"]])
        got = coupling(items, 4)
        self.assertEqual(got["lexicographic_is_value_order"], 1.0)

    def test_mixed_widths_break_it(self):
        items = self._items([["9.5", "12.7", "100", "3"]])
        got = coupling(items, 4)
        self.assertEqual(got["lexicographic_is_value_order"], 0.0)

    def test_distinct_score_counts_are_reported(self):
        items = self._items([["1.0", "2.0", "3.0", "4.0"]])
        got = coupling(items, 4)
        # one decimal place on all four; four different leading digits
        self.assertEqual(got["decimal places"]["mean_distinct_scores"], 1.0)
        self.assertEqual(got["leading digit"]["mean_distinct_scores"], 4.0)


class DiagnosticAttributionTests(unittest.TestCase):
    """The held-out orderings can be attributed without being repaired."""

    def test_promoting_them_adds_exactly_four_channels(self):
        import channel_attribution
        before = set(channel_attribution.CHANNELS)
        added = channel_attribution.add_held_out_channels()
        after = set(channel_attribution.CHANNELS)
        try:
            self.assertEqual(set(added), set(HELD_OUT))
            self.assertEqual(after - before, set(HELD_OUT))
            # measuring more than you repair is fine; the reverse is not
            self.assertTrue(set(mcq_audit.ORDERINGS) <= after)
        finally:
            for name in HELD_OUT:
                channel_attribution.CHANNELS.pop(name, None)
                channel_attribution.DIAGNOSTIC_ORDERINGS.pop(name, None)


class PlaceboStemlessTests(unittest.TestCase):
    """The stem-withheld placebo is the control for a stem-withheld claim."""

    def test_it_is_available_but_not_a_default_arm(self):
        from no_data_probe import ARMS, EXTRA_ARMS
        self.assertIn("placebo_stemless", EXTRA_ARMS)
        self.assertNotIn("placebo_stemless", ARMS)

    def test_the_conditions_carry_the_placebo_options_with_no_question(self):
        from no_data_probe import WITHHELD, build_conditions
        items = [{"question_id": f"q{i}", "cluster": "c0",
                  "question": f"how many cells in sample {i}?",
                  "options": ["12.5", "7.25", "19.0", "3.4"],
                  "key": 0, "rank": 1} for i in range(6)]
        records = build_conditions(items, 1, 20260920, k=4,
                                   arms=("placebo", "placebo_stemless"))
        by_arm = {}
        for record in records:
            by_arm.setdefault(record["arm"], []).append(record)
        self.assertEqual(set(by_arm), {"placebo", "placebo_stemless"})
        for shown, withheld in zip(by_arm["placebo"], by_arm["placebo_stemless"]):
            self.assertEqual(shown["question_id"], withheld["question_id"])
            self.assertIn(WITHHELD, withheld["prompt"])
            self.assertNotIn("how many cells", withheld["prompt"])
            # same options, same letters: only the question differs
            self.assertEqual(shown["target_letter"], withheld["target_letter"])
            self.assertEqual(shown["key_rank"], withheld["key_rank"])


@unittest.skipUnless(RESULT.exists(), "held_out_orderings.py has not been run")
class ShippedResultTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.doc = json.loads(RESULT.read_text(encoding="utf-8"))

    def test_both_option_counts_are_present_with_a_clean_reference(self):
        self.assertEqual(set(self.doc["files"]), {"4", "10"})
        self.assertEqual(set(self.doc["clean"]), {"4", "10"})
        for k in ("4", "10"):
            self.assertEqual(set(self.doc["clean"][k]), set(HELD_OUT))

    def test_four_options_close_and_ten_do_not(self):
        four, ten = self.doc["files"]["4"], self.doc["files"]["10"]
        self.assertFalse([name for name, stats in four["repaired"].items()
                          if name != "_coupling" and stats["leaks"]])
        self.assertTrue([name for name, stats in ten["repaired"].items()
                         if name != "_coupling" and stats["leaks"]])

    def test_the_coupling_separates_the_two_files(self):
        four = self.doc["files"]["4"]["repaired"]["_coupling"]
        ten = self.doc["files"]["10"]["repaired"]["_coupling"]
        self.assertGreater(four["lexicographic_is_value_order"],
                           ten["lexicographic_is_value_order"] + 0.2)


if __name__ == "__main__":
    unittest.main()
