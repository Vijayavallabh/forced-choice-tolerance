"""Tests for the exchangeable repair: what it guarantees and what it costs.

The claim is narrower than the construction makes it look, and both halves are
tested here. Drawing the key's slot *after* checking that all k slots are legal
makes the key's rank uniform under any ordering that is a function of the
offsets -- the numeric one above all. It does not make the key's own *value*
exchangeable with the distractors', so an ordering that reads the value's
identity rather than its position (roundness) stays open. A repair whose
guarantee is not tested at its boundary is a guarantee about the abstract.
"""
import math
import random
import statistics
import unittest
from collections import Counter

from exchangeable_repair import (MEAN_ABS_DIFF, exchangeable_row, naive_row,
                                 spread)
from mcq_audit import format_like, ordering_rank, parse_number


def _rank_counts(items, draws=1, seed=11, k=4, ordering="sorted value"):
    rng = random.Random(seed)
    counts, misses = Counter(), 0
    for _ in range(draws):
        for options in items:
            result = exchangeable_row(options, rng)
            if result is None:
                misses += 1
                continue
            rank = ordering_rank(result[0], ordering)
            if rank is not None:
                counts[rank] += 1
    return counts, misses


def _chi_square(counts, k):
    total = sum(counts.values())
    expected = total / k
    return sum((counts[r] - expected) ** 2 / expected for r in range(k))


ROOMY = [["50", "20", "80", "35"], ["1200", "800", "1500", "2100"],
         ["4.5", "2.5", "6.5", "9.5"], ["0.42", "0.18", "0.77", "0.55"],
         ["310", "180", "640", "95"], ["7.25", "3.10", "9.80", "5.44"]]


class UniformityTests(unittest.TestCase):
    def test_the_key_value_rank_is_uniform(self):
        counts, misses = _rank_counts(ROOMY, draws=90)
        self.assertEqual(misses, 0)
        # 3 df, 0.1% level. The construction makes this exact rather than
        # approximate, so a failure here is a bug and not a bad seed.
        self.assertLess(_chi_square(counts, 4), 16.27,
                        f"key rank not uniform: {dict(sorted(counts.items()))}")

    def test_drawing_the_slot_before_checking_it_is_not_uniform(self):
        """The trap the all-slots rule exists to avoid, held as a regression.

        ``naive_row`` differs from ``exchangeable_row`` in one line: it draws a
        slot and retries until *that* slot is legal, so it conditions on an
        event that depends on which slot the key took. On items whose written
        style leaves little room below the key that pushes the key downwards.
        The shipped code is what is exercised here, because the paper quotes
        this version's histogram.
        """
        items = [["3", "1", "2", "4"], ["6", "12", "30", "105"],
                 ["4", "1", "2", "3"], ["2", "1", "3", "4"]]
        rng = random.Random(5)
        counts = Counter()
        for _ in range(500):
            for options in items:
                got = naive_row(options, rng)
                if got is not None:
                    counts[ordering_rank(got[0], "sorted value")] += 1
        self.assertGreater(sum(counts.values()), 400)
        self.assertGreater(_chi_square(counts, 4), 16.27,
                           "the naive retry came out uniform; if that is real the "
                           f"all-slots rule can be simplified: {dict(counts)}")

    def test_the_all_slots_rule_is_uniform_on_the_same_items(self):
        """The contrast, on exactly the items the naive retry skews on."""
        items = [["6", "12", "30", "105"], ["50", "20", "80", "35"],
                 ["310", "180", "640", "95"]]
        counts, misses = _rank_counts(items, draws=200, seed=8)
        self.assertGreater(sum(counts.values()), 400)
        self.assertLess(_chi_square(counts, 4), 16.27,
                        f"the all-slots rule is not uniform: {dict(counts)}")

    def test_the_key_value_and_text_survive(self):
        rng = random.Random(3)
        for options in ROOMY:
            result = exchangeable_row(options, rng)
            self.assertIsNotNone(result)
            new = result[0]
            self.assertEqual(new[0], options[0])
            self.assertEqual(len(new), len(options))
            values = [parse_number(o) for o in new]
            self.assertEqual(len(set(values)), len(values))
            self.assertTrue(all(v > 0 for v in values))

    def test_the_slot_is_reported_and_lands_everywhere(self):
        rng = random.Random(17)
        slots = Counter()
        for _ in range(60):
            for options in ROOMY:
                result = exchangeable_row(options, rng)
                if result is not None:
                    slots[result[1]] += 1
        self.assertEqual(set(slots), {0, 1, 2, 3})


class FrontierTests(unittest.TestCase):
    def test_small_integers_admit_no_exchangeable_rewriting(self):
        """Three distinct positive integers below 3 do not exist.

        This is a property of the item's written style, not of the search, and
        it is why the repair reports a feasible subset instead of forcing every
        item. Roughly a tenth of MMLU's numeric items are of this shape.
        """
        rng = random.Random(29)
        self.assertIsNone(exchangeable_row(["3", "1", "2", "4"], rng))

    def test_non_numeric_and_duplicate_options_are_refused(self):
        rng = random.Random(31)
        self.assertIsNone(exchangeable_row(["red", "green", "blue", "grey"], rng))
        self.assertIsNone(exchangeable_row(["10", "10", "20", "30"], rng))


class SpreadTests(unittest.TestCase):
    def test_the_scale_reproduces_the_observed_spread(self):
        """Offsets enter as differences, so the divisor is E|X-Y|, not E|X|.

        Using E|X| would widen every repaired item by sqrt(2), which is the kind
        of error that leaves the file looking regenerated without failing any
        test of the guarantee.
        """
        self.assertAlmostEqual(MEAN_ABS_DIFF, 2 / math.sqrt(math.pi))
        options = ["100", "50", "200", "400"]
        values = [parse_number(o) for o in options]
        sigma = spread(values[0], values, log=True)
        rng = random.Random(2)
        gaps = []
        for _ in range(4000):
            z = [rng.gauss(0.0, 1.0) for _ in range(4)]
            slot = rng.randrange(4)
            new = [values[0] * math.exp(sigma * (x - z[slot])) for x in z]
            gaps += [abs(math.log(v / values[0])) for i, v in enumerate(new)
                     if i != slot]
        observed = statistics.fmean(
            abs(math.log(v / values[0])) for v in values[1:])
        self.assertLess(abs(statistics.fmean(gaps) / observed - 1), 0.1)

    def test_a_degenerate_item_gets_a_default_scale(self):
        self.assertEqual(spread(5.0, [5.0], log=True), 0.25)
        self.assertGreater(spread(5.0, [5.0], log=False), 0)


class ProbeArmTests(unittest.TestCase):
    """The probe can run the exchangeable repair as its repaired arm.

    Shipping a ``--repair-mode`` nobody exercises is how a CLI option becomes a
    claim the code cannot support, so this runs the arm end to end on one item.
    """

    def test_the_repaired_arm_uses_the_exchangeable_rewriting(self):
        from no_data_probe import REPAIRS, CALIBRATED, build_conditions
        self.assertIn("exchangeable", REPAIRS)
        self.assertNotIn("exchangeable", CALIBRATED,
                         "the exchangeable repair fits no target distribution")
        items = [{"question_id": "q0", "cluster": "c", "question": "",
                  "options": ["50", "20", "80", "35"], "key": 0, "rank": 2}]
        records = build_conditions(items, 4, 11, 4,
                                   ("original", "repaired"), repair="exchangeable")
        repaired = [r for r in records if r["arm"] == "repaired"]
        self.assertTrue(repaired)
        self.assertTrue(all(r["redraw_applied"] for r in repaired))
        # The key's value survives; the distractors do not have to.
        for record in repaired:
            self.assertIn("50", record["prompt"])

    def test_an_item_with_no_exchangeable_rewriting_is_left_alone(self):
        from no_data_probe import build_conditions
        items = [{"question_id": "q0", "cluster": "c", "question": "",
                  "options": ["3", "1", "2", "4"], "key": 0, "rank": 2}]
        records = build_conditions(items, 2, 11, 4,
                                   ("original", "repaired"), repair="exchangeable")
        repaired = [r for r in records if r["arm"] == "repaired"]
        self.assertTrue(repaired)
        self.assertFalse(any(r["redraw_applied"] for r in repaired),
                         "an item the repair cannot touch must be flagged, not "
                         "silently shipped as repaired")


if __name__ == "__main__":
    unittest.main()
