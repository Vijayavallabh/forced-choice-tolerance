"""Tests for the repair-operator frontier.

Two things here have already been wrong once and are held as tests.

The borrowed arms must draw from the *empirical multiset* of a subject's keys.
Drawing from the distinct values instead reweights the subject towards its rare
written forms, and a per-option solver reads exactly that reweighting: it was
worth several points of apparent leak before it was caught.

The arms must hold the same items in the same order, or a difference between
two rows of Table 2 is the item set rather than the operator.
"""
import json
import random
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

from frontier_align import signature
from keyblind_operators import parse_marked
from learned_probe_nonlinear import add_set_context, hashed_ngrams
from repair_frontier import draw_distinct, numeric_rows, parse_values


class DrawTests(unittest.TestCase):
    def test_the_draw_follows_the_multiset_not_the_distinct_values(self):
        """A value keyed nine times is nine times as likely to be borrowed."""
        pool = ["7"] * 90 + ["3%"] * 10
        rng = random.Random(0)
        drawn = Counter()
        for _ in range(2000):
            drawn.update(draw_distinct(pool, 1, set(), rng))
        share = drawn["3%"] / sum(drawn.values())
        self.assertAlmostEqual(share, 0.10, delta=0.03)

    def test_the_draw_never_repeats_a_value_or_hits_the_key(self):
        pool = ["1", "2", "2", "3", "3", "3", "4"]
        rng = random.Random(1)
        for _ in range(200):
            drawn = draw_distinct(pool, 3, {"4"}, rng)
            self.assertEqual(len(drawn), 3)
            self.assertEqual(len(set(drawn)), 3)
            self.assertNotIn("4", drawn)

    def test_the_draw_stops_rather_than_looping_when_the_pool_is_too_small(self):
        drawn = draw_distinct(["1", "1", "1"], 3, set(), random.Random(2))
        self.assertEqual(drawn, ["1"])


class ParseTests(unittest.TestCase):
    def test_a_comma_separated_value_keeps_its_first_digit(self):
        """The regression. ``lstrip`` over a character class ate it."""
        self.assertEqual(parse_values("1,2,3\n1,3,4\n2,3,4", set(), 3),
                         ["1,2,3", "1,3,4", "2,3,4"])

    def test_a_negative_value_keeps_its_sign(self):
        self.assertEqual(parse_values("-14\n0.5\n25%", set(), 3),
                         ["-14", "0.5", "25%"])

    def test_a_written_value_already_in_the_options_is_not_reused(self):
        self.assertEqual(parse_values("12\n7\n9\n", {"7"}, 3), ["12", "9"])

    def test_numbering_the_model_adds_is_stripped(self):
        self.assertEqual(parse_values("1. 40\n2. 55\n3. 61\n", set(), 3),
                         ["40", "55", "61"])


class AlignmentTests(unittest.TestCase):
    def test_a_signature_separates_two_items_sharing_a_key(self):
        a = {"ideal": "4", "question": "How many?", "distractors": ["1"]}
        b = {"ideal": "4", "question": "How old?", "distractors": ["9"]}
        self.assertNotEqual(signature(a), signature(b))

    def test_a_signature_ignores_the_distractors_the_operator_rewrote(self):
        a = {"ideal": "4", "question": "How many?", "distractors": ["1", "2"]}
        b = {"ideal": "4", "question": "How many?", "distractors": ["8", "9"]}
        self.assertEqual(signature(a), signature(b))


class ShippedArmTests(unittest.TestCase):
    """The files the paper reports on, checked as files."""

    ARMS = ("released_matched", "rank_uniform", "exchangeable", "imitation",
            "key_marginal_near", "key_marginal")

    def setUp(self):
        self.paths = {arm: Path(f"build/mmlu_frontier_{arm}_aligned.jsonl")
                      for arm in self.ARMS}
        if not all(p.exists() for p in self.paths.values()):
            self.skipTest("the aligned frontier arms are not built")
        self.rows = {arm: [json.loads(line) for line in
                           path.read_text(encoding="utf-8").splitlines() if line.strip()]
                     for arm, path in self.paths.items()}

    def test_every_arm_holds_the_same_items_in_the_same_order(self):
        reference = [signature(row) for row in self.rows["released_matched"]]
        for arm in self.ARMS:
            self.assertEqual([signature(row) for row in self.rows[arm]], reference,
                             f"{arm} does not line up with the released arm")

    def test_every_arm_keeps_the_key_and_four_distinct_options(self):
        for arm, rows in self.rows.items():
            for i, row in enumerate(rows):
                options = [str(row["ideal"]).strip()] + [str(d).strip()
                                                         for d in row["distractors"]]
                self.assertEqual(len(options), 4, f"{arm} item {i}")
                self.assertEqual(len(set(options)), 4,
                                 f"{arm} item {i} repeats an option: {options}")

    def test_the_borrowed_arms_use_only_keys_of_other_items(self):
        """Borrowed distractors are keys, so they are human-written by construction.

        The pool is every numeric item carrying question text, not just the
        aligned subset, because the draw happens before alignment.
        """
        source = Path("build/mmlu_questions.jsonl")
        if not source.exists():
            # 7 MB of upstream MMLU, which the bundle is too near its size
            # ceiling to carry. Say how to get it rather than dead-ending.
            self.skipTest("build/mmlu_questions.jsonl not built; "
                          "run: python3 fetch_mmlu_questions.py")
        keys = {str(row["ideal"]).strip() for row in numeric_rows(source)}
        for arm in ("key_marginal", "key_marginal_near"):
            for row in self.rows[arm]:
                for option in row["distractors"]:
                    self.assertIn(str(option).strip(), keys,
                                  f"{arm} borrowed a value that is no item's key")

    def test_no_shipped_option_is_a_mangled_number(self):
        """The list-marker bug wrote ",2,3" into the distractors of 14 items.

        A leading separator appears only on a distractor and never on a key, so
        it makes the key easier to find: the bug pushed the measurement in the
        direction of the paper's own claim, which is the worst direction.
        """
        for arm, rows in self.rows.items():
            for i, row in enumerate(rows):
                for option in [row["ideal"], *row["distractors"]]:
                    text = str(option).strip()
                    self.assertFalse(text.startswith((",", "%")) or text.endswith(","),
                                     f"{arm} item {i} carries {text!r}")


if __name__ == "__main__":
    unittest.main()


class KeyBlindOperatorTests(unittest.TestCase):
    """The wrong-step writer's reply is parsed off its marker, not off its working."""

    def test_the_values_come_from_the_answers_line_and_not_the_working(self):
        reply = ("We need 120/4. A student might divide by 2 instead, giving 60, "
                 "or forget the division entirely.\n"
                 "ANSWERS: 60; 120; 15; 30; 240")
        self.assertEqual(parse_marked(reply, 5), ["60", "120", "15", "30", "240"])

    def test_a_reply_with_no_marker_is_dropped_rather_than_guessed(self):
        self.assertIsNone(parse_marked("The answer is probably 42, or maybe 21.", 5))

    def test_the_last_marker_wins_when_the_writer_restates_it(self):
        reply = "ANSWERS: 1; 2; 3\nOn reflection:\nANSWERS: 4; 5; 6"
        self.assertEqual(parse_marked(reply, 3), ["4", "5", "6"])

    def test_the_marker_parser_keeps_signs_and_separators(self):
        self.assertEqual(parse_marked("ANSWERS: -14; 1,200; 3.5%; 0.02; -0.5", 5),
                         ["-14", "1,200", "3.5%", "0.02", "-0.5"])

    def test_duplicates_inside_one_reply_are_dropped(self):
        self.assertEqual(parse_marked("ANSWERS: 7; 7; 8; 9; 9", 5), ["7", "8", "9"])


class NonlinearReaderTests(unittest.TestCase):
    """The character reader must be deterministic and must not see the other options."""

    def test_the_hash_is_stable_across_calls(self):
        a = hashed_ngrams("12.5%", 256)
        b = hashed_ngrams("12.5%", 256)
        self.assertTrue((a == b).all())

    def test_different_strings_hash_differently(self):
        self.assertFalse((hashed_ngrams("12.5%", 512)
                          == hashed_ngrams("12.6%", 512)).all())

    def test_set_context_is_invariant_to_permuting_the_other_options(self):
        rows = np.stack([hashed_ngrams(t, 64) for t in ("1", "2", "3", "4")])
        first = add_set_context(rows)[0]
        shuffled = np.stack([rows[0], rows[3], rows[1], rows[2]])
        self.assertTrue(np.allclose(first, add_set_context(shuffled)[0], atol=1e-6))

    def test_the_option_context_row_is_the_option_alone(self):
        rows = np.stack([hashed_ngrams(t, 64) for t in ("1", "2", "3", "4")])
        self.assertEqual(rows.shape[1], 64)
        self.assertEqual(add_set_context(rows).shape[1], 192)
