"""Tests for the one-option-at-a-time solver.

The point of ``key_identity.py`` is that none of its features looks at any other
option, so a margin it reports cannot be option geometry. That property is
mechanical and therefore testable: permuting the other options must leave a
given option's features exactly where they were.
"""
import random
import unittest

import numpy as np

from key_identity import FEATURES, ROUNDNESS, build_matrix, option_features
from learned_probe import cross_validate
from mcq_audit import parse_number


def _item(options, cluster="c0"):
    return {"options": options, "cluster": cluster, "rank": 0}


class IsolationTests(unittest.TestCase):
    def test_a_feature_row_ignores_the_other_options(self):
        """The whole claim, as an identity."""
        a = build_matrix([_item(["12", "3.75", "100", "0.5"])])[0][0]
        b = build_matrix([_item(["12", "0.5", "3.75", "100"])])[0][0]
        self.assertTrue(np.array_equal(a[0], b[0]))
        shuffled = build_matrix([_item(["12", "9999.125", "7", "2.5"])])[0][0]
        self.assertTrue(np.array_equal(a[0], shuffled[0]))

    def test_no_feature_name_is_relative(self):
        """A guard against a set-relative feature being added by hand later."""
        for name in FEATURES:
            self.assertNotIn("relative", name)
            self.assertNotIn("median", name)
            self.assertNotIn("rank", name)
        self.assertTrue(set(ROUNDNESS) <= set(FEATURES))

    def test_the_features_read_what_they_say(self):
        row = dict(zip(FEATURES, option_features("1,200", parse_number("1,200"))))
        self.assertEqual(row["has separator"], 1.0)
        self.assertEqual(row["is integer"], 1.0)
        self.assertEqual(row["trailing zeros"], 2.0)
        self.assertEqual(row["significant digits"], 2.0)
        self.assertEqual(row["decimal places"], 0.0)
        row = dict(zip(FEATURES, option_features("0.0450%", parse_number("0.0450%"))))
        self.assertEqual(row["has percent"], 1.0)
        self.assertEqual(row["decimal places"], 4.0)
        self.assertEqual(row["has point"], 1.0)
        row = dict(zip(FEATURES, option_features("−7", parse_number("−7"))))
        self.assertEqual(row["has minus"], 1.0, "U+2212 is a minus sign")


class TieTests(unittest.TestCase):
    def test_the_roundness_family_ties_and_is_scored_at_chance_anyway(self):
        """Four integer features tie constantly; the tie-break must not favour slot 0.

        Built here so it cannot be read as a property of one dataset: every
        option is the same written shape, so every score is identical and the
        solver has nothing to go on. Resolving ties to the first index would
        score this 100%.
        """
        rng = random.Random(4)
        items = []
        for i in range(400):
            values = rng.sample(range(11, 99), 4)
            items.append(_item([str(v) for v in values], cluster=f"c{i % 8}"))
        x, clusters = build_matrix(items)
        columns = [FEATURES.index(n) for n in ROUNDNESS]
        picks, tied = cross_validate(x[:, :, columns], clusters, 200, 0.5, 1e-3)
        self.assertGreater(tied, 0.5 * len(picks),
                           "expected ties; the fixture stopped being degenerate")
        accuracy = sum(hit for _, hit in picks) / len(picks)
        self.assertLess(accuracy, 0.45, f"ties favour the key: {accuracy:.1%}")


if __name__ == "__main__":
    unittest.main()
