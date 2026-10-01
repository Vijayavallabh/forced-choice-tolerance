"""The learned solver, the joint-vs-marginal distinction, and the long build."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

import mcq_audit
from learned_probe import (AUDITED, ORDERINGS, PLAIN, audited_columns, build_matrix,
                           cross_validate, feature_names, fit, item_features,
                           rank_only_columns)


class FeatureTests(unittest.TestCase):
    def test_rank_enters_as_indicators_not_as_a_number(self):
        """A linear score has to be able to say "the second-smallest"."""
        x = item_features(["12.5", "7.25", "19.0", "3.4"])
        self.assertIsNotNone(x)
        self.assertEqual(x.shape, (4, len(ORDERINGS) * 4 + len(PLAIN)))
        block = x[:, :4]                      # the value ordering's indicators
        self.assertTrue(np.array_equal(block.sum(axis=1), np.ones(4)))
        self.assertTrue(np.array_equal(block.sum(axis=0), np.ones(4)))

    def test_the_value_indicator_matches_the_audit_s_own_rank(self):
        options = ["12.5", "7.25", "19.0", "3.4"]
        x = item_features(options)
        for slot in range(4):
            rank = int(np.argmax(x[slot, :4]))
            self.assertEqual(rank, mcq_audit.ordering_rank(options, "sorted value", slot))

    def test_non_numeric_options_are_declined(self):
        self.assertIsNone(item_features(["alpha", "beta", "gamma", "delta"]))

    def test_the_three_families_nest(self):
        audited, rank_only = set(audited_columns(4)), set(rank_only_columns(4))
        self.assertTrue(audited < rank_only)
        self.assertEqual(len(audited), len(AUDITED) * 4)
        self.assertEqual(len(rank_only), len(ORDERINGS) * 4)

    def test_audited_columns_name_the_audited_orderings(self):
        names = feature_names(4)
        picked = {names[i].rsplit(" rank ", 1)[0] for i in audited_columns(4)}
        self.assertEqual(picked, set(AUDITED))
        self.assertEqual(picked, {"value", "length", "roundness", "isolation"})


class FitTests(unittest.TestCase):
    def _file(self, keyed_rank, n=40):
        """Items whose key always sits at one value rank: a solver should find it."""
        items = []
        for index in range(n):
            base = 10 + index
            values = sorted([base, base + 3, base + 7, base + 11])
            key = values[keyed_rank]
            options = [str(key)] + [str(v) for v in values if v != key]
            items.append({"options": options, "rank": keyed_rank,
                          "cluster": f"c{index % 8}"})
        return build_matrix(items)

    def test_it_learns_a_rank_rule_it_can_express(self):
        x, clusters = self._file(1)
        picks, tied = cross_validate(x[:, :, rank_only_columns(4)], clusters,
                                     400, 0.5, 1e-3)
        accuracy = sum(hit for _, hit in picks) / len(picks)
        self.assertGreater(accuracy, 0.9, "the second-smallest rule was not learned")
        self.assertEqual(tied, 0)

    def test_a_tie_is_not_resolved_to_the_key(self):
        """The key is slot 0, so ``np.argmax`` would score every tie correct.

        A solver with weights of exactly zero ties on every item and should
        land at chance, not at 100%. This is the failure the roundness family
        of ``key_identity.py`` runs into for real.
        """
        x, clusters = self._file(3)
        flat = np.zeros((x.shape[0], x.shape[1], 1))
        picks, tied = cross_validate(flat, clusters, 1, 0.0, 0.0)
        accuracy = sum(hit for _, hit in picks) / len(picks)
        self.assertEqual(tied, len(picks))
        self.assertLess(abs(accuracy - 0.25), 0.08,
                        f"ties are not being broken at random: {accuracy:.2%}")

    def test_weights_are_finite_and_shaped(self):
        x, _ = self._file(2)
        weights = fit(x, 50, 0.5, 1e-3)
        self.assertEqual(weights.shape, (x.shape[2],))
        self.assertTrue(np.isfinite(weights).all())

    def test_an_empty_file_returns_no_weights(self):
        self.assertIsNone(fit(np.zeros((0, 4, 3))))


class JointTests(unittest.TestCase):
    def test_marginal_and_joint_fits_differ(self):
        from mcq_audit import JOINT, fit_pair_weights
        rows = [["12.5", "7.25", "19.0", "3.4"], ["0.05", "0.12", "0.09", "0.3"],
                ["1200", "900", "1500", "1750"], ["4.5%", "6.2%", "3.1%", "8.8%"]]
        items = [{"options": o, "rank": 0} for o in rows * 8]
        marginal, _ = fit_pair_weights(items, 4, 11)
        joint, _ = fit_pair_weights(items, 4, 11, joint=True)
        self.assertNotIn(JOINT, marginal)
        self.assertIn(JOINT, joint)
        self.assertEqual(len(joint[JOINT]), 16)
        self.assertIn("written length", marginal)

    def test_the_joint_flag_needs_the_construction(self):
        out = subprocess.run(
            [sys.executable, "mcq_audit.py", "--joint-draw", "--search", "sample"],
            capture_output=True, text=True,
            cwd=str(Path(__file__).resolve().parents[1]))
        self.assertNotEqual(out.returncode, 0)
        self.assertIn("--joint-draw needs --search construct", out.stderr)



if __name__ == "__main__":
    unittest.main()
