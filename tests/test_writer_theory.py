"""Tests for the three propositions and the agentic read-out.

Each of these guards a claim that would otherwise rest on prose. The first
group enumerates finite laws, so it checks the statements exactly rather than
to within a simulation's noise. The last group guards a parsing bug that cost
sixteen accuracy points before it was caught: reading the first capital letter
after ``FINAL:`` takes the T of "The answer is B", and the arm then reports a
compliant model as one with a strong option preference.
"""
import math
import unittest

import numpy as np

from agentic_probe import build_rollouts, final_letter, run_code
from text_options import clean_like, fold_of
from writer_theory import (detailed_balance_residual, gamma_exact,
                           joint_item_conditional, joint_item_independent,
                           joint_reject_against_key, joint_sorted_distractors,
                           joint_symmetric, reversible_kernel)


class PropositionOneTests(unittest.TestCase):
    def test_sorted_distractors_separate_gamma_from_exchangeability(self):
        """The counterexample to necessity: Gamma = 0 without an exchangeable tuple."""
        law, exchangeable = joint_sorted_distractors(n_values=8, k=3)
        self.assertAlmostEqual(gamma_exact(law, 3), 0.0, places=12)
        self.assertFalse(exchangeable)

    def test_gamma_is_never_negative(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            kappa = rng.dirichlet(np.ones(5))
            nu = rng.dirichlet(np.ones(5))
            for k in (2, 3):
                self.assertGreaterEqual(
                    gamma_exact(joint_item_independent(kappa, nu, k), k), -1e-12)


class PropositionTwoTests(unittest.TestCase):
    def test_item_independent_closes_only_at_the_key_marginal(self):
        kappa = np.array([0.40, 0.30, 0.20, 0.10])
        for k in (2, 3, 4):
            self.assertAlmostEqual(
                gamma_exact(joint_item_independent(kappa, kappa, k), k), 0.0, places=12)
            for other in (np.full(4, 0.25), kappa[::-1].copy()):
                self.assertGreater(
                    gamma_exact(joint_item_independent(kappa, other, k), k), 0.05)

    def test_reversibility_closes_the_channel_at_two_options(self):
        kappa = np.array([0.50, 0.30, 0.20])
        nu = reversible_kernel(kappa)
        self.assertAlmostEqual(detailed_balance_residual(kappa, nu), 0.0, places=12)
        np.testing.assert_allclose(nu.sum(axis=1), 1.0, atol=1e-12)
        self.assertAlmostEqual(
            gamma_exact(joint_item_conditional(kappa, nu, 2), 2), 0.0, places=12)
        not_reversible = np.array([[0.0, 0.9, 0.1], [0.1, 0.0, 0.9], [0.9, 0.1, 0.0]])
        self.assertGreater(detailed_balance_residual(kappa, not_reversible), 0.1)
        self.assertGreater(
            gamma_exact(joint_item_conditional(kappa, not_reversible, 2), 2), 0.1)

    def test_redrawing_against_the_key_opens_what_a_symmetric_draw_closes(self):
        """The assembler's own loop is a dependence on the key."""
        for kappa, expect_open in ((np.full(6, 1 / 6), False),
                                   (np.array([.45, .25, .15, .08, .05, .02]), True)):
            symmetric = gamma_exact(joint_symmetric(kappa, 4), 4)
            sequential = gamma_exact(joint_reject_against_key(kappa, 4), 4)
            self.assertAlmostEqual(symmetric, 0.0, places=12)
            if expect_open:
                self.assertGreater(sequential, 0.15)
            else:
                self.assertAlmostEqual(sequential, 0.0, places=12)


class AgenticReadoutTests(unittest.TestCase):
    def test_final_letter_ignores_the_first_capital_of_a_sentence(self):
        self.assertEqual(final_letter("FINAL: The answer is B", 4), "B")
        self.assertEqual(final_letter("FINAL: Based on my analysis, C.", 4), "C")
        self.assertEqual(final_letter("FINAL: D. I am confident", 10), "D")

    def test_final_letter_prefers_the_last_marker_and_a_named_option(self):
        self.assertEqual(final_letter("FINAL: A\nFINAL: D", 4), "D")
        self.assertEqual(final_letter("FINAL: I think it is option D here", 10), "D")
        self.assertEqual(final_letter("FINAL: (C)", 4), "C")

    def test_final_letter_refuses_a_letter_the_item_does_not_have(self):
        self.assertIsNone(final_letter("FINAL: Z", 4))
        self.assertIsNone(final_letter("FINAL: J", 4))
        self.assertEqual(final_letter("FINAL: J", 10), "J")
        self.assertIsNone(final_letter("no marker at all", 4))

    def test_gold_letter_follows_the_permutation(self):
        rows = [{"ideal": "7", "distractors": ["1", "2", "3"],
                 "question": "q", "cluster": "c"}]
        for roll in build_rollouts(rows, "withheld", draws=8, seed=1):
            shown = dict(line[1:].split(") ", 1)
                         for line in roll["user"].split("\n") if line.startswith("("))
            self.assertEqual(shown[roll["gold"]], "7")
            self.assertIn("[withheld]", roll["user"])

    def test_the_sandbox_runs_arithmetic_and_refuses_the_rest(self):
        self.assertEqual(run_code("print(sorted([3,1,2])[1])"), "2")
        self.assertIn("arithmetic only", run_code("import os\nprint(os.getcwd())"))
        self.assertIn("arithmetic only", run_code("print(open('/etc/passwd').read())"))


class TextOptionTests(unittest.TestCase):
    def test_clean_control_keeps_the_shape_and_the_file_own_vocabulary(self):
        items = [{"options": [f"opt{i}{j}" for j in range(4)], "cluster": f"c{i % 3}"}
                 for i in range(20)]
        pool = {o for item in items for o in item["options"]}
        clean = clean_like(items, 4, seed="s")
        self.assertEqual(len(clean), len(items))
        for item in clean:
            self.assertEqual(len(set(item["options"])), 4)
            self.assertTrue(set(item["options"]) <= pool)
        self.assertEqual([i["cluster"] for i in clean], [i["cluster"] for i in items])

    def test_folds_are_deterministic_and_bounded(self):
        labels = [f"group-{i}" for i in range(500)]
        folds = {fold_of(label, 20) for label in labels}
        self.assertLessEqual(len(folds), 20)
        self.assertEqual(fold_of("group-7", 20), fold_of("group-7", 20))


if __name__ == "__main__":
    unittest.main()


class AgenticMechanismTests(unittest.TestCase):
    """The post-hoc split on the agent's own words."""

    @staticmethod
    def _rollout(cluster, reply, right):
        return {"arm": "file", "cluster": cluster, "gold": "B",
                "answer": "B" if right else "C", "reply": [reply]}

    def test_middle_matches_its_synonyms_and_not_its_absence(self):
        from agentic_mechanism import MIDDLE
        for said in ("the median value", "the MIDDLE one", "a central estimate"):
            self.assertTrue(MIDDLE.search(said.lower()), said)
        for unsaid in ("the second smallest", "the largest", "an outlier"):
            self.assertIsNone(MIDDLE.search(unsaid.lower()), unsaid)

    def test_split_partitions_the_rollouts_and_scores_each_side(self):
        from agentic_mechanism import split
        rows = ([self._rollout(f"c{i}", "the middle one", True) for i in range(6)]
                + [self._rollout(f"c{i}", "the largest one", False) for i in range(6)])
        out = split(rows, draws=200, seed=0)
        self.assertEqual(out["all"]["n"], 12)
        self.assertEqual(out["says_middle"]["n"] + out["does_not"]["n"], 12)
        self.assertAlmostEqual(out["says_middle"]["accuracy"], 100.0)
        self.assertAlmostEqual(out["does_not"]["accuracy"], 0.0)
        self.assertAlmostEqual(out["reaches_for_it"], 50.0)

    def test_an_empty_side_does_not_raise(self):
        from agentic_mechanism import split
        out = split([self._rollout("c0", "the largest one", False)], draws=10, seed=0)
        self.assertEqual(out["says_middle"]["n"], 0)
        self.assertTrue(math.isnan(out["says_middle"]["accuracy"]))
