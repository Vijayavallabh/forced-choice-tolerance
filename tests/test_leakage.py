import json
import math
import unittest
from pathlib import Path

from no_data_baseline import (cluster_bootstrap_ci, cluster_permutation_diff,
                              icc_and_design_effect, load_zero_shot, mean)
from option_artifacts import (analyse, chi_square_uniform, loco_rule_accuracy,
                              numeric_rank, parse_number)

RESULTS = Path("results/option_artifacts.json")
BASELINE = Path("results/no_data_baseline.json")


class NumericParsingTests(unittest.TestCase):
    def test_scientific_percent_and_comma_forms(self):
        self.assertEqual(parse_number("7.820659E-05"), 7.820659e-05)
        self.assertEqual(parse_number("35%"), 35.0)
        self.assertEqual(parse_number("3,556.0"), 3556.0)

    def test_non_numeric_returns_none(self):
        for text in ("PCSK5", "0.5 x S <= L", "", "p < 0.05", None):
            self.assertIsNone(parse_number(text))

    def test_rank_is_position_among_sorted_values(self):
        # key 0.05 with distractors 0.015, 0.075, 0.105 -> second smallest
        self.assertEqual(numeric_rank(["0.05", "0.015", "0.075", "0.105"], 0), 1)
        self.assertEqual(numeric_rank(["0.105", "0.015", "0.05", "0.075"], 0), 3)

    def test_ties_and_non_numeric_sets_are_skipped(self):
        self.assertIsNone(numeric_rank(["1", "1", "2", "3"], 0))
        self.assertIsNone(numeric_rank(["1", "2", "three", "4"], 0))
        self.assertIsNone(numeric_rank(["1", "2", "3"], 0))


class InferenceTests(unittest.TestCase):
    def test_bootstrap_interval_brackets_the_point_estimate(self):
        clusters = {f"c{i}": [1.0, 0.0, 1.0] for i in range(30)}
        lo, hi = cluster_bootstrap_ci(clusters, mean, reps=400)
        self.assertLessEqual(lo, 2 / 3)
        self.assertGreaterEqual(hi, 2 / 3)

    def test_identical_arms_give_a_large_permutation_p(self):
        a = {f"a{i}": [1.0, 0.0] for i in range(12)}
        b = {f"b{i}": [1.0, 0.0] for i in range(12)}
        result = cluster_permutation_diff(a, b, reps=500)
        self.assertAlmostEqual(result["difference"], 0.0)
        self.assertGreater(result["p_value"], 0.5)

    def test_separated_arms_give_a_small_permutation_p(self):
        a = {f"a{i}": [1.0, 1.0] for i in range(12)}
        b = {f"b{i}": [0.0, 0.0] for i in range(12)}
        self.assertLess(cluster_permutation_diff(a, b, reps=500)["p_value"], 0.01)

    def test_design_effect_is_one_without_between_cluster_variance(self):
        clusters = {f"c{i}": [1.0, 0.0] for i in range(20)}
        rec = icc_and_design_effect(clusters)
        self.assertAlmostEqual(rec["icc"], 0.0)
        self.assertAlmostEqual(rec["design_effect"], 1.0)

    def test_perfectly_clustered_outcomes_shrink_the_effective_sample(self):
        clusters = {f"c{i}": [float(i % 2)] * 4 for i in range(20)}
        rec = icc_and_design_effect(clusters)
        self.assertGreater(rec["icc"], 0.9)
        self.assertLess(rec["effective_n"], rec["n_questions"] / 3)

    def test_chi_square_matches_a_known_value(self):
        # 40/20/20/20 against uniform expectation 25 each
        rec = chi_square_uniform({0: 40, 1: 20, 2: 20, 3: 20})
        self.assertAlmostEqual(rec["chi2"], 12.0)
        self.assertLess(rec["p_value_upper_bound"], 0.01)
        self.assertAlmostEqual(chi_square_uniform({0: 25, 1: 25, 2: 25, 3: 25})["chi2"], 0.0)


class CrossValidationTests(unittest.TestCase):
    def test_a_rule_that_holds_everywhere_survives(self):
        items = [{"rank": 1, "cluster": f"c{i}"} for i in range(10)]
        self.assertAlmostEqual(loco_rule_accuracy(items)["accuracy"], 1.0)

    def test_a_rule_fitted_to_noise_is_rejected(self):
        # each cluster favours a different rank, so no rank generalises
        items = [{"rank": i % 4, "cluster": f"c{i}"} for i in range(40)]
        self.assertLess(loco_rule_accuracy(items)["accuracy"], 0.35)

    def test_held_out_cluster_is_excluded_from_rule_selection(self):
        # one deviant cluster must not be able to pick its own rank
        items = [{"rank": 0, "cluster": f"c{i}"} for i in range(9)]
        items += [{"rank": 3, "cluster": "deviant"}] * 5
        rec = loco_rule_accuracy(items)
        self.assertEqual(rec["folds"], 10)
        self.assertLess(rec["accuracy"], 1.0)


class JoinTests(unittest.TestCase):
    def test_question_ids_join_case_insensitively(self):
        """Two published rows use capitalised ids; a naive join drops them."""
        path = Path("data/external/zero_shot_v15/gpt-4o-grader-mcq-refusal-False.csv")
        if not path.exists():
            self.skipTest("published baselines not vendored")
        loaded = load_zero_shot(path)
        self.assertEqual(len(loaded), 205)
        self.assertIn("bix-33-q6", loaded)
        self.assertIn("bix-47-q3", loaded)


class ReportedResultTests(unittest.TestCase):
    """Guard the numbers the manuscript states against the released results."""

    @classmethod
    def setUpClass(cls):
        if not (RESULTS.exists() and BASELINE.exists()):
            raise unittest.SkipTest("results not generated")
        cls.art = {b["benchmark"]: b for b in json.loads(RESULTS.read_text())["benchmarks"]}
        cls.base = json.loads(BASELINE.read_text())

    def test_v15_interior_excess_is_significant(self):
        rec = self.art["BixBench v1.5"]
        self.assertEqual(rec["n_numeric_items"], 105)
        self.assertGreater(rec["interior_rate"], 0.80)
        self.assertGreater(rec["interior_rate_ci95"][0], 0.5)

    def test_v15_rule_beats_chance_and_cross_validates(self):
        rec = self.art["BixBench v1.5"]
        self.assertGreater(rec["best_single_rank_rule"]["ci95"][0], 0.25)
        self.assertAlmostEqual(rec["cross_validated_rule"]["accuracy"],
                               rec["best_single_rank_rule"]["accuracy"], places=6)

    def test_v10_rule_does_not_cross_validate(self):
        rec = self.art["BixBench v1.0"]
        self.assertGreater(rec["interior_rate_ci95"][0], 0.5)
        self.assertLess(rec["cross_validated_rule"]["accuracy"],
                        rec["best_single_rank_rule"]["accuracy"] - 0.1)

    def test_presentation_order_control_is_null(self):
        ctl = self.art["BixBench v1.0"]["presentation_order_control"]["chi_square_vs_uniform"]
        self.assertGreater(ctl["p_value_upper_bound"], 0.05)

    def test_no_data_mcq_exceeds_chance_but_open_ended_does_not(self):
        runs = {r["run"]: r for r in self.base["runs"]}
        mcq = runs["gpt-4o-grader-mcq-refusal-False"]
        self.assertEqual(mcq["n_questions_joined"], 205)
        self.assertGreater(mcq["ci95_cluster_bootstrap"][0], 0.25)
        self.assertLess(runs["gpt-4o-grader-openended"]["ci95_cluster_bootstrap"][1], 0.10)

    def test_pooled_provenance_gap_does_not_survive_restriction(self):
        contrasts = {c["restriction"]: c for c in self.base["confounding_analysis"]["contrasts"]}
        self.assertLess(contrasts["all questions"]["p_value"], 0.05)
        for restricted in ("numeric-option questions only", "LLM-graded questions only"):
            self.assertGreater(contrasts[restricted]["p_value"], 0.05)

    def test_reproduces_the_authors_published_accuracies(self):
        published = json.loads(Path("data/external/zero_shot_summary_v15.json").read_text())
        for run in self.base["runs"]:
            if run["run"] in published:
                self.assertTrue(math.isclose(run["accuracy"], published[run["run"]]["accuracy"],
                                             rel_tol=1e-9),
                                f"{run['run']} does not reproduce the published accuracy")


if __name__ == "__main__":
    unittest.main()
