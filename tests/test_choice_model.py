import json
import random
import statistics
import unittest
from pathlib import Path

from choice_model import (CHANCE, N_OPTIONS, _chi2_sf, decompose, fit, geometry_bounds,
                          goodness_of_fit, key_distribution, log_likelihood,
                          lrt_no_knowledge, pairs_from_v10, ranks_of, realised_geometry,
                          strip_label)

RESULT = Path("results/choice_model.json")


def synthesise(lam, b, p, n, seed, clusters=46):
    """Draw (key rank, chosen rank) pairs from the two-source model itself."""
    rng = random.Random(seed)
    out = []
    for i in range(n):
        k = rng.choices(range(N_OPTIONS), weights=p)[0]
        r = k if rng.random() < lam else rng.choices(range(N_OPTIONS), weights=b)[0]
        out.append({"cluster": f"c{i % clusters}", "key_rank": k,
                    "chosen_rank": r, "correct": int(r == k)})
    return out


class GeometryLemmaTests(unittest.TestCase):
    """The claim the repair rests on, checked rather than asserted."""

    def test_uniform_key_ranks_give_chance_for_every_preference(self):
        uniform = [0.25] * N_OPTIONS
        rng = random.Random(0)
        for _ in range(500):
            weights = [rng.random() for _ in range(N_OPTIONS)]
            total = sum(weights)
            b = [w / total for w in weights]
            self.assertAlmostEqual(realised_geometry(uniform, b)["G"], CHANCE, places=12)

    def test_credit_is_bounded_by_the_extremes_of_p(self):
        p = [0.124, 0.514, 0.295, 0.067]
        rng = random.Random(1)
        for _ in range(500):
            weights = [rng.random() for _ in range(N_OPTIONS)]
            b = [w / sum(weights) for w in weights]
            g = realised_geometry(p, b)["G"]
            self.assertLessEqual(g, max(p) + 1e-12)
            self.assertGreaterEqual(g, min(p) - 1e-12)

    def test_worst_case_is_attained_by_a_point_mass(self):
        p = [0.124, 0.514, 0.295, 0.067]
        b = [0.0, 1.0, 0.0, 0.0]          # "always take the second-smallest"
        self.assertAlmostEqual(realised_geometry(p, b)["G"], 0.514, places=12)
        self.assertAlmostEqual(geometry_bounds(p)["max_geometry_credit"], 0.514 - CHANCE,
                               places=12)

    def test_a_uniform_p_is_reported_as_not_exploitable(self):
        self.assertFalse(geometry_bounds([0.25] * N_OPTIONS)["exploitable"])
        self.assertTrue(geometry_bounds([0.124, 0.514, 0.295, 0.067])["exploitable"])

    def test_letter_preference_alone_earns_nothing_under_shuffling(self):
        """Shuffled presentation makes letter independent of rank, so b is flat."""
        rng = random.Random(3)
        letter_pref = [0.1, 0.2, 0.6, 0.1]
        counts = [0] * N_OPTIONS
        for _ in range(40000):
            permutation = list(range(N_OPTIONS))
            rng.shuffle(permutation)       # permutation[slot] = source rank
            slot = rng.choices(range(N_OPTIONS), weights=letter_pref)[0]
            counts[permutation[slot]] += 1
        b = [c / sum(counts) for c in counts]
        for value in b:
            self.assertAlmostEqual(value, 0.25, delta=0.015)


class RecoveryTests(unittest.TestCase):
    def test_no_knowledge_is_recovered_as_no_knowledge(self):
        data = synthesise(0.0, [0.10, 0.45, 0.35, 0.10], [0.124, 0.514, 0.295, 0.067],
                          3000, 11)
        fitted = fit(data)
        self.assertLess(fitted["lam"], 0.03)
        self.assertGreater(lrt_no_knowledge(data, fitted)["p_value"], 0.05)

    def test_knowledge_is_recovered_when_present(self):
        for truth in (0.10, 0.25, 0.40):
            data = synthesise(truth, [0.25] * N_OPTIONS, [0.124, 0.514, 0.295, 0.067],
                              3000, 12)
            fitted = fit(data)
            self.assertAlmostEqual(fitted["lam"], truth, delta=0.05)

    def test_rank_preference_is_recovered(self):
        b = [0.10, 0.45, 0.35, 0.10]
        data = synthesise(0.15, b, [0.226, 0.302, 0.308, 0.164], 4000, 13)
        fitted = fit(data)
        for r in range(N_OPTIONS):
            self.assertAlmostEqual(fitted["b"][r], b[r], delta=0.05)

    def test_decomposition_adds_up_exactly(self):
        data = synthesise(0.2, [0.10, 0.45, 0.35, 0.10], [0.124, 0.514, 0.295, 0.067],
                          2000, 14)
        fitted = fit(data)
        rec = decompose(data, fitted["lam"], fitted["b"])
        self.assertAlmostEqual(rec["knowledge_component"] + rec["geometry_component"],
                               rec["fitted_accuracy"] - CHANCE, places=12)

    def test_em_does_not_decrease_the_likelihood(self):
        data = synthesise(0.2, [0.2, 0.3, 0.3, 0.2], [0.25] * N_OPTIONS, 800, 15)
        fitted = fit(data)
        flat = log_likelihood(data, 0.0, [0.25] * N_OPTIONS)
        self.assertGreaterEqual(fitted["loglik"], flat)


class ParsingTests(unittest.TestCase):
    def test_option_labels_are_stripped(self):
        self.assertEqual(strip_label("(B) 0.0031"), "0.0031")
        self.assertEqual(strip_label("A. 12"), "12")
        self.assertEqual(strip_label("plain text"), "plain text")

    def test_rank_vector_positions_each_option(self):
        self.assertEqual(ranks_of([0.05, 0.015, 0.075, 0.105]), [1, 0, 2, 3])

    def test_ties_and_non_numbers_are_refused(self):
        self.assertIsNone(ranks_of([1.0, 1.0, 2.0, 3.0]))
        self.assertIsNone(ranks_of([1.0, None, 2.0, 3.0]))
        self.assertIsNone(ranks_of([]))

    def test_ranks_follow_the_option_count_given(self):
        """The option count is read off the data, so ten options rank fine."""
        self.assertEqual(ranks_of([3.0, 1.0, 2.0]), [2, 0, 1])
        self.assertEqual(ranks_of(list(range(10, 0, -1))), list(range(9, -1, -1)))

    def test_a_wrong_option_count_is_refused_where_it_matters(self):
        """The guard belongs to the reader of a fixed-width release, not to ranking."""
        path = Path("data/external/zero_shot_v10/"
                    "bixbench_llm_baseline_refusal_False_mcq_gpt-4o_1.0.csv")
        if not path.exists():
            self.skipTest("published v1.0 runs not vendored")
        _, skipped = pairs_from_v10(path, n_options=5)
        self.assertGreater(skipped.get("not 5 options", 0), 0)

    def test_key_distribution_sums_to_one(self):
        data = synthesise(0.1, [0.25] * N_OPTIONS, [0.1, 0.5, 0.3, 0.1], 500, 16)
        self.assertAlmostEqual(sum(key_distribution(data)), 1.0, places=12)


class ChiSquareTests(unittest.TestCase):
    def test_matches_known_tail_probabilities(self):
        for x, df, expected in ((3.841459, 1, 0.05), (5.991465, 2, 0.05),
                                (7.814728, 3, 0.05), (19.675138, 11, 0.05)):
            self.assertAlmostEqual(_chi2_sf(x, df), expected, places=5)

    def test_a_correct_fit_is_not_rejected(self):
        data = synthesise(0.15, [0.10, 0.45, 0.35, 0.10], [0.124, 0.514, 0.295, 0.067],
                          3000, 17)
        fitted = fit(data)
        gof = goodness_of_fit(data, fitted["lam"], fitted["b"])
        self.assertGreater(gof["p_value"], 0.01)


class ReportedResultTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not RESULT.exists():
            raise unittest.SkipTest("choice model not fitted")
        cls.doc = json.loads(RESULT.read_text())

    def test_every_run_reports_an_additive_decomposition(self):
        runs = self.doc["published_runs"] + self.doc["open_weight_runs"]
        self.assertGreater(len(runs), 0)
        for run in runs:
            self.assertAlmostEqual(run["knowledge_component"] + run["geometry_component"],
                                   run["fitted_accuracy"] - CHANCE, places=9,
                                   msg=run["run"])

    def test_fitted_preference_matches_the_directly_measured_one(self):
        """The estimator is checked against a quantity it never saw.

        ``b`` is inferred from runs where the question is shown; it is also
        measured directly in the arm where the question is withheld. The two
        need not coincide exactly -- a solver may well prefer differently when it
        has a question to read -- so the claim is about the panel, not about any
        single model: the typical distance is small and none is large.
        """
        distances = [run["b_vs_measured_total_variation"]
                     for run in self.doc["open_weight_runs"]
                     if run.get("b_vs_measured_total_variation") is not None]
        self.assertGreater(len(distances), 5)
        self.assertLess(statistics.median(distances), 0.10)
        self.assertLess(max(distances), 0.25)

    def test_no_solver_realises_more_than_a_small_part_of_the_channel(self):
        """The paper's central negative claim, guarded against the results file."""
        summary = self.doc["realised_vs_worst_case"]
        available = summary["worst_case_available_on_benchmark"]
        self.assertGreater(available, 0.20)
        self.assertLess(summary["max_realised_geometry_credit"], 0.10)
        self.assertLess(summary["max_realised_geometry_credit"], available / 3)

    def test_lambda_orders_with_accuracy_where_answers_are_known(self):
        """Construct validity: lambda must track knowing, not just fit."""
        path = Path("results/choice_model_mmlu.json")
        probe = Path("results/no_data_probe_mmlu.json")
        if not (path.exists() and probe.exists()):
            self.skipTest("MMLU validation not generated")
        fits = {r["run"]: r for r in json.loads(path.read_text())["open_weight_runs"]}
        rows = []
        for model in json.loads(probe.read_text())["models"]:
            block = model["subsets"]["numeric options"]["arms"]
            rows.append((block["original"]["accuracy"], fits[model["model"]]["lambda"],
                         block["original_stemless"]["accuracy"]))
        rows.sort()
        lambdas = [r[1] for r in rows]
        self.assertEqual(lambdas, sorted(lambdas), "lambda does not order with accuracy")
        self.assertLess(lambdas[0], 0.01)
        self.assertGreater(lambdas[-1], 0.5)
        for _, _, withheld in rows:
            self.assertAlmostEqual(withheld, 0.25, delta=0.04,
                                   msg="withholding the question should land at chance")

    def test_the_same_models_show_far_less_knowledge_without_the_data(self):
        for name in ("results/choice_model_mmlu.json", "results/choice_model.json"):
            if not Path(name).exists():
                self.skipTest("fits not generated")
        mmlu = {r["run"]: r for r in
                json.loads(Path("results/choice_model_mmlu.json").read_text())["open_weight_runs"]}
        bix = {r["run"]: r for r in self.doc["open_weight_runs"]}
        shared = [m for m in mmlu if m in bix and mmlu[m]["lambda"] > 0.1]
        self.assertGreater(len(shared), 1)
        for model in shared:
            self.assertGreater(mmlu[model]["lambda"], 3 * bix[model]["lambda"], model)

    def test_uniformity_is_reported_as_closing_the_channel(self):
        rec = self.doc["geometry_bounds"]["BixBench v1.5"]
        self.assertTrue(rec["exploitable"])
        self.assertTrue(rec["uniform_key_ranks_give_chance_for_every_solver"])


if __name__ == "__main__":
    unittest.main()


class RankDependentRecallTests(unittest.TestCase):
    """The refit that bounds what the goodness-of-fit failure can cost.

    Freeing recall per key rank must reduce to the shared model when recall
    really is shared, or a difference between the two fits would be the
    estimator rather than the data.
    """

    @staticmethod
    def _table(lam, b, p, n, seed=0):
        import numpy as np
        rng = np.random.default_rng(seed)
        k = len(b)
        counts = [[0] * k for _ in range(k)]
        for _ in range(n):
            j = int(rng.choice(k, p=p))
            known = rng.random() < (lam[j] if isinstance(lam, (list, tuple)) else lam)
            r = j if known else int(rng.choice(k, p=b))
            counts[j][r] += 1
        return counts

    def test_per_rank_fit_matches_the_shared_fit_when_recall_is_shared(self):
        from rank_dependent_fit import fit_per_rank, fit_shared
        b = [0.10, 0.45, 0.30, 0.15]
        p = [0.12, 0.51, 0.30, 0.07]
        counts = self._table(0.30, b, p, 40000, seed=3)
        lam_a, b_a = fit_shared(counts)
        lam_b, b_b = fit_per_rank(counts)
        self.assertAlmostEqual(lam_a[0], 0.30, delta=0.03)
        self.assertAlmostEqual(sum(pj * lj for pj, lj in zip(p, lam_b)), 0.30, delta=0.04)
        for one, two in zip(b_a, b_b):
            self.assertAlmostEqual(one, two, delta=0.02)

    def test_per_rank_fit_recovers_recall_that_varies_with_the_key_rank(self):
        from rank_dependent_fit import fit_per_rank
        b = [0.25] * 4
        p = [0.25] * 4
        lam = [0.05, 0.60, 0.60, 0.05]
        counts = self._table(lam, b, p, 60000, seed=5)
        fitted, _ = fit_per_rank(counts)
        for want, got in zip(lam, fitted):
            self.assertAlmostEqual(want, got, delta=0.05)

    def test_the_refit_can_move_the_geometry_term_when_the_data_ask_it_to(self):
        """The bound quoted in the text is only worth something if the refit could
        have moved the geometry term. Under a recall that swings from 0.05 to 0.60
        with the key's rank it moves it by more than two points, where on the real
        MMLU draws it moves it by at most 0.19."""
        from rank_dependent_fit import fit_per_rank, fit_shared, report
        counts = self._table([0.05, 0.6, 0.6, 0.05], [0.1, 0.45, 0.3, 0.15],
                             [0.12, 0.51, 0.3, 0.07], 40000, seed=7)
        shared = report(counts, *fit_shared(counts), free=4)
        per_rank = report(counts, *fit_per_rank(counts), free=7)
        moved = abs(shared["geometry_G"] - per_rank["geometry_G"])
        self.assertGreater(moved, 0.02)
        self.assertLess(per_rank["chi2"], shared["chi2"] + 1e-9)

    def test_the_two_fits_agree_where_the_shared_model_is_true(self):
        from rank_dependent_fit import fit_per_rank, fit_shared, report
        counts = self._table(0.30, [0.10, 0.45, 0.30, 0.15],
                             [0.12, 0.51, 0.30, 0.07], 40000, seed=11)
        shared = report(counts, *fit_shared(counts), free=4)
        per_rank = report(counts, *fit_per_rank(counts), free=7)
        self.assertLess(abs(shared["geometry_G"] - per_rank["geometry_G"]), 0.01)
