import json
import unittest
from pathlib import Path

from channel_survey import (_f_sf, bixbench_v10, bixbench_v15, chi_square_uniform,
                            cluster_robust_uniformity, credit_lower_bound, grouped_cv_credit,
                            null_in_sample_credit, numeric_ranks, survey_one)

RESULT = Path("results/channel_survey.json")
PROVENANCE = Path("data/external/survey/PROVENANCE.json")


class RankTests(unittest.TestCase):
    def test_ranks_generalise_beyond_four_options(self):
        self.assertEqual(numeric_ranks(["5", "1", "3", "2", "4"]), [4, 0, 2, 1, 3])
        self.assertEqual(numeric_ranks(["0.2", "0.1"]), [1, 0])

    def test_non_numeric_ties_and_text_are_refused(self):
        self.assertIsNone(numeric_ranks(["1", "1", "2", "3"]))
        self.assertIsNone(numeric_ranks(["1", "two", "3", "4"]))

    def test_chi_square_scales_with_the_option_count(self):
        rec = chi_square_uniform({r: 25 for r in range(10)}, 10, 250)
        self.assertAlmostEqual(rec["chi2"], 0.0)
        self.assertEqual(rec["df"], 9)
        skewed = chi_square_uniform({0: 100, 1: 50, 2: 50, 3: 50}, 4, 250)
        self.assertLess(skewed["p_value"], 0.01)


class ClusterRobustTests(unittest.TestCase):
    def test_the_f_tail_matches_a_reference(self):
        try:
            from scipy.stats import f as fdist
        except ImportError:
            self.skipTest("scipy not installed")
        for x, d1, d2 in ((0.5, 3, 42), (2.9, 3, 42), (10.3, 3, 36), (1.0, 9, 57), (40.0, 2, 3)):
            self.assertAlmostEqual(_f_sf(x, d1, d2), fdist.sf(x, d1, d2), places=10)

    def test_items_as_their_own_clusters_give_neymans_chi_square(self):
        """With every item its own cluster the robust covariance is the multinomial one at
        the estimated shares, so the Wald statistic is Neyman's chi-square times (N-1)/N."""
        ranks = [0] * 30 + [1] * 60 + [2] * 40 + [3] * 20
        out = cluster_robust_uniformity([{"cluster": None, "rank": r} for r in ranks], 4)
        n = len(ranks)
        observed = [ranks.count(j) for j in range(4)]
        neyman = sum((o - n / 4) ** 2 / o for o in observed)
        self.assertAlmostEqual(out["wald"] * n, neyman * (n - 1), places=6)
        self.assertEqual(out["n_clusters"], n)

    def test_clustered_items_weaken_the_evidence(self):
        """The same rank shares carried by blocks of identical items: the i.i.d. test
        counts every copy, the cluster-robust one counts the blocks."""
        blocks = [1] * 12 + [0] * 5 + [2] * 6 + [3] * 7
        items = [{"cluster": f"c{b}", "rank": r} for b, r in enumerate(blocks) for _ in range(6)]
        counts = {j: sum(i["rank"] == j for i in items) for j in range(4)}
        iid = chi_square_uniform(counts, 4, len(items))["p_value"]
        robust = cluster_robust_uniformity(items, 4)
        self.assertEqual(robust["n_clusters"], 30)
        self.assertLess(iid, 0.001)
        self.assertGreater(robust["p_value"], 10 * iid)

    def test_too_few_clusters_give_no_test(self):
        items = [{"cluster": c, "rank": r} for c, r in (("a", 0), ("a", 1), ("b", 2))]
        self.assertIsNone(cluster_robust_uniformity(items, 4)["p_value"])

    def test_the_in_sample_maximum_is_biased_upward_under_a_uniform_rank(self):
        """The best of k estimated shares beats 1/k with no channel at all, and less so
        on a larger file."""
        small, large = null_in_sample_credit(105, 4, reps=4000), null_in_sample_credit(1263, 4, reps=4000)
        self.assertGreater(small["mean"], 0.03)
        self.assertLess(large["mean"], small["mean"])
        self.assertGreater(small["q95"], small["mean"])


class SelectorTests(unittest.TestCase):
    """The estimator must not be the leave-one-out selector that failed."""

    def test_near_tied_ranks_do_not_produce_zero_accuracy(self):
        # 40 items split almost evenly between two ranks. Leave-one-out picks the
        # other rank every time and scores 0; K-fold must report about a third.
        items = ([{"rank": 1, "cluster": None}] * 20 + [{"rank": 2, "cluster": None}] * 21)
        rec = grouped_cv_credit(items, 4, reps=200)
        self.assertGreater(rec["accuracy"], 0.2)
        self.assertLessEqual(rec["n_folds"], 10)

    def test_a_dominant_rank_is_recovered(self):
        items = ([{"rank": 1, "cluster": f"c{i}"} for i in range(60)]
                 + [{"rank": 3, "cluster": f"d{i}"} for i in range(10)])
        rec = grouped_cv_credit(items, 4, reps=200)
        self.assertGreater(rec["accuracy"], 0.7)
        self.assertEqual(set(rec["rank_selected_per_fold"]), {1})

    def test_held_out_selection_is_biased_low_under_uniformity(self):
        """Why the survey does not use it: the bias is structural, not small."""
        items = [{"rank": i % 4, "cluster": f"c{i}"} for i in range(400)]
        rec = grouped_cv_credit(items, 4, reps=200)
        self.assertLess(rec["accuracy"], 0.22)      # chance is 0.25

    def test_folds_are_never_single_items_when_groups_are_many(self):
        items = [{"rank": i % 4, "cluster": None} for i in range(500)]
        rec = grouped_cv_credit(items, 4, reps=100)
        self.assertEqual(rec["n_folds"], 10)
        self.assertEqual(rec["n_groups"], 500)


class CreditBoundTests(unittest.TestCase):
    """The estimator the verdict actually rests on."""

    def test_a_uniform_benchmark_yields_no_positive_bound(self):
        items = [{"rank": i % 4, "cluster": f"c{i}"} for i in range(400)]
        rec = credit_lower_bound(items, 4, reps=2000)
        self.assertLessEqual(rec["credit_lower_bound"], 0.0)

    def test_a_skewed_benchmark_yields_a_positive_bound_below_the_plug_in(self):
        items = ([{"rank": 1, "cluster": f"a{i}"} for i in range(200)]
                 + [{"rank": r, "cluster": f"b{i}-{r}"}
                    for i in range(50) for r in (0, 2, 3)])
        rec = credit_lower_bound(items, 4, reps=2000)
        plug_in = 200 / 350 - 0.25
        self.assertGreater(rec["credit_lower_bound"], 0.0)
        self.assertLess(rec["credit_lower_bound"], plug_in)
        self.assertEqual(rec["argmax_rank"], 1)

    def test_the_bound_is_a_lower_bound_on_the_realised_maximum(self):
        items = [{"rank": r, "cluster": f"c{i}"}
                 for i, r in enumerate([0] * 40 + [1] * 60 + [2] * 50 + [3] * 50)]
        shares = [40 / 200, 60 / 200, 50 / 200, 50 / 200]
        rec = credit_lower_bound(items, 4, reps=2000)
        self.assertLess(rec["lower_bound_on_worst_case"], max(shares) + 1e-9)

    def test_the_bonferroni_level_scales_with_the_option_count(self):
        items = [{"rank": i % 10, "cluster": f"c{i}"} for i in range(500)]
        rec = credit_lower_bound(items, 10, reps=500)
        self.assertAlmostEqual(rec["bonferroni_level"], 0.05 / 10)
        self.assertEqual(len(rec["per_rank_lower_bound"]), 10)


class SurveyUnitTests(unittest.TestCase):
    def test_a_file_with_too_few_numeric_items_is_not_assessed(self):
        records = [{"options": ["a", "b", "c", "d"], "key": 0, "n_options": 4,
                    "cluster": None} for _ in range(50)]
        rec = survey_one("toy", records, reps=50)
        self.assertNotIn("exploitable", rec)
        self.assertEqual(rec["n_numeric_items"], 0)

    def test_the_modal_option_count_is_used(self):
        records = [{"options": [str(i), str(i + 1), str(i + 2), str(i + 3), str(i + 4)],
                    "key": 0, "n_options": 5, "cluster": None} for i in range(40)]
        records += [{"options": ["1", "2", "3"], "key": 0, "n_options": 3, "cluster": None}]
        rec = survey_one("toy", records, reps=50)
        self.assertEqual(rec["modal_n_options"], 5)
        self.assertAlmostEqual(rec["chance"], 0.2)


class LoaderTests(unittest.TestCase):
    def test_bixbench_v15_loads_with_the_key_first(self):
        if not Path("data/bixbench.jsonl").exists():
            self.skipTest("question file not present")
        records = list(bixbench_v15())
        self.assertEqual(len(records), 205)
        self.assertTrue(all(r["key"] == 0 for r in records))
        self.assertTrue(all(r["n_options"] == len(r["options"]) for r in records))

    def test_bixbench_v10_recovers_the_key_from_the_presented_letter(self):
        records = list(bixbench_v10())
        if not records:
            self.skipTest("v1.0 run not vendored")
        self.assertTrue(all(r["key"] == 0 for r in records))
        # Duplicated option values do occur upstream; the rank computation
        # refuses them rather than the loader dropping the row.
        self.assertTrue(all(r["n_options"] == len(r["options"]) for r in records))
        self.assertTrue(all(isinstance(o, str) for r in records for o in r["options"]))


class ReportedSurveyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not RESULT.exists():
            raise unittest.SkipTest("survey not generated")
        cls.doc = json.loads(RESULT.read_text())
        cls.by_name = {r["benchmark"]: r for r in cls.doc["benchmarks"]}

    def test_the_survey_covers_the_reported_number_of_files(self):
        self.assertEqual(self.doc["summary"]["n_benchmark_files"], 18)
        self.assertEqual(self.doc["summary"]["n_assessed"], 9)

    def test_five_channels_are_open_and_four_rest_on_enough_groups(self):
        opened = {r["benchmark"] for r in self.doc["benchmarks"] if r.get("exploitable")}
        self.assertEqual(opened, {"BixBench v1.5", "SciQ", "MMLU-Pro", "MMLU",
                                  "LAB-Bench SeqQA"})
        self.assertEqual(self.doc["summary"]["n_channel_open"], 5)
        self.assertEqual(self.doc["summary"]["n_channel_open_with_enough_groups"], 4)
        seqqa = self.by_name["LAB-Bench SeqQA"]
        self.assertFalse(seqqa["bootstrap_reliable"],
                         "SeqQA's four groups should be flagged as too few")

    def test_every_bound_is_below_its_plug_in_estimate(self):
        for rec in self.doc["benchmarks"]:
            if not rec.get("credit_bound"):
                continue
            self.assertLess(rec["credit_bound"]["credit_lower_bound"],
                            rec["plug_in_credit"] + 1e-9, rec["benchmark"])

    def test_mmlu_pro_is_the_large_sample_case_the_paper_cites(self):
        rec = self.by_name["MMLU-Pro"]
        self.assertEqual(rec["modal_n_options"], 10)
        self.assertGreater(rec["n_numeric_items"], 1200)
        self.assertGreater(rec["credit_bound"]["credit_lower_bound"], 0.05)
        self.assertGreater(rec["chi_square_vs_uniform"]["chi2"], 500)

    def test_aqua_rat_is_the_counterexample(self):
        rec = self.by_name["AQuA-RAT"]
        self.assertGreater(rec["chi_square_vs_uniform"]["p_value"], 0.5)
        self.assertFalse(rec["exploitable"])
        self.assertGreater(rec["n_numeric_items"], 100)

    def test_bixbench_agrees_with_the_standalone_analysis(self):
        rec = self.by_name["BixBench v1.5"]
        artifacts = json.loads(Path("results/option_artifacts.json").read_text())
        v15 = next(b for b in artifacts["benchmarks"] if b["benchmark"] == "BixBench v1.5")
        self.assertEqual(rec["n_numeric_items"], v15["n_numeric_items"])
        self.assertAlmostEqual(rec["interior_rate"], v15["interior_rate"], places=9)
        for r in range(4):
            self.assertAlmostEqual(rec["key_by_rank"][str(r)],
                                   v15["rank_fractions"][str(r)], places=9)

    def test_seven_of_nine_reject_uniformity(self):
        rejected = [r for r in self.doc["benchmarks"]
                    if "chi_square_vs_uniform" in r
                    and r["chi_square_vs_uniform"]["p_value"] < 0.05]
        self.assertEqual(len(rejected), 7)
        self.assertEqual(self.doc["summary"]["n_rejecting_uniformity"], 7)

    def test_five_of_nine_reject_uniformity_with_items_clustered(self):
        """SeqQA's four subtasks and MedMCQA's subjects carry their i.i.d. rejections."""
        rejected = {r["benchmark"] for r in self.doc["benchmarks"]
                    if (r.get("cluster_robust_vs_uniform") or {}).get("p_value") is not None
                    and r["cluster_robust_vs_uniform"]["p_value"] < 0.05}
        self.assertEqual(rejected, {"BixBench v1.5", "BixBench v1.0", "MMLU", "MMLU-Pro", "SciQ"})
        self.assertEqual(self.doc["summary"]["n_rejecting_uniformity_cluster_robust"], 5)

    def test_converter_keeps_the_key_and_the_released_distractor_order(self):
        """survey_to_jsonl is what makes the ten-option results rebuildable."""
        from survey_to_jsonl import convert
        path = Path("data/external/survey/mmlu_pro.jsonl.gz")
        if not path.exists():
            self.skipTest("vendored survey not present")
        import gzip, json as _json
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            source = [_json.loads(line) for line in handle]
        converted = list(convert(path))
        self.assertEqual(len(converted), len(source))
        for row, original in zip(converted[:200], source[:200]):
            options, key = original["options"], int(original["key"])
            self.assertEqual(row["ideal"], options[key])
            self.assertEqual(row["distractors"],
                             [o for i, o in enumerate(options) if i != key])
            self.assertEqual(1 + len(row["distractors"]), len(options))
            self.assertEqual(row["cluster"], original["cluster"])

    def test_vendored_option_sets_are_provenanced(self):
        if not PROVENANCE.exists():
            self.skipTest("survey provenance not present")
        import hashlib
        doc = json.loads(PROVENANCE.read_text())
        self.assertEqual(len(doc["files"]), 8)
        for entry in doc["files"]:
            blob = Path(entry["local_path"]).read_bytes()
            self.assertEqual(hashlib.sha256(blob).hexdigest(), entry["sha256"],
                             entry["name"])


if __name__ == "__main__":
    unittest.main()
