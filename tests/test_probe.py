import json
import random
import statistics
import unittest
from collections import Counter
from pathlib import Path

from no_data_probe import (ANSWER_PREFIX, ARMS, LETTERS, MCQ_PROMPT_TEMPLATE, WITHHELD,
                           build_conditions, load_items, option_block, rank_of_each_slot)
from option_artifacts import N_OPTIONS, parse_number
from probe_analysis import (cluster_bootstrap_ci, icc, paired_contrast, per_item,
                            sign_flip_test, spearman)
from rule_solvers import FAMILIES, RANK_FAMILY, SOLVERS

JSONL = Path("data/bixbench.jsonl")
PROBE = Path("results/no_data_probe.json")
RULES = Path("results/no_data_probe_rules.json")


def conditions(n_items=12, draws=2):
    items, k = load_items(JSONL)
    items = items[:n_items]
    return items, build_conditions(items, draws, 20260920, k)


class PromptTests(unittest.TestCase):
    """The prompt must be the benchmark's, not a paraphrase of it."""

    def test_options_render_as_the_benchmark_renders_them(self):
        block = option_block(["a", "b", "c", "d"], [2, 0, 3, 1])
        self.assertEqual(block, "(A) c\n(B) a\n(C) d\n(D) b")
        self.assertFalse(block.endswith("\n"))

    def test_template_keeps_the_upstream_missing_newline(self):
        filled = MCQ_PROMPT_TEMPLATE.format(question="q", options="(A) x")
        self.assertIn("(A) xIMPORTANT", filled)

    def test_answer_prefix_stops_before_the_space(self):
        self.assertEqual(ANSWER_PREFIX, "<answer>")


class ConditionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not JSONL.exists():
            raise unittest.SkipTest("question file not present")
        cls.items, cls.records = conditions()

    def test_every_arm_appears_for_every_item_and_draw(self):
        counts = Counter((r["question_id"], r["draw"]) for r in self.records)
        self.assertTrue(all(v == len(ARMS) for v in counts.values()))

    def test_the_target_letter_points_at_the_keyed_answer(self):
        by_id = {it["question_id"]: it for it in self.items}
        for record in self.records:
            slot = LETTERS.index(record["target_letter"])
            shown = record["options_shown"][slot]
            self.assertEqual(shown, by_id[record["question_id"]]["options"][0],
                             record["question_id"])

    def test_slot_ranks_agree_with_the_values_shown(self):
        for record in self.records:
            if record["slot_ranks"] is None:
                continue
            values = [parse_number(o) for o in record["options_shown"]]
            order = sorted(range(N_OPTIONS), key=lambda i: values[i])
            self.assertEqual(record["slot_ranks"],
                             [order.index(i) for i in range(N_OPTIONS)])

    def test_items_the_redraw_cannot_touch_are_identical_across_arms(self):
        """The control items must carry no arm difference at all."""
        grouped = {}
        for record in self.records:
            if record["is_numeric"]:
                continue
            grouped.setdefault((record["question_id"], record["draw"]), {})
            grouped[(record["question_id"], record["draw"])][record["arm"]] = record["prompt"]
        self.assertTrue(grouped, "no non-numeric items in the sample")
        for arms in grouped.values():
            self.assertEqual(arms["original"], arms["placebo"])
            self.assertEqual(arms["original"], arms["repaired"])

    def test_the_stemless_arms_withhold_the_question(self):
        for record in self.records:
            if record["arm"].endswith("stemless"):
                self.assertIn(WITHHELD, record["prompt"])

    def test_the_placebo_keeps_the_released_rank_and_the_repair_need_not(self):
        placebo = [r for r in self.records if r["arm"] == "placebo" and r["is_numeric"]]
        self.assertTrue(placebo)
        for record in placebo:
            if record["redraw_applied"]:
                self.assertEqual(record["key_rank"], record["released_rank"])

    def test_conditions_are_reproducible(self):
        _, again = conditions()
        self.assertEqual([r["prompt"] for r in self.records], [r["prompt"] for r in again])

    def test_rank_vector_is_none_when_options_are_not_four_distinct_numbers(self):
        self.assertIsNone(rank_of_each_slot(["a", "b", "c", "d"], [0, 1, 2, 3]))
        self.assertIsNone(rank_of_each_slot(["1", "1", "2", "3"], [0, 1, 2, 3]))


class AttributionRebuildTests(unittest.TestCase):
    """The attribution rebuilds the options; it must notice when they differ.

    The outcome files carry no option text, so ``channel_attribution`` redraws
    the repair from the question file and the seed. If the repair has changed
    since the probe ran, every channel is then measured on options the solver
    never saw -- silently, until this guard.
    """

    def _rows_and_conditions(self):
        from no_data_probe import build_conditions
        items = [{"question_id": "q0", "cluster": "c", "question": "",
                  "options": ["1.0", "2.0", "3.0", "4.0"], "key": 0, "rank": 0}]
        records = build_conditions(items, 1, 7, 4, ("original",))
        rows = [{"question_id": r["question_id"], "arm": r["arm"], "draw": r["draw"],
                 "cluster": r["cluster"], "correct": 1,
                 "target_letter": r["target_letter"],
                 "predicted_letter": r["target_letter"],
                 "key_rank": r["key_rank"]} for r in records]
        conditions = {(r["question_id"], r["arm"], r["draw"]): r for r in records}
        return rows, conditions

    def test_a_matching_rebuild_reports_no_mismatch(self):
        from channel_attribution import attribute
        rows, conditions = self._rows_and_conditions()
        out = attribute(rows, conditions, 4, 2)
        self.assertEqual(out["_rebuild"]["mismatched"], 0)
        self.assertEqual(out["_rebuild"]["matched"], len(rows))

    def test_a_changed_repair_is_counted_rather_than_absorbed(self):
        from channel_attribution import attribute
        rows, conditions = self._rows_and_conditions()
        for row in rows:
            row["key_rank"] = (row["key_rank"] + 1) % 4
        out = attribute(rows, conditions, 4, 2)
        self.assertEqual(out["_rebuild"]["mismatched"], len(rows))


class InferenceTests(unittest.TestCase):
    def test_spearman_matches_known_values(self):
        self.assertAlmostEqual(spearman([1, 2, 3, 4], [1, 2, 3, 4]), 1.0)
        self.assertAlmostEqual(spearman([1, 2, 3, 4], [4, 3, 2, 1]), -1.0)
        self.assertAlmostEqual(spearman([1, 2, 3, 4], [1, 2, 4, 3]), 0.8)

    def test_spearman_handles_ties(self):
        self.assertAlmostEqual(spearman([1, 1, 2, 2], [1, 1, 2, 2]), 1.0)

    def test_sign_flip_is_null_on_zero_mean_differences(self):
        clusters = {f"c{i}": [0.1, -0.1] for i in range(20)}
        self.assertGreater(sign_flip_test(clusters, reps=500)["p_value"], 0.5)

    def test_sign_flip_detects_a_consistent_shift(self):
        clusters = {f"c{i}": [0.3, 0.25] for i in range(20)}
        result = sign_flip_test(clusters, reps=500)
        self.assertLess(result["p_value"], 0.01)
        self.assertAlmostEqual(result["difference"], 0.275)

    def test_sign_flip_respects_clustering(self):
        """One big cluster must not be shattered into many independent signs."""
        clustered = {"c0": [0.5] * 40}
        self.assertGreater(sign_flip_test(clustered, reps=500)["p_value"], 0.4)

    def test_bootstrap_brackets_the_mean(self):
        clusters = {f"c{i}": [0.4, 0.6] for i in range(30)}
        lo, hi = cluster_bootstrap_ci(clusters, reps=500)
        self.assertLessEqual(lo, 0.5)
        self.assertGreaterEqual(hi, 0.5)

    def test_design_effect_is_one_without_between_cluster_variance(self):
        rec = icc({f"c{i}": [1.0, 0.0] for i in range(20)})
        self.assertAlmostEqual(rec["design_effect"], 1.0)


class PairedContrastTests(unittest.TestCase):
    @staticmethod
    def rows(diff):
        out = []
        for i in range(40):
            for arm, value in (("original", 1.0), ("repaired", 1.0 - diff)):
                out.append({"question_id": f"q{i}", "cluster": f"c{i % 8}", "arm": arm,
                            "draw": 0, "correct": value, "is_numeric": True,
                            "redraw_applied": True, "key_rank": 1, "chosen_rank": 1})
        return out

    def test_a_zero_difference_is_reported_as_zero(self):
        rec = paired_contrast(self.rows(0.0), "original", "repaired", None, 400)
        self.assertAlmostEqual(rec["difference"], 0.0)
        self.assertGreater(rec["p_value"], 0.5)

    def test_a_real_difference_is_recovered_with_its_sign(self):
        rec = paired_contrast(self.rows(0.25), "original", "repaired", None, 400)
        self.assertAlmostEqual(rec["difference"], 0.25)
        self.assertLess(rec["p_value"], 0.05)

    def test_per_item_averages_over_draws(self):
        rows = [{"question_id": "q", "cluster": "c", "arm": "original", "draw": d,
                 "correct": float(d % 2), "is_numeric": True, "redraw_applied": True}
                for d in range(10)]
        scores, _ = per_item(rows, "original")
        self.assertAlmostEqual(scores["q"], 0.5)


class RuleSolverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not JSONL.exists():
            raise unittest.SkipTest("question file not present")
        cls.items, cls.records = conditions(n_items=40, draws=1)

    def test_the_committed_family_is_the_rank_family(self):
        """Only the rank solvers are version-controlled; the rest regenerate."""
        self.assertEqual(set(SOLVERS), set(RANK_FAMILY))
        self.assertEqual(len(FAMILIES["all"]), len(RANK_FAMILY) + 18)

    def test_a_rank_rule_selects_the_slot_holding_that_rank(self):
        rng = random.Random(0)
        for record in self.records:
            if record["slot_ranks"] is None:
                continue
            slot = SOLVERS["rule:rank-2"](record, rng)
            self.assertEqual(record["slot_ranks"][slot], 2)

    def test_every_solver_returns_a_valid_slot(self):
        rng = random.Random(1)
        for name, solver in FAMILIES["all"].items():
            for record in self.records[:40]:
                self.assertIn(solver(record, rng), range(N_OPTIONS), name)

    def test_avoid_rules_never_select_what_the_rule_selects(self):
        rng = random.Random(2)
        family = FAMILIES["all"]
        take, avoid = family["surface:longest-option"], family["avoid:longest-option"]
        for record in self.records[:40]:
            self.assertNotEqual(avoid(record, rng), take(record, random.Random(2)))


class ReportedProbeTests(unittest.TestCase):
    """Guard the experiment's headline numbers against the released results."""

    @classmethod
    def setUpClass(cls):
        if not (PROBE.exists() and RULES.exists()):
            raise unittest.SkipTest("probe results not generated")
        cls.probe = json.loads(PROBE.read_text())
        cls.rules = {m["model"]: m["subsets"]["numeric options"]
                     for m in json.loads(RULES.read_text())["models"]}

    def test_a_rank_rule_scores_exactly_the_key_rank_share(self):
        """The pipeline's end-to-end consistency check, exact by construction."""
        for name, share in (("rule:rank-0", 0.1238), ("rule:rank-1", 0.5143),
                            ("rule:rank-2", 0.2952), ("rule:rank-3", 0.0667)):
            got = self.rules[name]["arms"]["original"]["accuracy"]
            self.assertAlmostEqual(got, share, places=3, msg=name)

    def test_the_placebo_leaves_rank_only_solvers_untouched(self):
        for name in ("rule:rank-0", "rule:rank-1", "rule:rank-2", "rule:rank-3"):
            arms = self.rules[name]["arms"]
            self.assertAlmostEqual(arms["original"]["accuracy"], arms["placebo"]["accuracy"],
                                   places=9, msg=name)

    def test_the_positive_control_moves_and_the_negative_one_does_not(self):
        def effect(name):
            return next(c for c in self.rules[name]["contrasts"]
                        if c["contrast"] == "placebo - repaired")
        self.assertGreater(effect("rule:rank-1")["difference"], 0.20)
        self.assertLess(effect("rule:rank-1")["p_value"], 0.01)
        self.assertLess(abs(effect("rule:uniform-random")["difference"]), 0.02)
        self.assertGreater(effect("rule:uniform-random")["p_value"], 0.05)

    def test_the_panel_is_complete_and_every_model_saw_every_arm(self):
        self.assertEqual(len(self.probe["models"]), 11)
        for model in self.probe["models"]:
            self.assertEqual(set(model["subsets"]["numeric options"]["arms"]), set(ARMS))
            self.assertEqual(model["n_draws"], 20)

    def test_the_prespecified_trend_test_is_reported_as_null(self):
        trend = self.probe["scale_trend"]["placebo - repaired"]
        self.assertEqual(trend["n_models"], 11)
        self.assertGreater(trend["p_value"], 0.05)

    def test_no_model_moves_as_much_as_the_positive_control(self):
        control = next(c for c in self.rules["rule:rank-1"]["contrasts"]
                       if c["contrast"] == "placebo - repaired")["difference"]
        for model in self.probe["models"]:
            effect = next(c for c in model["subsets"]["numeric options"]["contrasts"]
                          if c["contrast"] == "placebo - repaired")["difference"]
            self.assertLess(abs(effect), control / 3, model["model"])


if __name__ == "__main__":
    unittest.main()
