"""Tests for the in-context-learning probe.

The value of that experiment rests entirely on its controls, so those are what
is tested here: that the target item is identical across arms, that
demonstrations never come from the target's own study, and that the repaired
demonstration pool differs from the original one in rank and in nothing else.
"""
import json
import unittest
from collections import Counter
from pathlib import Path

from icl_analysis import trend
from icl_probe import build_conditions
from no_data_probe import load_items
from option_artifacts import parse_number

RESULT = Path("results/icl_probe.json")


class ControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path("data/bixbench.jsonl")
        if not path.exists():
            raise unittest.SkipTest("question file not available")
        items, cls.k = load_items(path)
        cls.items = [it for it in items if it["rank"] is not None][:24]
        cls.arms = [(0, "none", "withheld"), (8, "original", "withheld"),
                    (8, "repaired", "withheld"), (8, "original", "shown")]
        cls.records = build_conditions(cls.items, cls.arms, 2, 20260920, cls.k)
        cls.by_key = {(r["question_id"], r["arm"], r["draw"]): r for r in cls.records}

    def test_every_item_appears_in_every_arm_and_draw(self):
        self.assertEqual(len(self.records), len(self.items) * len(self.arms) * 2)

    def test_the_target_is_byte_identical_across_demo_pools(self):
        """The zero-shot prompt is an exact suffix of each few-shot prompt."""
        for item in self.items:
            zero = self.by_key[(item["question_id"], "withheld_s0_none", 0)]
            for pool in ("original", "repaired"):
                few = self.by_key[(item["question_id"], f"withheld_s8_{pool}", 0)]
                self.assertTrue(few["prompt"].endswith(zero["prompt"]),
                                f"{item['question_id']} {pool}: target text differs")
                self.assertEqual(few["target_letter"], zero["target_letter"])
                self.assertEqual(few["slot_ranks"], zero["slot_ranks"])

    def test_withholding_the_stem_removes_the_question(self):
        shown = self.by_key[(self.items[0]["question_id"], "withheld_s8_original", 0)]
        stem = self.by_key[(self.items[0]["question_id"], "shown_s8_original", 0)]
        self.assertNotIn(self.items[0]["question"][:40], shown["prompt"])
        self.assertIn(self.items[0]["question"][:40], stem["prompt"])

    def test_demonstrations_never_come_from_the_targets_own_study(self):
        """A demonstration from the same capsule could teach the target's answer."""
        cluster_of = {it["question_id"]: it["cluster"] for it in self.items}
        for record in self.records:
            if not record["demo_ids"]:
                continue
            for demo_id in record["demo_ids"]:
                self.assertNotEqual(cluster_of[demo_id], record["cluster"],
                                    f"{record['question_id']}: demonstration from own study")
            self.assertNotIn(record["question_id"], record["demo_ids"])

    def test_the_two_demo_pools_show_the_same_example_items(self):
        """Only the values differ, so the contrast isolates rank from selection."""
        for item in self.items:
            for draw in range(2):
                a = self.by_key[(item["question_id"], "withheld_s8_original", draw)]
                b = self.by_key[(item["question_id"], "withheld_s8_repaired", draw)]
                self.assertEqual(a["demo_ids"], b["demo_ids"])

    def test_repaired_demonstrations_keep_the_letters_and_change_the_ranks(self):
        """The control differs in rank and not in letter, count, or format."""
        rank_counts = {"original": Counter(), "repaired": Counter()}
        for item in self.items:
            for draw in range(2):
                blocks = {}
                for pool in ("original", "repaired"):
                    record = self.by_key[(item["question_id"], f"withheld_s8_{pool}", draw)]
                    demos = record["prompt"].split("Extract the single letter")[0]
                    blocks[pool] = [b for b in demos.split("\n\n") if "<answer>" in b]
                self.assertEqual(len(blocks["original"]), len(blocks["repaired"]))
                for pool, parsed in blocks.items():
                    for block in parsed:
                        lines = block.strip().split("\n")
                        letter = lines[-1].split(">")[1][0]
                        values = [parse_number(ln.split(") ", 1)[1])
                                  for ln in lines[1:-1] if ") " in ln]
                        if any(v is None for v in values) or len(set(values)) < len(values):
                            continue
                        order = sorted(range(len(values)), key=lambda i: values[i])
                        rank_counts[pool][order.index("ABCDEFGHIJ".index(letter))] += 1
                # the keyed letters are the same examples in the same positions
                self.assertEqual([b.strip().split("\n")[-1] for b in blocks["original"]],
                                 [b.strip().split("\n")[-1] for b in blocks["repaired"]])
        # the original pool is interior-heavy; the repaired pool is not
        orig, rep = rank_counts["original"], rank_counts["repaired"]
        self.assertGreater(orig[1] / sum(orig.values()), 0.35)
        self.assertLess(max(rep.values()) / sum(rep.values()),
                        max(orig.values()) / sum(orig.values()))


class TrendTests(unittest.TestCase):
    def test_a_monotone_ladder_gets_the_smallest_attainable_p(self):
        records = {f"withheld_s{n}_{'none' if n == 0 else 'original'}": {"accuracy": a}
                   for n, a in zip([0, 8, 32, 64], [0.25, 0.30, 0.38, 0.44])}
        rec = trend(records, "withheld", [0, 8, 32, 64], "original")
        self.assertAlmostEqual(rec["rho"], 1.0)
        self.assertAlmostEqual(rec["p_value"], 1 / 24)

    def test_a_flat_ladder_is_not_significant(self):
        records = {f"withheld_s{n}_{'none' if n == 0 else 'original'}": {"accuracy": a}
                   for n, a in zip([0, 8, 32, 64], [0.25, 0.24, 0.26, 0.25])}
        self.assertGreater(trend(records, "withheld", [0, 8, 32, 64], "original")["p_value"],
                           0.4)


class ReportedResultTests(unittest.TestCase):
    """Guard the numbers the manuscript states, once the probe has been run."""

    @classmethod
    def setUpClass(cls):
        if not RESULT.exists():
            raise unittest.SkipTest("ICL probe not run")
        cls.doc = json.loads(RESULT.read_text())

    def test_every_model_reports_the_full_arm_grid(self):
        for rec in self.doc["models"]:
            arms = set(rec["arms"])
            for stem in ("withheld", "shown"):
                self.assertIn(f"{stem}_s0_none", arms)
                for n in (8, 32, 64):
                    self.assertIn(f"{stem}_s{n}_original", arms)
                    self.assertIn(f"{stem}_s{n}_repaired", arms)

    def test_the_zero_shot_stemless_arm_sits_near_chance(self):
        """Baseline sanity: with no question and no examples there is nothing to go on."""
        for rec in self.doc["models"]:
            chance = 1.0 / rec["n_options"]
            accuracy = rec["arms"]["withheld_s0_none"]["accuracy"]
            self.assertLess(abs(accuracy - chance), 0.12,
                            f"{rec['model']}: zero-shot stemless is {accuracy:.1%}")

    def test_the_causal_contrast_is_reported_for_every_model(self):
        for rec in self.doc["models"]:
            labels = {(c["question"], c["stem"], c["n_shots"]) for c in rec["contrasts"]}
            self.assertIn(("attributable to key ranks", "withheld", 64), labels)


if __name__ == "__main__":
    unittest.main()
