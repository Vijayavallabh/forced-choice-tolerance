"""The rewritten option sets are single draws of one generator (``mcq_audit.redraw_row``) at fixed seeds:
regenerating them reproduces, option for option, the files every analysis reads. $U$ asks the generator
for a rank drawn uniformly, $P$ for the key's released rank; an item whose drawn rank cannot be reached
takes another under $U$, and is left out under $P$ when it has no reachable redraw."""
import json
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mcq_audit as ma  # noqa: E402

V15 = ROOT / "build" / "bixbench_numeric_q.jsonl"
V15_U = ROOT / "build" / "bixbench_numeric_repaired.jsonl"
V15_P = ROOT / "build" / "bixbench_numeric_placebo.jsonl"
V10 = ROOT / "build" / "bixbench_v10_items.jsonl"
V10_NUMERIC = ROOT / "build" / "bixbench_v10_numeric_released.jsonl"
V10_U = ROOT / "build" / "bixbench_v10_repaired.jsonl"
V10_P = ROOT / "build" / "bixbench_v10_numeric_placebo.jsonl"
U_SEED, P_SEED = 20260918, 20260920
pytestmark = pytest.mark.skipif(not all(p.exists() for p in (V15, V15_U, V15_P, V10, V10_NUMERIC, V10_U, V10_P)),
                                reason="option sets not built")


def load(path):
    return [json.loads(line) for line in path.open()]


def rewrite(path, seed, preserve_rank=False):
    rows = ma.read_rows(path)
    items, k = ma.build_items(rows, "ideal", "distractors", "capsule_uuid", "question", None)
    return ma.apply_repair(rows, items, "distractors", seed, k=k, preserve_rank=preserve_rank)


def test_v15_rank_uniform_rewrite_regenerates():
    rows, changed, infeasible, off_target, _ = rewrite(V15, U_SEED)
    assert [r["distractors"] for r in rows] == [r["distractors"] for r in load(V15_U)]
    assert (changed, len(infeasible), off_target) == (105, 0, 3)


def test_v15_rank_preserving_rewrite_regenerates():
    rows, changed, infeasible, off_target, _ = rewrite(V15, P_SEED, preserve_rank=True)
    assert [r["distractors"] for r in rows] == [r["distractors"] for r in load(V15_P)]
    assert (changed, len(infeasible), off_target) == (105, 0, 0)
    for row in rows:                                   # the key keeps its released rank
        released = next(r for r in load(V15) if r["question"] == row["question"] and r["ideal"] == row["ideal"])
        assert ma.rank_of([row["ideal"], *row["distractors"]]) == ma.rank_of([released["ideal"], *released["distractors"]])


def test_v10_rank_uniform_rewrite_regenerates():
    rows, changed, infeasible, off_target, _ = rewrite(V10, U_SEED)
    assert [r["distractors"] for r in rows] == [r["distractors"] for r in load(V10_U)]
    assert (changed, len(infeasible), off_target) == (159, 0, 10)


def test_v10_rank_preserving_rewrite_regenerates():
    rng, kept, missed = random.Random(P_SEED), [], 0
    for row in load(V10_NUMERIC):
        drawn = ma.placebo_row([row["ideal"], *row["distractors"]], rng)
        if drawn is None:
            missed += 1
        else:
            kept.append(drawn)
    assert kept == [[r["ideal"], *r["distractors"]] for r in load(V10_P)]
    assert (len(kept), missed) == (158, 1)


def test_the_generator_keeps_the_key_its_sign_and_the_style_of_what_it_replaces():
    rng = random.Random(0)
    options = ["0.25", "0.18", "0.31", "0.40"]
    for target in range(4):
        new = ma.redraw_row(options, rng, target=target)
        assert new[0] == "0.25" and ma.rank_of(new) == target
        assert all(float(v) > 0 and len(v.split(".")[1]) == 2 for v in new[1:])
        assert len(set(new)) == 4
