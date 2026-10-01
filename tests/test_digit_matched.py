"""The digit-matched rewrites ($P'$, $U'$) are single draws of the rewrites' generator at the same seeds, with
each new distractor written to exactly the significant digits of the released distractor it replaces:
regenerating them reproduces the files, their distractors' digit counts are the released ones, and $P'$ keeps
each key's released rank. An item with no such draw is left out."""
import json
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import digit_matched as dm  # noqa: E402
import mcq_audit as ma  # noqa: E402

FILES = (dm.V15, dm.V15_PD, dm.V15_UD, dm.V10_ITEMS, dm.V10_NUMERIC, dm.V10_PD, dm.V10_UD)
pytestmark = pytest.mark.skipif(not all(p.exists() for p in FILES), reason="option sets not built")


def load(path):
    return [json.loads(line) for line in path.open()]


def placebo(path):
    rng, kept = random.Random(dm.P_SEED), []
    for row in load(path):
        drawn = ma.placebo_row([row["ideal"], *row["distractors"]], rng, match_digits=True)
        if drawn is not None:
            kept.append({**row, "distractors": drawn[1:]})
    return kept


@pytest.mark.parametrize("source,dest,n", [(dm.V15, dm.V15_UD, 103), (dm.V10_ITEMS, dm.V10_UD, 296)])
def test_rank_uniform_digit_matched_regenerates(source, dest, n):
    rows, _, infeasible, _, _ = dm.rewrite(source, dm.U_SEED)
    kept = [r for r in rows if [r["ideal"], *r["distractors"]] not in infeasible]
    assert [r["distractors"] for r in kept] == [r["distractors"] for r in load(dest)]
    assert len(kept) == n


@pytest.mark.parametrize("source,dest,n", [(dm.V15, dm.V15_PD, 99), (dm.V10_NUMERIC, dm.V10_PD, 157)])
def test_rank_preserving_digit_matched_regenerates_and_keeps_the_rank(source, dest, n):
    kept = placebo(source)
    assert [r["distractors"] for r in kept] == [r["distractors"] for r in load(dest)]
    assert len(kept) == n
    released = {(r["question"], r["ideal"]): r["distractors"] for r in load(source)}
    for row in kept:
        assert (ma.rank_of([row["ideal"], *row["distractors"]])
                == ma.rank_of([row["ideal"], *released[(row["question"], row["ideal"])]]))


@pytest.mark.parametrize("source,dest", [(dm.V15, dm.V15_UD), (dm.V15, dm.V15_PD), (dm.V10_NUMERIC, dm.V10_PD)])
def test_distractors_keep_the_released_digit_counts(source, dest):
    released = {(r["question"], r["ideal"]): r["distractors"] for r in load(source)}
    for row in load(dest):
        old = released[(row["question"], row["ideal"])]
        assert (sorted(ma.significant_digits(x) for x in row["distractors"])
                == sorted(ma.significant_digits(x) for x in old))


def test_fewest_digits_rule_is_at_the_released_rate():
    sets = dm.v15_sets(dm.V15_PD, dm.V15_UD)
    report = json.loads(dm.OUT.read_text())["v1.5"]["digit-matched"]["option_sets"]
    for arm in ("released", "placebo", "repaired"):
        present = [s[arm] for s in sets.values() if arm in s]
        credit = 100.0 * sum(dm.fewest_digits_credit(o) for o in present) / len(present)
        assert abs(credit - report[arm]["fewest_digits"]) < 1e-9
        assert abs(credit - 25.0) < 1.0
