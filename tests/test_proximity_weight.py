"""Remark 1: a forced grader that, on a miss with a single nearest option, selects none with
probability delta, the nearest with probability q and each other option with probability (1-q)/(k-1)
bears lambda = (1-delta)(qk-1)/(k-1) of the nearest-option rule's gain from a change of options. The
weight is 1 for the rule, 0 for a grader that ignores proximity, and ``fit`` recovers it from reads."""
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import proximity_weight as pw  # noqa: E402

K = pw.K


def weight(q, delta):
    return (1 - delta) * (q * K - 1) / (K - 1)


def test_the_rule_bears_everything_and_a_blind_grader_nothing():
    assert weight(1.0, 0.0) == 1.0
    assert weight(1.0 / K, 0.0) == 0.0
    assert weight(1.0, 0.5) == 0.5


def test_zero_is_nearest_the_smallest_option():
    # every positive option is at relative distance 1 from zero; the absolute distance breaks the tie
    assert pw.nearest_ranks("0", ["30", "10", "20", "40"]) == {0}
    assert pw.single_nearest_rank("0", ["30", "10", "20", "40"]) == 0
    assert pw.rule("0", ["10", "20", "30", "40"]) == 1.0      # the key is the smallest option
    assert pw.rule("0", ["30", "10", "20", "40"]) == 0.0


def test_the_nearest_option_on_a_log_scale():
    assert pw.nearest_ranks("2.9", ["2", "1", "3", "4"]) == {2}
    assert pw.rule("25", ["20", "5", "10", "15"]) == 1.0       # beyond an extreme key, the key is nearest
    assert pw.rule("no number here", ["20", "5", "10", "15"]) == 0.0


def _simulated(q, delta, n=20000, seed=7):
    """Entries read by a synthetic grader of the proposition's form, through an option set whose key is
    extreme (the moved key) and one whose key is bracketed; every answer is a miss beyond the largest value."""
    rng = random.Random(seed)
    entries = []
    for i in range(n):
        extreme = ["100", "10", "20", "30"]           # key 100 is the largest: a miss of 500 is nearest it
        bracketed = ["100", "10", "150", "300"]       # a miss of 500 is nearest 300
        reads = {}
        for arm, options in (("repaired", extreme), ("placebo", bracketed)):
            near = pw.single_nearest_rank("500", options)
            if rng.random() < delta:
                rank = None
            elif rng.random() < q:
                rank = near
            else:
                rank = rng.choice([r for r in range(K) if r != near])
            key = 3 if arm == "repaired" else 1
            reads[arm] = [{"correct": rank == key, "rank": rank}]
        entries.append({"q": f"q{i % 40}", "run": "r", "answer": "500",
                        "sets": {"released": bracketed, "placebo": bracketed, "repaired": extreme},
                        "reads": reads, "group": "moved to an edge"})
    return entries


@pytest.mark.parametrize("q,delta", [(1.0, 0.0), (0.7, 0.1), (0.25, 0.0), (0.6, 0.5)])
def test_fit_recovers_the_weight_and_the_gain(q, delta):
    entries = _simulated(q, delta)
    f = pw.fit(entries)
    assert abs(f["lambda"] - weight(q, delta)) < 0.03
    c = pw.contrast(entries, f)
    assert abs(c["rule"] - 100.0) < 1e-9                       # the key becomes nearest on every run
    assert abs(c["observed"] - 100.0 * weight(q, delta)) < 3.0
    assert abs(c["predicted"] - 100.0 * f["lambda"]) < 1e-6
