"""The capsule sign-flip test (randomization.py), the registered predictions' table
(replication.prediction_rows), the claim family's table (claim_budget.table_rows), the
release history of v1.5's keys (leak_origin.py), the ranking check (ranking_check.py) and the
reader of the published runs (published_reads.py)."""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import claim_budget as cb  # noqa: E402
import leak_origin as lo  # noqa: E402
import published_reads as pr  # noqa: E402
import randomization as rz  # noqa: E402
import ranking_check as rc  # noqa: E402
import replication as rp  # noqa: E402


def test_signflip_is_exact_over_every_pattern():
    """With few capsules every sign pattern is enumerated, so the p-value is a count over them: three
    capsules of sums 1, 2 and 3 give |T| = 6, reached only by all-plus and all-minus, 2 of 8."""
    per_item = {"a": 1.0, "b": 2.0, "c": 3.0}
    capsule = {"a": 0, "b": 1, "c": 2}
    r = rz.signflip(per_item, capsule)
    assert r["exact"] and r["n_clusters"] == 3
    assert r["p"] == pytest.approx(2 / 8)
    assert r["mean"] == pytest.approx(200.0)
    # no pattern count of three capsules reaches below 5%, so no shift is rejected
    assert r["lo"] is None and r["hi"] is None


def test_signflip_interval_inverts_the_test():
    """The interval is the set of shifts the test does not reject: at its ends the p-value crosses 5%."""
    rng = np.random.default_rng(3)
    per_item = {i: float(v) for i, v in enumerate(rng.choice((-1.0, 0.0, 1.0), p=(0.1, 0.6, 0.3), size=30))}
    capsule = {i: i // 2 for i in per_item}
    r = rz.signflip(per_item, capsule)
    sums, counts = rz._clusters(per_item, capsule)
    eps = rz._patterns(len(sums), 20260925)
    assert r["lo"] <= r["mean"] <= r["hi"]
    inside = [rz._p(sums, counts, d / 100, eps) for d in (r["lo"] + 0.01, r["hi"] - 0.01)]
    outside = [rz._p(sums, counts, d / 100, eps) for d in (r["lo"] - 1.0, r["hi"] + 1.0)]
    assert all(p > 0.05 for p in inside) and all(p <= 0.05 for p in outside)


def test_signflip_interval_is_bounded_from_six_capsules():
    """Six capsules give a smallest p-value of 2/64, below 5%, so both ends are found."""
    per_item = {i: float(i % 3 == 0) for i in range(12)}
    r = rz.signflip(per_item, {i: i // 2 for i in per_item})
    assert r["n_clusters"] == 6 and r["lo"] is not None and r["lo"] <= r["mean"] <= r["hi"]


def test_signflip_enumerates_up_to_twenty_capsules_and_samples_beyond():
    assert rz._patterns(3, 1).shape == (8, 3)
    assert len({tuple(row) for row in rz._patterns(3, 1)}) == 8
    assert rz._patterns(21, 1).shape == (rz.DRAWS, 21)


def _res(kept, verdicts, capped=False):
    iv = lambda m, lo, hi: {"mean": m, "lo": lo, "hi": hi, "level": 0.97, "level_capped": False}
    return {"gain|moved to an edge": iv(10.0, 2.0, 18.0), "gain|kept": iv(*kept),
            "gain|moved inward": dict(iv(-5.0, -9.0, -1.0), level_capped=capped),
            "repaired-placebo|moved to an edge": iv(9.0, 1.0, 17.0),
            "newly_credited": {"n_gained": 10, "share_over_25pct": 0.9, "share_open_side": 1.0},
            "verdicts": verdicts}


def test_prediction_rows_state_each_verdict_and_the_equivalence_reading():
    """Every prediction gets a row with each data set's estimate and verdict; a capped interval is not
    printed; a pass by H2's margin alone is printed as a marked failure; and H2 is read again as equivalence,
    which a kept-key interval reaching past 5 fails."""
    ok = {h: True for h in ("H1", "H2", "H3", "H4", "H5")}
    report = {"D1": _res((3.7, 0.8, 6.8), ok),
              "D2": {"reruns|data": _res((1.0, -2.0, 4.0), ok),
                     "qwen3-235b|data": _res((7.4, 1.4, 14.9), dict(ok, H2=False, H4=False), capped=True),
                     "reruns|nodata": {"gain|moved to an edge": {"mean": 20.0, "lo": 10.0, "hi": 30.0},
                                       "verdicts": {"H6": True}},
                     "qwen3-235b|nodata": {"gain|moved to an edge": {"mean": 21.0, "lo": 9.0, "hi": 33.0},
                                           "verdicts": {"H6": True}}}}
    rows = rp.prediction_rows(report)
    assert len(rows) == len(rp.PREDICTIONS) + 1
    # H2 on D1 passes the plan's rule only by its 5-point clause while its interval excludes 0: printed as a
    # marked failure, beside Qwen3-235B-A22B's plain one
    assert rows[1].count("\\textbf{fail}$^{\\S}$") == 1 and rows[1].count("\\textbf{fail}") == 2
    assert rows[3].count("\\textbf{fail}") == 1
    assert "$-5.0$$^{\\ddagger}$ pass" in rows[4]                     # the capped inward interval
    assert rows[5].startswith("H6") and " -- " in rows[5]              # D1 registered no H6
    eq = rows[-1]
    assert eq.count("not shown") == 2 and "& shown &" in eq           # the new seeds' interval sits inside
    assert "not shown (upper end $+6.8$)" in eq and "not shown (upper end $+14.9$)" in eq


def test_claims_table_marks_what_does_not_clear():
    """Each row prints the interval at the level that attains 95% coverage, as the paper states it elsewhere,
    beside the family's, and marks a part that does not clear at the family level; the rank claim's reference is
    what a rank reader would show."""
    report = {"claims": [
        {"claim": "rank rule", "parts": [{"part": "rank rule", "estimate": 26.43, "ci_family": [9.9, 42.5],
                                          "ci_quoted_95": [14.3, 38.3], "clears_family": True}]},
        {"claim": "corrected below tolerance", "parts": [
            {"part": "corrected below tolerance", "estimate": 5.79, "ci_family": [-0.99, 12.55],
             "ci_quoted_95": [1.34, 10.1], "clears_family": False}]},
        {"claim": "not the rank", "parts": [
            {"part": "gpt-4o", "estimate": -7.84, "ci_family": [-34.5, 20.7], "ci_quoted_95": [-26.4, 11.1],
             "clears_family": True, "rank_reader_difference": 45.946},
            {"part": "claude-3-5-sonnet-latest", "estimate": -13.4, "ci_family": [-40.1, 16.2],
             "ci_quoted_95": [-32.8, 7.9], "clears_family": True, "rank_reader_difference": 44.1}]}]}
    rows = cb.table_rows(report)
    assert len(rows) == 4
    assert rows[0].endswith("$[+14.3,+38.3]$ & $[+9.9,+42.5]$\\\\")
    assert rows[1].endswith("$[+1.3,+10.1]$ & $[-1.0,+12.6]$$^{\\times}$\\\\")
    assert "below $+45.9$" in rows[2] and "below $+44.1$" in rows[3]


def test_leak_origin_parts_add_up():
    """Every v1.5 numeric item is kept from v1.0 or new, and every v1.0 numeric item kept or dropped."""
    report = json.loads((ROOT / "results" / "leak_origin.json").read_text())
    c = report["counts"]
    assert c["carried over"] + c["new"] == c["v1.5 numeric"] == 105
    assert c["carried over"] + c["v1.0 dropped"] == c["v1.0 numeric"] == 159
    assert c["kept distractors"] + c["rewritten"] == c["carried over"]
    for part in report["parts"].values():
        assert sum(part["rank_shares"]) == pytest.approx(100.0)
    assert len(lo.table_rows(report)) == len(lo.ROWS)


def test_kendall_counts_concordant_pairs():
    assert rc.kendall([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert rc.kendall([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)
    # one swapped pair of six: (5 - 1) / 6
    assert rc.kendall([1, 2, 3, 4], [1, 3, 2, 4]) == pytest.approx(4 / 6)


def test_published_reads_score_as_upstream():
    """A run with no answer is wrong without being read; a run read twice scores its share of shuffles
    that name the key; an option set it was not read through gives no score."""
    blank = {"answer": "", "reads": {}}
    read = {"answer": "3.1", "reads": {"released": [{"correct": True}, {"correct": False}]}}
    assert pr.score(blank, "released") == 0.0
    assert pr.score(read, "released") == pytest.approx(0.5)
    assert pr.score(read, "placebo") is None


def test_published_reads_shuffle_is_read_mcq_s():
    """The reader shows the options in the order bixbench_withdata.read_mcq would, so a published run
    and a new one on the same question see the same letters."""
    import hashlib
    import random
    import bixbench_withdata as bw
    q = "bix-1-q1"
    seed = int(hashlib.sha256(f"{q}|0|False".encode()).hexdigest()[:8], 16)
    a = bw.questions_to_mcq("Q?", ["1", "2", "3", "4"], False, random.Random(seed))
    b = bw.questions_to_mcq("Q?", ["1", "2", "3", "4"], False, random.Random(seed))
    assert a == b
