"""What the released options do to a score (score_decomposition.py), the may-decline reads and the
other readers of the registered runs (published_reads.py), the predictions table's margin-only mark
(replication.prediction_rows) and the ranking check against a noise-matched tolerance
(ranking_check.against_tolerance)."""
import hashlib
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import bixbench_withdata as bw  # noqa: E402
import published_reads as pr  # noqa: E402
import ranking_check as rc  # noqa: E402
import replication as rp  # noqa: E402
import score_decomposition as sd  # noqa: E402


@pytest.mark.parametrize("hit", [0.0, 1.0])
@pytest.mark.parametrize("named", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("near,nonum", [(0.0, 0.0), (0.5, 0.0), (1.0, 0.0), (0.0, 1.0)])
def test_a_runs_cells_add_up_and_the_excess_is_credits_less_refusals(hit, named, near, nonum):
    """Every run falls in the cells whole, and the reading's score minus the tolerance's is the misses it
    credits -- the key nearest the number, no number, or a number nearer another option -- less the hits it
    refuses. An answer with no number has no nearest option, so its credit is all in its own cell."""
    c = sd.cells(hit, named, near, nonum)
    assert sum(c.values()) == pytest.approx(1.0)
    credited = c["miss_credited_nearest"] + c["miss_credited_no_number"] + c["miss_credited_other"]
    assert named - hit == pytest.approx(credited - c["hit_refused"])
    if nonum:
        assert c["miss_credited_nearest"] == c["miss_credited_other"] == 0.0


def test_a_slip_of_scale_is_found_and_nothing_else_is():
    assert sd.rescaled_near(0.35, 35.0)                 # a fraction for a per cent
    assert sd.rescaled_near(2.0, 4.0)                   # a log2 value for the value
    assert sd.rescaled_near(-1.5, 1.5)                  # the sign dropped
    assert sd.rescaled_near(1200.0, 1.2)                # a thousand times over
    assert not sd.rescaled_near(0.5, 0.9)               # a miss is not a rescaling
    assert not sd.rescaled_near(3.0, 0.0)               # a key of zero has no relative band
    assert not sd.rescaled_near(None, 1.0)
    assert not sd.rescaled_near(1e300, 5.0)             # no overflow on the way


def test_three_way_agreement_counts_declines_apart_from_wrong_picks():
    """Two readings that agree on which runs name the key can disagree on the rest: one declines where the
    other names a distractor. The three-way kappa sees that; the two-way one does not."""
    pub = {"a": {"correct": True, "refused": False}, "b": {"correct": False, "refused": True},
           "c": {"correct": False, "refused": False}, "d": {"correct": False, "refused": True}}
    reads = lambda correct, refused: {"released": [{"correct": correct, "refused": refused}]}
    rows = [{"k": "a", "reads": reads(True, False)}, {"k": "b", "reads": reads(False, False)},
            {"k": "c", "reads": reads(False, True)}, {"k": "d", "reads": reads(False, True)},
            {"k": "e", "reads": {}}]                     # no answer: not read, left out
    pub["e"] = {"correct": False, "refused": True}
    got = pr.agreement3(rows, lambda r: pub[r["k"]])
    assert got["n_reads"] == 4
    assert got["agree"] == pytest.approx(50.0)           # a and d agree; b and c swap decline and distractor
    assert got["reader_by_published"]["other"]["declined"] == 1
    assert got["reader_by_published"]["declined"]["other"] == 1
    assert got["kappa"] < 0.5


def test_the_nearest_ranks_are_the_rules_and_ties_are_kept():
    options = ["2", "1", "3", "4"]                       # key first; ranks 1, 0, 2, 3
    assert sd.nearest_ranks("0.2", options) == {0}       # nearest the smallest value, the distractor 1
    assert sd.nearest_ranks("2.05", options) == {1}      # the key, second-smallest
    assert sd.nearest_ranks("no number", options) is None
    # a number the same distance from two options keeps both
    assert sd.nearest_ranks("0", ["1", "-1", "5", "7"]) == {0, 1, 2, 3}


def test_a_value_counts_as_written_at_the_precision_the_option_is_written_to():
    text = "log2FC = 2.2714, adj. p-value = 0.226; n = 12 cells; tiny = 3.1e-05"
    assert sd.values_written(text, ["2.27", "0.23", "12", "3.1e-05"]) == [True, True, True, True]
    assert sd.values_written(text, ["2.28", "0.3", "13", "3.2e-05"]) == [False, False, False, False]
    assert sd.values_written(text, ["not a number"]) == [False]


def test_the_decline_read_takes_the_forced_reads_seed_and_adds_the_option():
    """A forced read is seeded exactly as the paper's reads were (``q|shuffle|False``), so the cached
    replies are the same calls; the may-decline read adds BixBench's option to the same shuffle."""
    q, shuffle = "bix-1-q1", 1
    forced = int(hashlib.sha256(f"{q}|{shuffle}|{False}".encode()).hexdigest()[:8], 16)
    assert forced == int(hashlib.sha256(f"{q}|{shuffle}|False".encode()).hexdigest()[:8], 16)
    _, _, refusal_letter, shown = bw.questions_to_mcq("Q?", ["1", "2", "3", "4"], True, random.Random(forced))
    assert bw.REFUSE in shown and len(shown) == 5 and shown[ord(refusal_letter) - 65] == bw.REFUSE


def test_each_reader_and_mode_keeps_its_own_files():
    """gemma-3-27b's forced reads keep the paper's first paths; any other reader or mode has its own."""
    assert pr.work_dir() == pr.WORK and pr.rows_path() == pr.ROWS and pr.out_path() == pr.OUT
    others = {(pr.work_dir(r, m), pr.rows_path(r, m), pr.out_path(r, m))
              for r, m in (("gemma27b", "decline"), ("qwen72b", "forced"), ("llama70b", "forced"))}
    assert len(others) == 3 and all(pr.WORK not in t and pr.ROWS not in t and pr.OUT not in t for t in others)


def test_h2_passing_on_its_margin_alone_is_marked():
    """The registered rule passes kept keys whose interval excludes 0 when the mean is within 5 points;
    the table prints that pass as a failure, marked, as the paper counts it."""
    def group(mean, lo, hi, **kw):
        return {"mean": mean, "lo": lo, "hi": hi, **kw}

    def res(kept):
        v = {"H1": True, "H2": True, "H3": True, "H4": True, "H5": True}
        return {"gain|moved to an edge": group(19.8, 10.4, 31.2), "gain|kept": kept,
                "gain|moved inward": group(-14.1, -20.5, -8.2),
                "repaired-placebo|moved to an edge": group(18.6, 8.7, 30.7),
                "newly_credited": {"share_over_25pct": 0.9, "share_open_side": 0.96, "n_gained": 283},
                "verdicts": v}
    nod = {"gain|moved to an edge": group(22.2, 14.2, 30.9), "verdicts": {"H6": True}}
    report = {"D1": res(group(3.7, 0.8, 6.8)),
              "D2": {"reruns|data": res(group(2.2, -0.8, 6.3)), "reruns|nodata": nod,
                     "qwen3-235b|data": res(group(2.2, -0.8, 6.3)), "qwen3-235b|nodata": nod}}
    h2 = next(r for r in rp.prediction_rows(report) if r.startswith("H2: "))
    cells = h2.split(" & ")[2:]
    assert "\\textbf{fail}$^{\\S}$" in cells[0] and "$^{\\S}$" not in cells[1] and cells[1].rstrip("\\").endswith("pass")


def test_a_reading_equal_to_the_tolerance_loses_nothing_and_adds_no_noise():
    """Against the graders, a reading identical to the tolerance differs from it by exactly 0 on every
    capsule draw; the noise-matched tolerance of such a reading credits no miss and refuses no hit."""
    runs = rc.br.RUNS
    items = [f"q{i}" for i in range(12)]
    capsule = {q: f"c{i // 2}" for i, q in enumerate(items)}
    # seven run sets whose graders' order is strict, and a tolerance that keeps it exactly
    grades = {run: {q: float((i + j) % 7 <= j) for i, q in enumerate(items)} for j, run in enumerate(runs)}
    scores = {"graders": grades, "within 5%": grades, "nearest|released": grades}
    out = rc.against_tolerance(scores, capsule)["nearest|released"]
    assert out["difference"]["mean"] == 0 and out["difference"]["lo"] == 0 and out["difference"]["hi"] == 0
    assert out["miss_credit_rate"] == 0 and out["hit_refusal_rate"] == 0
    assert out["noise_matched"]["median"] == pytest.approx(out["tau"])


def test_following_the_nearest_option_is_counted_apart_on_misses():
    """A read counts toward the misses only where the number is beyond 5% of the key; a run with no
    number, or a read that names no option, counts nowhere."""
    sets = {"q": {"released": ["2", "1", "3", "4"]}}      # key 2 at rank 1; 1, 3 and 4 at ranks 0, 2, 3
    rows = [{"q": "q", "answer": "2.05", "reads": {"released": [{"rank": 1}, {"rank": 0}]}},   # a hit, the key nearest
            {"q": "q", "answer": "3.9", "reads": {"released": [{"rank": 3}, {"rank": 1}]}},    # a miss, 4 nearest
            {"q": "q", "answer": "none", "reads": {"released": [{"rank": 1}]}},
            {"q": "q", "answer": "0.1", "reads": {"released": [{"rank": None}]}}]
    out = pr.pick_is_nearest(rows, sets)
    assert out["all"] == {"share": 50.0, "n_reads": 4}
    assert out["miss"] == {"share": 50.0, "n_reads": 2}
    # the published reading's one pick per run goes through the same count
    one = pr.pick_is_nearest(rows[:2], sets, picks=lambda r: [3])
    assert one["all"] == {"share": 50.0, "n_reads": 2} and one["miss"] == {"share": 100.0, "n_reads": 1}


def _graded_run(q, answer, correct, rank):
    options = ["2", "3", "4", "6"]                       # the key, 2, is the smallest value
    return {"q": q, "answer": answer, "options": options,
            "reads": {"g forced": [{"correct": correct, "rank": rank, "refused": False}]}}


def test_misses_accepted_are_counted_over_the_misses_graded_and_set_beside_a_random_choice():
    """An empty answer is scored wrong without grading, so it is not a graded miss; the rest are split by what the
    answer gives, and the accepted misses in points are the share accepted times the graded misses' share."""
    runs = [_graded_run("a", "0.2", True, 0),           # a miss whose nearest option is the key, accepted
            _graded_run("b", "5", False, 0),            # a miss nearest 6, graded wrong
            _graded_run("c", "2", True, 0),             # correct
            _graded_run("d", "", False, None),          # empty: not graded
            _graded_run("e", "cannot say", True, 0)]    # no number, accepted
    hit = lambda r: float(bool(sd.graded(r["answer"], r["options"][0], sd.TOL))) if r["answer"] else 0.0
    m = sd.miss_rates(runs, "g forced", hit, {r["q"]: r["q"] for r in runs}, reps=200)
    assert m["all"]["accepted"] == pytest.approx(200 / 3) and m["all"]["share_of_runs"] == pytest.approx(60.0)
    assert m["nearest"]["accepted"] == 100.0 and m["other"]["accepted"] == 0.0
    assert m["neither"]["accepted"] == m["no number"]["accepted"] == 100.0
    assert m["chance"] == 25.0 and m["at_chance"] == pytest.approx(15.0)
    for r in runs:                                      # with a refusal option, a random choice is among five
        r["reads"]["g may-decline"] = r["reads"]["g forced"]
    m5 = sd.miss_rates(runs, "g may-decline", hit, {r["q"]: r["q"] for r in runs}, reps=200)
    assert m5["chance"] == 20.0 and m5["at_chance"] == pytest.approx(12.0)


def test_what_a_grading_selects_on_a_miss_splits_the_key_from_the_nearest_option():
    runs = [_graded_run("a", "0.2", True, 0),           # nearest is the key, and the key is selected
            _graded_run("b", "5", False, 0),            # nearest is 6; the key is selected
            _graded_run("c", "5", False, None),         # nearest is 6; nothing is selected
            _graded_run("d", "-3", False, 1)]           # no single nearest option: left out
    hit = lambda r: float(bool(sd.graded(r["answer"], r["options"][0], sd.TOL))) if r["answer"] else 0.0
    p = sd.picks_on_misses(runs, "g forced", hit)
    assert p["key"]["n_runs"] == 1 and p["key"]["nearest"] == 100.0
    assert p["other"]["n_runs"] == 2 and p["other"]["key"] == 50.0 and p["other"]["none"] == 50.0
    assert p["other"]["nearest_of_selecting"] == 0.0 and p["key"]["nearest_of_selecting"] == 100.0


def test_a_decline_is_told_from_a_pick_by_what_the_answer_is():
    import grader_declines as gd
    options = ["2", "3", "4", "6"]
    assert gd.kinds_of("2.01", options) == ("correct",)
    assert gd.kinds_of("0.2", options) == ("miss", "miss, nearest the key")
    assert gd.kinds_of("5.5", options) == ("miss", "miss, nearest another option")
    assert gd.kinds_of("no value", options) == ("miss", "miss, no number")
    assert gd.NO_MATCH.search("None of the provided answer options match this value")
