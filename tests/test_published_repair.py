"""The published-baseline and robustness analyses, tested where they could fail quietly.

Each test here covers a bug that was actually made and caught, or one the code
would not announce:

  * ``published_repair.declined`` read v1.0's literal ``empty`` as a letter and
    reported a decline rate of zero for an arm that had no decline option.
  * ``published_repair.capsule_v15`` replaced a question-text join that matched
    nothing, so every row became its own cluster and an item bootstrap wore a
    capsule bootstrap's name.
  * ``cell_split.numeric`` classified all 205 BixBench items as non-numeric,
    because BixBench's template puts an instruction *after* the options and a
    greedy match swallowed it.
  * ``cell_split.cell`` subset the control by its own option text rather than by
    item, which broke the pairing and widened one interval by twenty points.
  * ``option_membership.auc`` must give exactly 0.5 on a constant score, or a
    membership attack reports purchase it does not have.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cell_split
import option_membership as om
import published_repair as pr

ROOT = Path(__file__).resolve().parents[1]


# --- published_repair --------------------------------------------------------

def test_capsule_is_the_datasets_own():
    """A question maps to the capsule the released dataset gives it.

    The first version split the identifier at its last '-q', which groups
    "bix-12-q3" under "bix-12" -- but the dataset's own capsule_uuid is the
    unit its questions share data in, and the two groupings differ (56 prefixes
    against 59 capsules). The bootstrap resamples the dataset's capsules.
    """
    import json
    capsule = {}
    for line in open(ROOT / "data/bixbench.jsonl", encoding="utf-8"):
        row = json.loads(line)
        capsule[row["question_id"].lower()] = row["capsule_uuid"]
    for qid in ("bix-12-q3", "bix-1-q1"):
        assert pr.capsule_v15(qid) == capsule[qid]
        assert pr.capsule_v15(qid.upper()) == capsule[qid]
    with pytest.raises(SystemExit):
        pr.capsule_v15("not-a-question")


def test_v15_identifiers_group_into_capsules():
    """205 questions in the dataset's 59 capsules, not 205 singletons."""
    rows = pr.read(ROOT / "data/external/zero_shot_v15"
                   / "gpt-4o-grader-mcq-refusal-False.csv")
    caps = {pr.capsule_v15(r["uuid"]) for r in rows}
    assert len(rows) == 205
    assert len(caps) == 59, (
        f"{len(caps)} capsules for {len(rows)} rows; the released dataset has 59")


def test_declined_is_none_without_a_decline_option():
    """v1.0 writes the literal 'empty'; reading it as a letter gave a false 0%."""
    assert pr.declined([{"unsure": "empty", "predicted": "B", "correct": "True"}]) is None
    assert pr.declined([{"unsure": "", "predicted": "B", "correct": "True"}]) is None
    assert pr.declined([]) is None


def test_declined_counts_and_scores_only_what_was_answered():
    rows = [{"unsure": "E", "predicted": "E", "correct": "False"},
            {"unsure": "E", "predicted": "E", "correct": "False"},
            {"unsure": "E", "predicted": "A", "correct": "True"},
            {"unsure": "E", "predicted": "B", "correct": "False"}]
    got = pr.declined(rows)
    assert got["rate"] == pytest.approx(50.0)
    assert got["n_answered"] == 2
    assert got["accuracy_when_answered"] == pytest.approx(50.0)


def test_said_no_data_only_fires_on_a_declination():
    rows = [{"unsure": "", "predicted": "The p-value cannot be determined without "
                                        "the actual data."},
            {"unsure": "", "predicted": "0.05"}]
    assert pr.said_no_data(rows) == pytest.approx(50.0)


def test_paired_is_paired():
    """The paired statistic must difference on the same resample, not two."""
    left = [{"id": i, "correct": 1.0, "chance": 0.25, "mode": "",
             "cluster": f"c{i % 4}"} for i in range(40)]
    right = [{"id": i, "correct": 1.0, "chance": 0.25, "mode": "",
              "cluster": f"c{i % 4}"} for i in range(40)]
    got = pr.paired(left, right, bootstrap=200, seed=0)
    assert got["difference"] == pytest.approx(0.0)
    # identical arms differenced on one resample give exactly zero every draw
    assert got["interval"] == pytest.approx([0.0, 0.0])
    assert got["n_paired"] == 40


def test_published_repair_reproduces_its_shipped_numbers():
    if not (ROOT / "results/published_repair.json").exists():
        pytest.skip("published_repair results not present")
    shipped = json.loads((ROOT / "results/published_repair.json").read_text())
    out = subprocess.run(
        [sys.executable, "published_repair.py", "--bootstrap", "400",
         "--output", "/tmp/published_repair_test.json"],
        cwd=ROOT, capture_output=True, text=True, timeout=900)
    assert out.returncode == 0, out.stderr[-2000:]
    got = json.loads(Path("/tmp/published_repair_test.json").read_text())
    for release, models in shipped["releases"].items():
        for model, entry in models.items():
            for arm, cell in entry["arms"].items():
                # accuracies are counts, so they do not depend on the bootstrap
                assert got["releases"][release][model]["arms"][arm]["accuracy"] \
                    == pytest.approx(cell["accuracy"]), f"{release} {model} {arm}"
                assert got["releases"][release][model]["arms"][arm]["n"] == cell["n"]


# --- cell_split --------------------------------------------------------------

def test_numeric_survives_the_bixbench_template():
    """The instruction after the options must not break option parsing."""
    template = ("Extract the single letter answer.\n\nQuestion: [withheld]\n\n"
                "Options:\n(A) 1.5\n(B) 2.5\n(C) 3.5\n(D) 4.5\n"
                "IMPORTANT: You must only output a single letter answer.")
    assert cell_split.numeric(template) is True
    assert cell_split.values(template) == ["1.5", "2.5", "3.5", "4.5"]


def test_numeric_is_false_for_text_options():
    block = ("Question: [withheld]\n\nOptions:\n(A) p-adj < 0.05\n(B) raw p\n"
             "(C) log2FC\n(D) none")
    assert cell_split.numeric(block) is False


def test_numeric_handles_the_neutral_frame():
    block = "Question: [withheld]\n\nOptions:\n(A) 46.0\n(B) 51.0\n(C) 7.8E-05"
    assert cell_split.numeric(block) is True


def test_bixbench_split_is_105_and_100():
    import arm_intervals
    dump = arm_intervals.resolve(ROOT / "build/dumps/qwen14b_bixall_bixprompt_argmax.jsonl")
    if not dump.exists():
        pytest.skip("BixBench template dump not present")
    rows = arm_intervals.load(dump, "file")
    items = {r["item"]: cell_split.numeric(r["user"]) for r in rows}
    assert sum(items.values()) == 105
    assert len(items) - sum(items.values()) == 100


def test_control_is_subset_by_item_not_by_its_own_text():
    """Selecting the control by text breaks the pairing and widens the interval."""
    import arm_intervals
    dump = arm_intervals.resolve(ROOT / "build/dumps/qwen14b_bixall_neutral_notools.jsonl")
    if not dump.exists():
        pytest.skip("BixBench neutral dump not present")
    got = cell_split.cell(dump, cell_split.numeric, 400, 0.95)
    whole = cell_split.cell(dump, None, 400, 0.95)
    width = got["vs_control_interval"][1] - got["vs_control_interval"][0]
    whole_width = whole["vs_control_interval"][1] - whole["vs_control_interval"][0]
    # half the items, so wider -- but not by the twenty points the text-based
    # subset produced
    assert width < whole_width * 2.0, (
        f"the numeric subset's control interval is {width:.1f} points wide "
        f"against the whole file's {whole_width:.1f}; the control is probably "
        "not paired to the same items")


# --- option_membership -------------------------------------------------------

def test_auc_is_a_half_on_a_constant_score():
    assert om.auc([1.0] * 20, [1.0] * 20) == pytest.approx(0.5)


def test_auc_is_one_when_separable():
    assert om.auc([2.0, 3.0, 4.0], [0.0, 0.5, 1.0]) == pytest.approx(1.0)
    assert om.auc([0.0, 0.5, 1.0], [2.0, 3.0, 4.0]) == pytest.approx(0.0)


def test_option_block_and_rank():
    user = "Question: [withheld]\n\nOptions:\n(A) 5\n(B) 1\n(C) 9\n(D) 3"
    block = om.option_block(user)
    assert block.startswith("(A) 5")
    # sorted values are 1, 3, 5, 9 -> the key at (A)=5 has rank 3
    assert om.rank_of_key(block, "A") == 3
    assert om.rank_of_key(block, "B") == 1
    assert om.rank_of_key(block, "C") == 4


def test_rank_is_none_on_text_options():
    block = "(A) yes\n(B) no"
    assert om.rank_of_key(block, "A") is None


def test_membership_gaps_all_cover_zero():
    """The finding is a null; if any statistic clears, the paper is stale."""
    path = ROOT / "results/option_membership.json"
    if not path.exists():
        pytest.skip("membership results not present")
    report = json.loads(path.read_text())
    assert len(report["correct_minus_wrong"]) == 6
    for key, gap in report["correct_minus_wrong"].items():
        lo, hi = gap["interval"]
        assert lo <= 0 <= hi, f"{key} clears zero at {gap['difference']:+.3f}"


# --- frontier_calibrated -----------------------------------------------------

def test_frontier_nominal_reproduces_the_shipped_intervals():
    """The recomputation must use learned_probe's own bootstrap, not a second."""
    cal = ROOT / "results/frontier_calibrated.json"
    probe = ROOT / "results/learned_probe_frontier.json"
    if not (cal.exists() and probe.exists()):
        pytest.skip("frontier results not present")
    cal = json.loads(cal.read_text())
    probe = json.loads(probe.read_text())
    worst = probe["clean"]["every feature"]["max"]
    names = {"released": "released", "wrong-step": "wrong-step",
             "rank-uniform": "rank-uniform", "exchangeable": "exchangeable",
             "imitation": "imitation", "key-marginal": "key-marginal",
             "key-marginal-near": "key-marginal-near"}
    for arm in names:
        lo, hi = probe["files"][arm]["every feature"]["cv_ci95"]
        want = (100 * (lo - worst), 100 * (hi - worst))
        got = cal["rows"][arm]["nominal_95"]
        assert got[0] == pytest.approx(want[0], abs=0.05), arm
        assert got[1] == pytest.approx(want[1], abs=0.05), arm


def test_covering_interval_is_wider_than_nominal():
    """88.1% coverage at nominal 95% means the covering level is stricter."""
    path = ROOT / "results/frontier_calibrated.json"
    if not path.exists():
        pytest.skip("frontier calibration not present")
    cal = json.loads(path.read_text())
    assert cal["calibration"]["coverage_at_95"] < 0.95
    assert cal["calibration"]["level_for_95"] > 0.95
    for arm, row in cal["rows"].items():
        nominal = row["nominal_95"][1] - row["nominal_95"][0]
        covering = row["covering"][1] - row["covering"][0]
        assert covering > nominal, (
            f"{arm}'s covering interval is narrower than its nominal one, which "
            "would mean the bootstrap over-covers at 4.8 effective clusters")


def test_wrong_step_no_longer_clears_at_the_covering_level():
    path = ROOT / "results/frontier_calibrated.json"
    if not path.exists():
        pytest.skip("frontier calibration not present")
    cal = json.loads(path.read_text())
    assert "wrong-step" in cal["clears_nominal"]
    assert "wrong-step" not in cal["clears_covering"]
    assert sorted(cal["clears_covering"]) == ["exchangeable", "rank-uniform"]


# --- lambda_relaxed ----------------------------------------------------------

def test_lambda_ordering_survives_the_relaxation():
    path = ROOT / "results/lambda_relaxed.json"
    if not path.exists():
        pytest.skip("relaxed fits not present")
    report = json.loads(path.read_text())
    assert report["ordering_survives"]
    assert report["max_abs_shift_any_draw"] < 0.05
    # the largest lambda-hat must come from a model with rejecting draws: those
    # are the draws the relaxation is about
    assert any(b["n_rejected"] > 0 for b in report["models"].values())


def test_vectorised_auc_matches_the_loop_it_replaced():
    """The fast AUC must give the values the slow one did, ties included.

    The original averaged ranks by looping over every distinct value, which made
    the bootstrap take ten minutes; rankdata does it in one sort. Identical
    output is the whole point of the change, so it is asserted rather than
    assumed.
    """
    def slow(positive, negative):
        pos, neg = np.asarray(positive, float), np.asarray(negative, float)
        both = np.concatenate([pos, neg])
        order = np.argsort(both, kind="stable")
        ranks = np.empty(len(order), dtype=float)
        ranks[order] = np.arange(1, len(order) + 1)
        for value in np.unique(both):
            tie = both == value
            ranks[tie] = ranks[tie].mean()
        return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
                     / (len(pos) * len(neg)))

    rng = np.random.default_rng(0)
    for _ in range(20):
        n, m = int(rng.integers(3, 60)), int(rng.integers(3, 60))
        # integers, so ties are common -- the case the loop existed for
        pos = rng.integers(0, 8, size=n).astype(float)
        neg = rng.integers(0, 8, size=m).astype(float)
        assert om.auc(pos, neg) == pytest.approx(slow(pos, neg), abs=1e-12)


def test_auc_is_none_on_an_empty_side():
    """A bootstrap draw can miss one side entirely; that must not crash."""
    assert om.auc([], [1.0, 2.0]) is None
    assert om.auc([1.0], []) is None


def test_vectorised_gap_reproduces_the_shipped_point_estimates():
    """The refactor must not move the six numbers the paper quotes.

    It did, the first time: the vectorised version keyed strata off an array
    where a rollout with non-numeric options carries -1, and so folded those
    rollouts in as a stratum of their own. Every within-rank gap shifted by
    about a thousandth -- small enough to publish, which is why this compares
    against the shipped file rather than against a reimplementation.
    """
    import gzip

    shipped_path = ROOT / "results/option_membership.json"
    cache = ROOT / "results/agentic_dumps/option_membership_scores.json.gz"
    if not (shipped_path.exists() and cache.exists()):
        pytest.skip("membership results or score cache not present")
    shipped = json.loads(shipped_path.read_text())["correct_minus_wrong"]
    with gzip.open(cache, "rt") as fh:
        released = json.load(fh)["released"]

    hit = np.array([r["correct"] for r in released], dtype=bool)
    rank = np.array([r["key_rank"] if r["key_rank"] else -1 for r in released])
    for stat in ("loglik", "mink", "minkpp"):
        score = np.array([r[stat] for r in released], dtype=float)
        pooled = float(score[hit].mean() - score[~hit].mean())
        assert pooled == pytest.approx(
            shipped[f"{stat}:pooled"]["difference"], abs=1e-9), stat
        total, weighted = 0, 0.0
        for value in np.unique(rank):
            if value < 0:
                continue
            cell = rank == value
            good, bad = cell & hit, cell & ~hit
            if good.any() and bad.any():
                weighted += cell.sum() * (score[good].mean() - score[bad].mean())
                total += cell.sum()
        assert weighted / total == pytest.approx(
            shipped[f"{stat}:within key rank"]["difference"], abs=1e-9), stat


# --- the membership grid, and the placebo that reads it ----------------------


def _rows(scores, hits, ranks, clusters=None):
    return [{"cluster": (clusters[i] if clusters else i // 3),
             "item": i, "correct": float(h), "key_rank": r,
             "loglik": s, "mink": s, "minkpp": s}
            for i, (s, h, r) in enumerate(zip(scores, hits, ranks))]


def test_gaps_is_zero_when_the_score_carries_nothing():
    """A constant membership score must read as no gap, not as a small one."""
    rows = _rows([0.5] * 12, [1, 0] * 6, [1, 1, 2, 2] * 3)
    report = om.gaps(rows, 50, 0)
    for key, cell in report.items():
        assert cell["difference"] == pytest.approx(0.0), key


def test_gaps_finds_a_gap_it_is_given():
    """And a score that is higher on every correct rollout must read positive."""
    scores = [1.0 if i % 2 == 0 else 0.0 for i in range(12)]
    rows = _rows(scores, [1, 0] * 6, [1, 1, 2, 2] * 3)
    report = om.gaps(rows, 50, 0)
    assert report["loglik:pooled"]["difference"] == pytest.approx(1.0)
    assert report["loglik:within key rank"]["difference"] == pytest.approx(1.0)


def test_gaps_reports_a_cell_with_no_usable_stratum():
    """Not every arm has one, and numpy raised IndexError on the empty list.

    A stratum contributes only when it holds both a correct and an incorrect
    rollout. Give every rank stratum one kind and the stratified gap is
    undefined on every draw -- which a refusing arm can produce, and which
    should be reported rather than crashed on.
    """
    rows = _rows([1.0, 0.0] * 6, [1, 0] * 6, [1, 2] * 6)
    report = om.gaps(rows, 50, 0)
    assert report["loglik:within key rank"]["difference"] is None
    assert report["loglik:within key rank"]["interval"] is None
    assert om.show_gap("loglik:within key rank",
                       report["loglik:within key rank"]).endswith(
        "no stratum holding both a right and a wrong rollout")
    # the pooled cell is still defined, and must still be reported
    assert report["loglik:pooled"]["difference"] == pytest.approx(1.0)


def test_gaps_ignores_the_rankless_stratum():
    """A rollout whose options are not numbers has no rank and no stratum.

    Folding those in as a stratum of their own moved every within-rank estimate
    by about a thousandth, which is why the released arm is compared against
    the shipped file and not against a reimplementation.
    """
    rows = _rows([1.0, 0.0] * 4, [1, 0] * 4, [1, 1, 2, 2, None, None, 3, 3])
    report = om.gaps(rows, 50, 0)
    assert report["loglik:within key rank"]["difference"] == pytest.approx(1.0)


def test_the_grid_reproduces_the_shipped_single_model_arm():
    """The grid's 14B entry is the arm the paper quotes; it must not move."""
    grid_path = ROOT / "results/option_membership_grid.json"
    shipped_path = ROOT / "results/option_membership.json"
    if not (grid_path.exists() and shipped_path.exists()):
        pytest.skip("membership results not present")
    grid = json.loads(grid_path.read_text())
    entry = grid.get("Qwen/Qwen2.5-14B-Instruct")
    if entry is None:
        pytest.skip("the 14B is not in the grid")
    shipped = json.loads(shipped_path.read_text())["correct_minus_wrong"]
    for key, cell in shipped.items():
        assert entry["correct_minus_wrong"][key]["difference"] == pytest.approx(
            cell["difference"], abs=1e-12), key


def test_every_grid_entry_carries_its_placebo():
    """Without the control arm a positive gap reads as recognition.

    The statistic is also a typicality statistic and a solver does better on
    typical option blocks, so the same difference is computed on the clean
    control -- which we generated, so there is nothing in it to recognise.
    """
    for name in ("results/option_membership_grid.json",
                 "results/option_membership_grid_bix.json"):
        path = ROOT / name
        if not path.exists():
            pytest.skip(f"{name} not present")
        grid = json.loads(path.read_text())
        assert len(grid) == 13, f"{name} holds {len(grid)} models, not thirteen"
        missing = [m for m, r in grid.items()
                   if len(r.get("control_correct_minus_wrong", {})) != 6
                   or len(r.get("released_minus_control", {})) != 6]
        assert not missing, f"{name}: no control arm for {missing}"


def test_the_grid_merge_keeps_both_writers():
    """Thirteen of these run at once and the merge is read-modify-write.

    An earlier arm's entry was lost to a path bug while its dump sat complete
    on disk, and an unlocked merge loses entries the same silent way.
    """
    import multiprocessing
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "x_grid.json"
        out.write_text("{}")
        procs = [multiprocessing.Process(target=_merge_one, args=(out, name))
                 for name in ("a", "b", "c", "d")]
        for p in procs:
            p.start()
        for p in procs:
            p.join(60)
        assert sorted(json.loads(out.read_text())) == ["a", "b", "c", "d"]


def _merge_one(out, name):
    """One writer, using the same lock the sweep uses."""
    import os
    import random
    import time

    lock = out.with_suffix(".lock")
    for _ in range(600):
        try:
            handle = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            time.sleep(0.01 + 0.02 * random.random())
    else:
        raise AssertionError("never got the lock")
    try:
        merged = json.loads(out.read_text())
        time.sleep(0.05)
        merged[name] = {"model": name}
        out.write_text(json.dumps(merged))
    finally:
        os.close(handle)
        lock.unlink(missing_ok=True)


def test_membership_grid_reports_the_widest_cell_and_counts_clearings():
    import membership_grid as mg

    grid = {
        "Qwen/Qwen2.5-7B-Instruct": {
            "correct_minus_wrong": {
                "loglik:pooled": {"difference": 0.01, "interval": [-0.1, 0.2]},
                "loglik:within key rank": {"difference": 0.02,
                                           "interval": [0.01, 0.2]},
                "minkpp:pooled": {"difference": -0.30,
                                  "interval": [-0.5, -0.1]}},
            "control_correct_minus_wrong": {
                "loglik:within key rank": {"difference": 0.00,
                                           "interval": [-0.1, 0.1]}}}}
    row, = mg.summary(grid)
    assert row["widest"] == "minkpp:pooled"
    assert row["clearing"] == 2 and row["cells"] == 3
    assert row["control"]["clearing"] == 0


def test_gap_difference_survives_disjoint_cluster_labels():
    """The two arms share no clusters, and a paired bootstrap crashed on that.

    The released file's clusters are MMLU's subject names; the control's are
    ``c0``--``c59``, and it holds a different number of rollouts because it is
    a synthetic file of the same shape rather than a rewriting of the same
    items. The first version of this resampled a shared cluster set and died
    on ``need at least one array to concatenate``.
    """
    released = _rows([1.0, 0.0] * 6, [1, 0] * 6, [1, 1, 2, 2] * 3,
                     clusters=["ori_mmlu-algebra", "ori_mmlu-biology"] * 6)
    control = _rows([1.0, 0.0] * 5, [1, 0] * 5, [1, 1, 2, 2, 1] * 2,
                    clusters=["c0", "c1"] * 5)
    report = om.gap_difference(released, control, 40, 0)
    assert len(report) == 6
    # identical gaps on both arms, so the difference is zero and the interval
    # is an interval and not a crash
    assert report["loglik:pooled"]["difference"] == pytest.approx(0.0)
    lo, hi = report["loglik:pooled"]["interval"]
    assert lo <= 0 <= hi


def test_gap_difference_reports_a_real_difference():
    """A gap present on the released arm and absent on the control must show."""
    released = _rows([1.0, 0.0] * 6, [1, 0] * 6, [1, 1, 2, 2] * 3,
                     clusters=["s1", "s2"] * 6)
    control = _rows([0.5] * 12, [1, 0] * 6, [1, 1, 2, 2] * 3,
                    clusters=["c0", "c1"] * 6)
    report = om.gap_difference(released, control, 40, 0)
    assert report["loglik:pooled"]["difference"] == pytest.approx(1.0)


def test_option_block_stops_at_bixbenchs_trailing_instruction():
    """BixBench's template puts "IMPORTANT:" after the options.

    A greedy match to end-of-string took the instruction as part of the option
    block, so no line parsed as an option -- every rollout lost its key rank,
    and the membership score was computed over the options plus a paragraph of
    boilerplate identical on every row. MMLU-Pro's prompts end at the options,
    which is why this survived there.
    """
    user = ("Extract the single letter answer to the following question from "
            "the given options.\n\nQuestion: [withheld]\n\nOptions:\n"
            "(A) 0.0003\n(B) 0.0002\n(C) 7.820659E-05\n(D) 1.847038E-05\n"
            "IMPORTANT: You must only output a single letter answer in XML "
            "format.\n\n Example Output: <answer> X </answer>")
    block = om.option_block(user)
    assert block.splitlines() == ["(A) 0.0003", "(B) 0.0002",
                                  "(C) 7.820659E-05", "(D) 1.847038E-05"]
    assert "IMPORTANT" not in block
    # smallest is 1.847038E-05, so (D) is rank 1 and (A) rank 4
    assert om.rank_of_key(block, "D") == 1
    assert om.rank_of_key(block, "A") == 4


def test_logistic_says_so_when_no_rollout_has_a_rank():
    """It crashed on ``axis 1 is out of bounds`` instead."""
    assert om.logistic([], ["loglik", "rank_is_modal"]) is None


def test_a_grid_output_merges_whatever_its_name_suffix_is():
    """``option_membership_grid_bix.json`` is a grid too.

    The test was ``endswith("_grid.json")``, so the BixBench sweep's output
    failed it and each of thirteen runs overwrote the file with its own single
    report. Nothing complained: one report is a valid file, and the only sign
    was that the grid had one model in it.
    """
    import pathlib
    for name, merged in (("results/option_membership_grid.json", True),
                         ("results/option_membership_grid_bix.json", True),
                         ("results/option_membership_grid_bixall.json", True),
                         ("results/option_membership.json", False),
                         ("results/option_membership_bix.json", False)):
        assert ("_grid" in pathlib.Path(name).stem) is merged, name


def test_the_quoted_cell_differs_by_file_and_says_why():
    """BixBench's control carries a rank on 66 of 615 rollouts, not 615.

    Its control is resampled from the benchmark's own pool of option strings,
    so most control items mix numbers with text and have no rank. A stratified
    difference between a 315-rollout arm and a 66-rollout one is mostly the
    second arm's noise, so that file is read pooled -- named in one table
    rather than swapped in silently.
    """
    import membership_grid as mg

    assert mg.CELL["mmlupro"] == "loglik:within key rank"
    assert mg.CELL["bixall"] == "loglik:pooled"
    grid = {
        "m": {"correct_minus_wrong": {
                  "loglik:pooled": {"difference": 0.5, "interval": [0.4, 0.6]},
                  "loglik:within key rank": {"difference": 0.1,
                                             "interval": [-0.1, 0.3]}}}}
    pooled, = mg.summary(grid, refused={}, cell="loglik:pooled")
    stratified, = mg.summary(grid, refused={},
                             cell="loglik:within key rank")
    assert pooled["stratified"]["difference"] == 0.5
    assert stratified["stratified"]["difference"] == 0.1


def test_control_vocabulary_tells_the_two_files_apart():
    """What the released-minus-control difference isolates depends on the file.

    BixBench's control draws its options from the file's own pool, so every
    string in it was published and only the grouping was not: the difference
    holds string-level familiarity fixed and isolates recognition of the
    published set. MMLU-Pro's control redraws numeric values, so most of its
    strings are new and the difference conflates the two -- conservatively,
    since string-level recognition would inflate the released arm.
    """
    import membership_grid as mg

    import arm_intervals

    bix = arm_intervals.resolve(ROOT / "build/dumps/qwen14b_bixall_bixprompt_argmax.jsonl")
    pro = arm_intervals.resolve(ROOT / "build/dumps/qwen14b_mmlupro_neutral_notools.jsonl")
    if not (bix.exists() and pro.exists()):
        pytest.skip("rollout dumps not present")
    assert mg.control_vocabulary(bix)["published_share"] == 1.0
    assert mg.control_vocabulary(bix)["released_pool"] == 709
    assert mg.control_vocabulary(pro)["published_share"] < 0.2


def test_the_cache_slug_survives_an_underscore_in_the_tag():
    """``qwen1_5b`` has an underscore; the other twelve tags do not.

    The slug was the second underscore-separated field of the dump's stem, so
    the 1.5B's BixBench scores were filed under ``_5b`` and the sweep looked
    for a cache that was never going to be there -- while every other model
    worked, which is why it took a missing thirteenth row to notice.
    """
    assert om.file_slug("build/dumps/qwen1_5b_bixall_bixprompt_argmax.jsonl") \
        == "_bixall"
    assert om.file_slug("build/dumps/qwen14b_bixall_bixprompt_argmax.jsonl") \
        == "_bixall"
    assert om.file_slug("build/dumps/qwen1_5b_mmlupro_neutral_notools.jsonl") \
        == ""
    # an unrecognised file keeps its whole stem rather than guessing
    assert om.file_slug("build/dumps/qwen1_5b_newfile_cell.jsonl") \
        == "_qwen1_5b_newfile_cell"


def test_chance_follows_the_option_count():
    """BixBench's items have four options, so chance there is 25%, not 10%.

    The reporter marks a model as uninformative when it reads within two points
    of chance, because then its correct and its wrong rollouts are nearly the
    same draw. A hardcoded tenth would have called every BixBench row
    comfortably above chance.
    """
    import membership_grid as mg

    pro = mg.compliance("results/scale_grid.json", "agent/generated", 10)
    bix = mg.compliance("results/scale_grid_bixbench.json",
                        "bixbench/argmax", 4)
    if not (pro and bix):
        pytest.skip("scale grids not present")
    assert {v["chance"] for v in pro.values()} == {10.0}
    assert {v["chance"] for v in bix.values()} == {25.0}


def test_the_same_weights_are_matched_under_either_id():
    """Our own files give Llama-3.1-8B two ids.

    The BixBench sweep recorded ``meta-llama/Meta-Llama-3.1-8B-Instruct`` and
    the MMLU-Pro one ``meta-llama/Llama-3.1-8B-Instruct``. Keying the refusal
    and accuracy columns on the full id left that row blank in both, which
    reads as "not measured" rather than "not matched".
    """
    import membership_grid as mg

    assert mg.short("meta-llama/Meta-Llama-3.1-8B-Instruct") == "Llama-3.1-8B"
    assert mg.short("meta-llama/Llama-3.1-8B-Instruct") == "Llama-3.1-8B"
    assert mg.short("google/gemma-3-4b-it") == "gemma-3-4b"
    assert mg.short("microsoft/phi-4") == "phi-4"
    grid = ROOT / "results/option_membership_grid_bix.json"
    if not grid.exists():
        pytest.skip("BixBench membership grid not present")
    rows = mg.summary(json.loads(grid.read_text()),
                      refused=mg.compliance("results/scale_grid_bixbench.json",
                                            "bixbench/argmax", 4),
                      cell=mg.CELL["bixall"])
    blank = [r["model"] for r in rows if r["over_chance"] is None]
    assert not blank, f"no accuracy matched for {blank}"


def test_a_missing_scale_grid_is_loud_not_silent():
    """Without it every model reads as compliant and above chance.

    The refusal and accuracy columns come from the scale grid. An empty map
    does not read as "unknown" downstream: nothing is marked as refusing or at
    chance, and the informative subgroup grows from ten models to thirteen.
    Reproducing on a host that had the membership grids but not the scale
    grids printed a different headline count with nothing to say it had.
    """
    import membership_grid as mg

    with pytest.raises(FileNotFoundError, match="refusal and accuracy"):
        mg.compliance("results/no_such_grid.json", "agent/generated", 10)
    # and the opt-out still works, because a bare grid is a legitimate thing
    # to summarise as long as the reader is told
    grid = {"m": {"correct_minus_wrong": {
        "loglik:pooled": {"difference": 0.1, "interval": [-0.1, 0.3]}}}}
    row, = mg.summary(grid, refused={}, cell="loglik:pooled")
    assert row["refused"] is None and row["over_chance"] is None


def test_the_scale_grid_index_skips_dumps_that_are_not_arms(tmp_path, monkeypatch):
    """The prompt factorial's dumps ship beside the arms and are not arms.

    Their rows record a cell's ``pick``, not an ``answer``, and ``dump_index``
    read ``answer`` off every dump it globbed, so shipping them made
    ``scale_grid.py`` die with a KeyError before printing a row -- in the tree
    and in the bundle alike, and no test ran it.
    """
    import gzip

    import scale_grid

    shipped = tmp_path / "results" / "agentic_dumps"
    shipped.mkdir(parents=True)
    arm = [{"arm": "file", "cluster": "c", "k": 4, "gold": "A", "answer": a} for a in "AAB"]
    cell = [{"cell": "TTTTT", "arm": "file", "cluster": "c", "k": 4, "gold": "A", "pick": "A"}]
    for name, rows in (("qwen14b_bixall_neutral_notools", arm),
                       ("prompt_factorial_qwen14b_bixbench", cell)):
        with gzip.open(shipped / f"{name}.jsonl.gz", "wt") as fh:
            fh.write("".join(json.dumps(r) + "\n" for r in rows))
    monkeypatch.chdir(tmp_path)
    index = scale_grid.dump_index()
    assert list(index) == [(3, 2)]
    assert "prompt_factorial_" not in index[(3, 2)]


# --- the waiters the sweeps serialise on ------------------------------------


def test_every_sweep_waiter_anchors_its_pgrep_on_the_interpreter():
    """``pgrep -f agentic_probe.py`` matches the shell that launched one.

    Thirteen sweep scripts serialise on a card by waiting for the previous
    probe to exit. An unanchored pattern also matches the `bash -c` wrapper of
    an interactive session that ran one earlier, and run_agentic17.sh records
    two of those holding a waiter for five hours with no probe running. The
    directory part is optional on purpose: a sweep run with PY=python3 writes
    a bare interpreter, and a waiter that MISSES is worse than one that
    self-matches, because it starts while a job still holds the card.
    """
    import re

    pattern = re.compile(r'pgrep -f "([^"]+)"')
    unanchored = []
    for script in sorted(ROOT.glob("run_*.sh")):
        for found in pattern.findall(script.read_text()):
            if not found.startswith("^([^ ]*/)?python[0-9.]* "):
                unanchored.append(f"{script.name}: {found}")
    assert not unanchored, "unanchored pgrep waiters: " + "; ".join(unanchored)


def test_the_anchored_pattern_rejects_a_wrapper_and_keeps_the_probe():
    """The two halves of the fix, on real command lines."""
    import re

    anchored = re.compile(
        r"^([^ ]*/)?python[0-9.]* agentic_probe\.py .*--device cuda:7")
    # the shell that launched a probe, which the old pattern matched
    assert not anchored.match(
        "/bin/bash -c /opt/conda/bin/python agentic_probe.py --device cuda:7")
    # the probe itself, under an absolute and under a bare interpreter
    assert anchored.match(
        "/opt/conda/bin/python agentic_probe.py --items x --device cuda:7")
    assert anchored.match("python3 agentic_probe.py --items x --device cuda:7")


def test_marker_waiters_are_bounded_and_check_their_path():
    """`grep -q ... 2>/dev/null` hides a wrong path as well as a late file.

    Eight sweep scripts serialise by waiting for the previous one's log to say
    ^done. An empty argument, a mistyped directory or a producer that died all
    looked identical to "not yet", and the loop spun forever. wait_for() names
    the first two and bounds the third.
    """
    import re

    waiters = [p for p in sorted(ROOT.glob("*.sh"))
               if re.search(r'(while|until) .*grep -q "\^done', p.read_text())]
    assert len(waiters) >= 9, f"only {len(waiters)} scripts wait on a marker"
    for script in waiters:
        text = script.read_text()
        assert "WAIT_TIMEOUT" in text, f"{script.name}'s wait is unbounded"
        assert "does not exist, so" in text, \
            f"{script.name} does not check its marker's directory"
        # and no raw unbounded loop survives
        assert not re.search(r'^\s*(while|until) .*grep -q "\^done[^\n]*done\s*$',
                             text, re.MULTILINE), \
            f"{script.name} still has a raw one-line marker loop"


def test_no_sweep_double_prefixes_its_out_path():
    """agentic_probe.py does ``out = RESULTS / args.out``.

    A --out that already begins results/ lands in results/results/, where the
    grid collector does not look. One sweep did that and its arm went missing
    while the dump sat complete on disk.
    """
    import re

    offenders = []
    for script in sorted(ROOT.glob("run_*.sh")):
        for found in re.findall(r"--out\s+(\S+)", script.read_text()):
            if found.startswith("results/"):
                offenders.append(f"{script.name}: --out {found}")
    assert not offenders, "double-prefixed --out: " + "; ".join(offenders)


def test_the_bibliography_is_structurally_sound():
    """One entry carried another paper's title under its metadata.

    Rodriguez (2005) in EM:IP 24(2) 3-13 is the three-options meta-analysis;
    it was filed under a distractor-functioning title it does not have and
    cited for a claim it does not make. That is not catchable structurally --
    BIBLIOGRAPHY_AUDIT.md records the check against Crossref -- but the
    structural faults that travelled with it are.
    """
    import re

    text = (ROOT / "refs.bib").read_text()
    entries = re.findall(r"@(\w+)\s*\{([^,]+),(.*?)\n\}", text, re.DOTALL)
    assert len(entries) >= 40, f"only {len(entries)} entries parsed"
    problems = []
    for kind, key, body in entries:
        fields = dict(re.findall(r"(\w+)\s*=\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}",
                                 body))
        key = key.strip()
        if kind == "article" and "journal" not in fields:
            problems.append(f"{key}: @article with no journal")
        if kind == "book" and "journal" in fields:
            problems.append(f"{key}: @book with a journal")
        if fields.get("journal", "").strip() in {
                "Routledge", "Springer", "Wiley", "Elsevier", "MIT Press"}:
            problems.append(f"{key}: journal names a publisher")
        for required in ("author", "title", "year"):
            if required not in fields:
                problems.append(f"{key}: no {required}")
    assert not problems, "; ".join(problems)


def test_every_cited_key_exists_and_every_entry_is_cited():
    """A dangling \\citep renders as [?]; an uncited entry is dead weight."""
    import re

    bib = (ROOT / "refs.bib").read_text()
    tex = (ROOT / "main.tex").read_text()
    defined = {m.strip() for m in re.findall(r"@\w+\s*\{([^,]+),", bib)}
    cited = set()
    for group in re.findall(r"\\cite[tp]?\{([^}]*)\}", tex):
        cited.update(k.strip() for k in group.split(","))
    assert not (cited - defined), f"cited but undefined: {sorted(cited - defined)}"
    # The other direction is not a defect: \\bibliography prints only what is
    # cited, and refs.bib is shared with the archived draft and the survey
    # pipeline. An entry used by neither is genuinely orphaned, so that is what
    # is checked.
    #
    # That check reads absence of evidence, so it needs its corpus to be there:
    # on a host without archive/ the glob returns nothing, every archived key
    # looks unused, and the test accuses three real references of being dead
    # weight. Absence of the corpus is not absence of the citation, so the two
    # are told apart here rather than reported as the same thing.
    corpus = [path for path in (list(ROOT.glob("*.py"))
                                + list(ROOT.glob("archive/original/*.tex")))
              if path.is_file()]
    if not any(path.suffix == ".tex" for path in corpus):
        pytest.skip("archive/original/*.tex not present, so 'unused' is unreadable")
    orphans = []
    for key in sorted(defined - cited):
        if not any(key in path.read_text(errors="ignore") for path in corpus):
            orphans.append(key)
    assert not orphans, f"defined, uncited and unused anywhere: {orphans}"


def test_the_packager_ships_every_script_and_self_tests_the_bundle():
    """The bundle passed every "is this file here?" guard and still did not run.

    Unpacked, it failed two tests and skipped seven more: ``chain_free.sh`` was
    outside a ``run_*.sh`` glob though it produces a number the paper reports,
    the aligned frontier arms sat under build/ where validate_artifact.py could
    not reach them, and three tests read uncompressed dumps that ship gzipped.
    None of that is visible from this tree, where all of it is present. A skip
    is the dangerous half: it is not a failure, so the suite reads green while
    pinning nothing.
    """
    text = (ROOT / "package_submission.py").read_text()

    # every script, not every script that happens to be named run_*
    assert 'root.glob("*.sh")' in text, "the packager globs only some scripts"
    shipped_by_glob = {p.name for p in ROOT.glob("*.sh")}
    assert "chain_free.sh" in shipped_by_glob and len(shipped_by_glob) >= 25

    # the arms validate_artifact.py reads
    assert 'relative("build/mmlu_frontier_*_aligned.jsonl")' in text

    # and the bundle has to pass its own tests before it can be written
    assert "extractall" in text and "-m\", \"pytest" in text, \
        "the packager does not run the bundle's own tests"
    assert "refusing to package; the unpacked bundle" in text

    # An allowed skip is a claim that the bundle deliberately omits an input.
    # Two are: the superseded draft and a 7 MB upstream re-fetch. Anything
    # else added here is a file that should have shipped.
    allowed = text.split("ALLOWED_SKIPS = {", 1)[1].split("}", 1)[0]
    reasons = [line for line in allowed.splitlines()
               if line.strip().startswith('"')]
    assert len(reasons) == 2, f"the skip allow-list has grown: {reasons}"


def test_control_vocabulary_reads_the_copy_the_bundle_ships():
    """It opened build/dumps/ directly, so the 12.9% in sec:limits could not be
    re-derived from the artifact at all -- the arm_intervals lesson, learned
    once and not carried across to the two modules written after it."""
    import inspect

    import membership_grid as mg

    source = inspect.getsource(mg.control_vocabulary)
    assert "arm_intervals.resolve" in source, \
        "control_vocabulary reads the working copy, not the shipped one"
    assert "gzip.open" in source, "it cannot read a .gz dump"


def test_page_budget_reads_a_mid_page_heading_on_the_last_page_as_room(monkeypatch, capsys):
    """It said "over by 46 lines" for a paper whose main text ended on page 9.

    A References heading part way down a page is an overshoot only when that
    page is past the limit; on the last allowed page it is room to spare, and
    reading it the other way asks for lines to be cut from a paper that fits.
    """
    import page_budget as pb

    monkeypatch.setattr(pb, "boundary", lambda pdf: (None, None, 33))
    monkeypatch.setattr(pb, "page_lines", lambda pdf, before: 54)
    monkeypatch.setattr(sys, "argv", ["page_budget.py", "--limit", "9"])

    monkeypatch.setattr(pb, "overshoot", lambda pdf: (9, 46, ["x"] * 46))
    assert pb.main() == 0
    assert "within the limit, about 8 lines to spare" in capsys.readouterr().out

    monkeypatch.setattr(pb, "overshoot", lambda pdf: (10, 5, ["x"] * 5))
    assert pb.main() == 1
    assert "over by 5 lines" in capsys.readouterr().out


def test_the_validators_page_check_reads_a_mid_page_heading_too(monkeypatch):
    """The validator accepted only a References heading at the top of a page,
    so a paper with fourteen lines to spare on page 9 failed its own check."""
    import page_budget as pb

    monkeypatch.setattr(pb, "boundary", lambda pdf: (9, 10, 34))
    assert pb.main_text_end("x.pdf") == (9, 10)
    monkeypatch.setattr(pb, "boundary", lambda pdf: (None, None, 34))
    monkeypatch.setattr(pb, "overshoot", lambda pdf: (9, 41, ["x"] * 41))
    assert pb.main_text_end("x.pdf") == (9, 9)
    monkeypatch.setattr(pb, "overshoot", lambda pdf: (None, None, []))
    assert pb.main_text_end("x.pdf") == (None, None)


def test_frontier_difficulty_draws_reproduce_the_shipped_interval():
    """The difficulty column is re-read off frontier_validity's own draws.

    Its per-item answers were not kept, so the script was re-run to keep them;
    the covering interval is only the same statistic if the nominal one comes
    back to the digit from the kept answers.
    """
    items = ROOT / "results/frontier_validity_items.json"
    shipped = ROOT / "results/frontier_validity.json"
    if not (items.exists() and shipped.exists()):
        pytest.skip("frontier per-item answers not present")
    import frontier_calibrated as fc

    doc = json.loads(items.read_text())
    ref = json.loads(shipped.read_text())
    model = "Qwen/Qwen2.5-14B-Instruct"
    point, drawn = fc.difficulty_draws(doc, model, "key-marginal-near", 4000, 20260919)
    want = ref["models"][model]["key-marginal-near"]["vs_reference"]
    assert point == pytest.approx(want["points"], abs=1e-9)
    assert [drawn[int(0.025 * len(drawn))], drawn[int(0.975 * len(drawn))]] == \
        pytest.approx(want["ci95"], abs=1e-9)
    # and the per-item answers carry every model, arm and draw
    assert len(doc["clusters"]) == doc["n_items"] == 464
    assert {len(h) for h in doc["models"][model]["released"]["hits"]} == {4}


def test_claim_budget_calibration_reproduces_the_frontier_level():
    """The budget's covering levels come from the paper's own simulation.

    Rebuilt from frontier_calibrated.json's stored fit at the aligned file's
    twenty subject sizes, it has to give back the 88% coverage and the nominal
    98.7% that the frontier table is printed at, or the family level it reads
    for the repair-cost cells is some other shape's.
    """
    if not (ROOT / "results/frontier_validity_items.json").exists():
        pytest.skip("frontier per-item answers not present")
    import claim_budget as cb

    shape = cb.frontier_shape(source=str(ROOT / "results/frontier_validity_items.json"),
                              stored=str(ROOT / "results/frontier_calibrated.json"))
    sizes = shape[2]
    assert len(sizes) == 20 and sizes.sum() == 464
    assert sizes.sum() ** 2 / (sizes ** 2).sum() == pytest.approx(4.822, abs=1e-3)
    levels, cov95 = cb.parametric_levels(shape, None, [0.95], 3000, 2000,
                                         np.random.default_rng(2))
    assert 0.86 < cov95 < 0.90
    assert 0.980 <= levels[0.95] <= 0.9925


def test_claim_budget_family_is_the_abstracts_claims():
    """Ten claims, every interval the abstract and the introduction state as a finding and the two no-data
    margins Appendix C rests on; eight survive, and the two that do not are the ones the paper says do not.

    The family is post hoc -- chosen as what the abstract leads with, after the
    results were known -- and the paper says so; what this pins is that the
    family and the abstract have not drifted apart.
    """
    path = ROOT / "results/claim_budget.json"
    if not path.exists():
        pytest.skip("budget not present")
    doc = json.loads(path.read_text())
    assert [c["claim"] for c in doc["claims"]] == [
        "rank rule", "published, gpt-4o", "published, claude", "not the rank", "forced over tolerance",
        "key over other", "corrected below tolerance", "current agents", "bracketing",
        "readers of the published runs"]
    assert doc["k"] == 10 and doc["target_coverage"] == pytest.approx(1 - 0.05 / 10)
    assert doc["survivors"] == [c["claim"] for c in doc["claims"]
                                if c["claim"] not in ("published, claude", "current agents")]
    # a claim survives when every part clears: the current agents' fails on gpt-6-luna alone
    for c in doc["claims"]:
        assert c["survives"] == all(p["clears_family"] for p in c["parts"]), c["claim"]
    current = next(c for c in doc["claims"] if c["claim"] == "current agents")
    assert [p["part"] for p in current["parts"]] == ["gpt-5.1", "gpt-6-luna", "DeepSeek-V4-Pro"]
    assert [p["part"] for p in current["parts"] if not p["clears_family"]] == ["gpt-6-luna"]
    # the rank claim is an intersection-union test over the two published models: each
    # part's whole corrected interval must fall below what a rank reader would show
    rank = next(c for c in doc["claims"] if c["claim"] == "not the rank")
    assert [p["part"] for p in rank["parts"]] == ["gpt-4o", "claude-3-5-sonnet-latest"]
    assert all(p["ci_family"][1] < p["rank_reader_difference"] for p in rank["parts"])
    # so is the forced reading's excess over the tolerance, in both published models, and only the
    # rank claim's parts carry a rank reader's bound
    for name in ("forced over tolerance", "key over other"):
        claim = next(c for c in doc["claims"] if c["claim"] == name)
        assert [p["part"] for p in claim["parts"]] == ["gpt-4o", "Claude 3.5 Sonnet"]
        assert claim["survives"] and all(p["ci_family"][0] > 0 and "rank_reader_difference" not in p
                                         for p in claim["parts"])
    # so is the claim that three open readers of the published runs each inherit part of the rank's credit
    readers = next(c for c in doc["claims"] if c["claim"] == "readers of the published runs")
    assert [p["part"] for p in readers["parts"]] == ["Qwen2.5-72B", "gemma-3-27b", "Llama-3.3-70B"]
    assert readers["survives"] == all(p["ci_family"][0] > 0 for p in readers["parts"])
    # the paired claims are calibrated on their capsules' own paired gains, the rest by simulation
    parametric = ("rank rule", "published, gpt-4o", "published, claude", "not the rank")
    for c in doc["claims"]:
        want = "parametric" if c["claim"] in parametric else "double bootstrap"
        assert all(p["calibration"] == want for p in c["parts"]), c["claim"]
    # every part counts its capsules, a joint ratio's included
    assert all(p["n_clusters"] > 20 for c in doc["claims"] for p in c["parts"])
    # and no claim was read at a level below the one its shape needs for 95%
    for c in doc["claims"]:
        for p in c["parts"]:
            assert p["level_covering_target"] > p["level_covering_95"] > 0.95


def test_a_rank_reader_at_chance_has_no_rank_preference():
    """rank_attribution inverts accuracy = p2*b + (1-p2)*(1-b)/3 for the rate b at which a
    reader drawing its accuracy from a second-smallest preference picks that rank."""
    import rank_attribution as ra
    shares = [0.10, 0.514, 0.20, 0.186]
    at_chance = ra.rank_reader_prediction(shares, 0.25)
    assert at_chance["b_second_smallest"] == pytest.approx(0.25)
    assert at_chance["difference"] == pytest.approx(0.0, abs=1e-9)
    # a reader that always takes the second-smallest option scores p2
    always = ra.rank_reader_prediction(shares, shares[1])
    assert always["b_second_smallest"] == pytest.approx(1.0)
    assert always["accuracy_on_second_smallest"] == pytest.approx(100.0)
    assert always["accuracy_elsewhere"] == pytest.approx(0.0)
    # and the prediction is clipped to a probability
    assert ra.rank_reader_prediction(shares, 0.9)["b_second_smallest"] == 1.0


def test_published_runs_are_no_more_accurate_on_second_smallest_keys():
    """The paper's rank test: each released forced run's accuracy where the key is
    second-smallest minus elsewhere sits far below what a rank reader of the same accuracy
    would show, and does not clear zero."""
    path = ROOT / "results/rank_attribution.json"
    if not path.exists():
        pytest.skip("rank attribution not present")
    doc = json.loads(path.read_text())
    for model in ("gpt-4o", "claude-3-5-sonnet-latest"):
        entry = doc["published"][f"v1.5|{model}"]
        diff = entry["second_smallest_minus_rest"]
        assert entry["n_numeric"] == 105
        assert diff["mean"] < 0 and diff["lo"] < 0 < diff["hi"]
        assert diff["hi"] < entry["rank_reader_would_score"]["difference"]
    withheld = doc["withheld_summary"]
    assert withheld["n_models"] == 13
    assert withheld["max_second_smallest_pick_share"] < 51.4
