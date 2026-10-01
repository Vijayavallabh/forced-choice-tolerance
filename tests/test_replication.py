"""The registered replication (replication.py), the tolerance checks (tolerance_check.py)
and the writer panel on BixBench (writer_panel.py)."""
import hashlib
import json
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import bixbench_withdata as bw  # noqa: E402
import replication as rp  # noqa: E402
import tolerance_check as tc  # noqa: E402
import writer_panel as wp  # noqa: E402


def test_key_classes_and_groups():
    """Bracketed means neither smallest nor largest; the group is fixed by the two sets alone."""
    assert rp.key_class(["2", "1", "3", "4"]) == "bracketed"
    assert rp.key_class(["4", "1", "3", "2"]) == "edge"
    assert rp.key_class(["1", "1", "3", "2"]) is None            # a tie has no rank
    moved = {"released": ["2", "1", "3", "4"], "repaired": ["2", "3", "4", "5"]}
    inward = {"released": ["1", "2", "3", "4"], "repaired": ["1", "0.5", "3", "4"]}
    kept = {"released": ["2", "1", "3", "4"], "repaired": ["2", "1.5", "2.5", "9"]}
    assert rp.group_of(moved) == "moved to an edge"
    assert rp.group_of(inward) == "moved inward"
    assert rp.group_of(kept) == "kept"
    assert rp.group_of({"released": ["2", "1", "3", "4"]}) is None


def test_gains_average_replicas_within_a_run_set_then_run_sets():
    """An item's gain is its repaired hit rate minus its released one per run set, then averaged
    over the run sets that answered it, so ten replicas count once."""
    sets = {"q": {"released": ["10", "9", "11", "12"], "repaired": ["10", "11", "12", "13"]}}
    # 9.4 is nearest 9 through the released set and nearest the key (10) through the repair
    runs = {"a": [("q", "9.4")] * 10, "b": [("q", "10"), ("q", "12.2")]}
    per_run, pooled = rp.gains(runs, sets)
    assert per_run["a"]["q"] == pytest.approx(1.0)
    assert per_run["b"]["q"] == pytest.approx(0.0)
    assert pooled["q"] == pytest.approx(0.5)


def test_newly_credited_is_counted_as_bracketing_counts_it():
    """``after > before`` on moved keys, binned by distance from the key and by open side."""
    sets = {"q": {"released": ["10", "9", "11", "12"], "repaired": ["10", "11", "12", "13"]}}
    runs = {"a": [("q", "9.4"), ("q", "3"), ("q", "10.2"), ("q", "no number")]}
    out = rp.newly_credited(runs, sets, {"q"})
    assert out["n_moved"] == 4 and out["n_with_a_number"] == 3
    # 9.4 (6% off, below the smallest repaired option: open side) and 3 (70% off) are newly
    # credited; 10.2 is nearest the key through both sets
    assert out["n_gained"] == 2
    assert out["gained"] == {"within 5%": 0, "5 to 25%": 1, "25 to 100%": 1, "over 100%": 0}
    assert out["gained_open_side"] == 2
    assert out["share_over_25pct"] == pytest.approx(0.5)


def test_packed_answers_read_back_as_the_runs_they_came_from(tmp_path, monkeypatch):
    """The shipped copies keep only what the analysis reads; a text-protocol run, which records
    no protocol, comes back as the same run and not as a run of its own."""
    made = [{"model": "qwen72b", "condition": "data", "question_id": "q", "rollout": 1, "answer": "3",
             "termination": "submitted", "steps": 4, "settings": {"work": "/scratch"}},
            {"model": "qwen72b", "protocol": "react", "condition": "nodata", "question_id": "q",
             "rollout": 1, "answer": "4", "termination": "submitted", "steps": 2}]
    read = rp.d2_trajectories
    monkeypatch.setattr(rp, "PACKED", tmp_path / "packed")
    monkeypatch.setattr(rp, "d2_trajectories", lambda root=None, packed="answers": made if packed else [])
    rp.pack()
    back = read(root=tmp_path / "none")          # nothing where runs are made: the shipped copies
    assert sorted(bw.run_key(t) for t in back) == ["qwen72b", "qwen72b-react"]
    assert all("settings" not in t for t in back)
    monkeypatch.setattr(rp, "d2_trajectories", lambda root=None, packed="answers": read(tmp_path / "none", packed))
    assert sorted(rp.d2_runs()) == ["qwen72b-react|r1", "qwen72b|r1"]


def test_the_extract_is_the_published_file():
    """The shipped extract names the published eval_df.csv by size and md5, and holds its four
    open-answer run sets over BixBench v1.0's 296 questions, 159 of them numeric."""
    doc = rp.load_extract()
    assert doc["source"] == rp.SOURCE
    assert len(doc["items"]) == 296
    assert sorted(doc["open_runs"]) == sorted(rp.OPEN_RUNS)
    assert sum(len(v) for v in doc["open_runs"].values()) == 9284
    numeric = [q for q, e in doc["items"].items() if rp.numeric_ranks(e["options"]) is not None]
    assert len(numeric) == 159
    assert all(e["options"][0] == e["ideal"] for e in doc["items"].values())


def test_the_registered_plan_is_unchanged():
    """PREREGISTRATION.md was pushed before its analyses ran; results are appended below it."""
    text = (ROOT / "PREREGISTRATION.md").read_bytes()
    assert hashlib.sha256(text[:6631]).hexdigest() == \
        "cedf2e706f55232d66ab513af1ad974c20391a51019c2e4e8dbd15878dc4a62d"
    assert text[:6631].decode().rstrip().endswith("(Appended after the analyses ran; nothing above this line changes.)")


def test_d1_passes_what_the_paper_says_it_passes():
    path = ROOT / "results" / "replication.json"
    if not path.exists():
        pytest.skip("replication not run")
    d1 = json.loads(path.read_text())["D1"]
    assert d1["verdicts"] == {"H1": True, "H2": True, "H3": True, "H4": True, "H5": True}
    assert d1["n_items"] == {"moved to an edge": 41, "moved inward": 32, "kept": 86}
    assert all(v["lo"] > 0 for v in d1["per_run"].values())


def test_kappa_and_the_grader_reading_of_a_number():
    assert tc.kappa([1, 1, 0, 0], [1, 1, 0, 0]) == pytest.approx(1.0)
    assert tc.kappa([1, 0, 1, 0], [0, 1, 0, 1]) == pytest.approx(-1.0)
    # the last number in the answer, a per cent written on one side only normalised
    assert tc.relative_error("about 0.33", "33%") == pytest.approx(0.0)
    assert tc.relative_error("first 5 then 11", "10") == pytest.approx(0.1)
    assert tc.relative_error("no number", "10") is None


def test_range_keys_are_read_as_their_authors_wrote_them():
    widths = tc.range_widths()
    v15 = [w for w in widths if w["release"] == "v1.5"]
    assert len(v15) == 60
    one = next(w for w in v15 if w["ideal"] == "(1.50,1.54)")
    assert one["half_width"] == pytest.approx(0.04 / 3.04)


def test_key_marginal_borrows_other_keys_and_the_near_draw_stays_within_ten():
    released = [{"ideal": str(v), "distractors": [str(v + 1), str(v + 2), str(v + 3)],
                 "question": f"q{v}", "cluster": f"c{v % 3}"} for v in (1, 2, 3, 5, 8, 13, 21, 34, 55, 89)]
    wide, near = wp.key_marginal(released, seed=1)
    keys = {r["ideal"] for r in released}
    for row in wide:
        assert len(set(row["distractors"])) == 3 and row["ideal"] not in row["distractors"]
        assert set(row["distractors"]) <= keys
    for row in near:
        k = float(row["ideal"])
        assert all(0.1 <= float(d) / k <= 10 for d in row["distractors"])


def test_the_catch_axis_counts_misses_and_hits_by_tolerance():
    rows = [{"question": "Q", "ideal": "10", "distractors": ["9", "11", "12"]}]
    qid_of, capsule_of = {("Q", "10"): "q"}, {"q": "c"}
    answers = [("q", "10.1"), ("q", "3"), ("q", "10.9"), ("q", "no number")]
    out = wp.catch(rows, qid_of, capsule_of, answers)
    # 10.1 is a hit read as the key; 3 is a miss whose nearest option is 9; 10.9 is 9% off and
    # nearest 11
    assert out["n_hits"] == 1 and out["n_misses"] == 2
    assert out["kept_hits"]["mean"] == pytest.approx(100.0)
    assert out["credited_misses"]["mean"] == pytest.approx(0.0)
