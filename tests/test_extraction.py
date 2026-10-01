"""How a number is taken from an answer (answer_extraction.py), knowledge short of a value in BixBench's own
no-data replies (partial_knowledge.py), who pays for a hidden rank (run_set_scaling.py) and the in-context
panel's table (icl_analysis.table_rows)."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import answer_extraction as ax  # noqa: E402
import bixbench_withdata as bw  # noqa: E402
import partial_knowledge as pk  # noqa: E402
import run_set_scaling as rss  # noqa: E402
from answer_numbers import graded  # noqa: E402


@pytest.mark.parametrize("answer,kind", [
    ("", "empty"), (None, "empty"), ("   ", "empty"),
    ("0.35", "a number"), ("35%", "a number"), ("1,234.5", "a number"), ("-2.1e-3", "a number"),
    ("The fold change is 2.27", "one number in text"),
    ("The log2 fold change is 2.27", "one number in text"),            # the 2 of log2 is part of a name
    ("ANOVA F statistic is 5.46, p-value is 0.03", "several numbers"),
    ("Insufficient data for conclusion", "no number"), ("N/A", "no number"),
])
def test_an_answer_is_classed_by_what_it_gives_to_read(answer, kind):
    assert ax.category(answer) == kind


def test_the_last_number_is_read_as_the_tolerance_reads_it():
    assert ax.last_number("The log2 fold change is 2.27", "2.3") == pytest.approx(2.27)
    assert ax.last_number("F is 5.46, p-value 0.03", "0.03") == pytest.approx(0.03)
    assert ax.last_number("0.35", "35%") == pytest.approx(35.0)          # a fraction against a per cent key
    assert ax.last_number("1,234.5 reads", "1200") == pytest.approx(1234.5)
    assert ax.last_number("no number here", "1") is None
    # the same number the tolerance grades: within 5% of the key exactly when this value is
    for answer, key in (("about 0.36", "35%"), ("2.27", "2.3"), ("it is 12", "10")):
        v, k = ax.last_number(answer, key), ax.last_number(key, key)
        assert graded(answer, key, 0.05) == (abs(v - k) <= 0.05 * abs(k))


def test_the_last_number_rule_reads_what_the_registered_rule_cannot():
    options = ["2.27", "1.5", "3.4", "5.0"]                                # key first
    answer = "The log2 fold change value for ENO1 is 2.27"
    assert bw.nearest_is_key(answer, options) == 0.0                       # registered: not a number
    assert ax.nearest_is_key_last(answer, options) == 1.0
    assert ax.nearest_is_key_last("2.27", options) == bw.nearest_is_key("2.27", options)
    assert ax.nearest_is_key_last("", options) == 0.0


def test_the_reading_is_swapped_and_put_back():
    saved = bw.nearest_is_key
    with ax.reading(ax.nearest_is_key_last):
        assert bw.nearest_is_key is ax.nearest_is_key_last
    assert bw.nearest_is_key is saved
    with pytest.raises(RuntimeError):
        with ax.reading(ax.nearest_is_key_last):
            raise RuntimeError
    assert bw.nearest_is_key is saved


def test_a_reply_that_says_it_cannot_tell_states_no_number():
    assert pk.stated("0.35", "0.3") == pytest.approx(0.35)
    assert pk.stated("The value cannot be determined without the data; typically 0.05", "0.05") is None
    assert pk.stated("", "1") is None
    assert pk.stated("I am unable to provide specific numerical results", "1") is None


def test_the_prediction_from_stated_numbers_is_their_nearest_reading_or_a_guess():
    rows = [{"capsule": "a", "forced": 1.0, "number": 1.0, "nearest": 1.0, "within": False},
            {"capsule": "a", "forced": 1.0, "number": None, "nearest": None, "within": None},
            {"capsule": "b", "forced": 0.0, "number": 2.0, "nearest": 0.0, "within": False},
            {"capsule": "b", "forced": 1.0, "number": None, "nearest": None, "within": None}]
    got = pk.summary(rows)
    assert got["stated_share"] == pytest.approx(50.0)
    assert got["predicted_forced"] == pytest.approx(100 * (1.0 + 0.25 + 0.0 + 0.25) / 4)
    assert got["forced_minus_predicted"]["mean"] == pytest.approx(100 * 3 / 4 - 37.5)


def test_run_sets_are_labelled_as_the_table_prints_them():
    assert rss.label("v1.5 new|gemma27b|r1") == "v1.5, gemma-3-27b, text, seed 2"
    assert rss.label("v1.5 new|qwen3-235b|r0") == "v1.5, Qwen3-235B-A22B, run 1"
    assert rss.label("v1.0|claude_open_image") == "v1.0, Claude 3.5 Sonnet, images"
    assert rss.label("v1.5|glm45air-react") == "v1.5, GLM-4.5-Air, published"


def test_the_scaling_table_is_in_order_of_how_often_a_run_set_is_near_the_key():
    report = json.loads((ROOT / "results" / "run_set_scaling.json").read_text())
    rows = rss.table_rows(report)
    assert len(rows) == len(report["run_sets"]) == 19
    within = [float(r.split("&")[1].strip().strip("$")) for r in rows]
    assert within == sorted(within)


def test_the_in_context_table_is_one_line_a_model_in_order_of_size():
    import icl_analysis
    from probe_analysis import PARAMETERS_B
    doc = json.loads((ROOT / "results" / "icl_probe.json").read_text())
    rows = icl_analysis.table_rows(doc)
    assert len(rows) == len(doc["models"]) == 8
    sizes = [float(r.split("&")[1].strip().strip("$")) for r in rows]
    assert sizes == sorted(sizes) and all(m["model"] in PARAMETERS_B for m in doc["models"])
