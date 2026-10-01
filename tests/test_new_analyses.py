"""The helpers of the analyses outside the pre-specified test: the proximity weight fitted on other keys
(lambda_holdout.py), a degenerate run (degenerate_runs.py), a key BixBench-Verified-50 shows as the same
(verified50.py), the options whose values are not numbers (nonnumeric_excess.py), and the pre-specified test read
from the last number (extraction_verdicts.py)."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import degenerate_runs as dr  # noqa: E402
import extraction_verdicts as ev  # noqa: E402
import lambda_holdout as lh  # noqa: E402
import nonnumeric_excess as nn  # noqa: E402
import verified50 as v5  # noqa: E402


def test_proximity_weight():
    assert lh.weight(100, 10, 90, 60) == pytest.approx((1 - 0.1) * (4 * 60 / 90 - 1) / 3)
    assert lh.weight(100, 0, 100, 25) == pytest.approx(0.0)       # the nearest no more often than any other
    assert lh.weight(100, 0, 100, 100) == pytest.approx(1.0)      # always the nearest
    assert lh.weight(0, 0, 0, 0) is None


def test_degenerate_flags():
    flags, turns, truncated, _ = dr.flags_of({"answer": "", "termination": "max_steps",
                                              "log": [{"finish": "stop"}, {"finish": ["length"]}]})
    assert flags == {"no_answer": True, "step_limit": True, "truncated_reply": True, "wall_clock": False,
                     "degenerate": True}
    assert (turns, truncated) == (2, 1)
    flags, *_ = dr.flags_of({"answer": "5", "termination": "submitted", "log": None})
    assert not flags["degenerate"] and flags["truncated_reply"] is None
    assert dr.flags_of({"answer": None, "termination": "wallclock", "log": None})[0]["wall_clock"]


def test_same_key():
    assert v5.same_key("35%", "35%") and v5.same_key("0.05", "0.050")
    assert v5.same_key("6.02", "5.81") is False


@pytest.mark.parametrize("options,kind", [(["(0.6, 0.7)", "(0.7, 0.8)", "(0.8,0.9)", "(0.5,0.6)"], "interval or bound"),
                                          (["Increased", "Decreased", "No change", "Doubled"], "direction or change"),
                                          (["BRCA1", "TP53", "EGFR", "KRAS"], "name")])
def test_classify(options, kind):
    assert nn.classify(options) == kind


def test_nearest_option_by_value_and_wording():
    spans = ["(0.7, 0.8)", "(0.6, 0.7)", "(0.8,0.9)", "(0.5,0.6)"]
    assert nn.number_nearest("about 0.75", spans) == "key"
    assert nn.number_nearest("0.65", spans) == "other"
    assert nn.number_nearest("none", spans) == "neither"
    names = ["TP53", "BRCA1", "EGFR", "KRAS"]
    assert nn.wording_nearest("TP53 is the driver", names) == "key"
    assert nn.wording_nearest("EGFR", names) == "other"
    assert nn.wording_nearest("nothing", names) == "none"


@pytest.mark.skipif(not ev.OUT.exists(), reason="report not built")
def test_last_number_reading_keeps_every_verdict():
    report = json.loads(ev.OUT.read_text())
    assert report["check"]["reproduced"] and not report["check"]["mismatches"]
    verdict = lambda h, x: (x["lo"] <= 0 <= x["hi"]) if h == "H2" else x["pass"]
    for h, sets in report["summary"].items():
        for v in sets.values():
            assert verdict(h, v["whole answer"]) == verdict(h, v["last number"])


import frontier_agents as fa  # noqa: E402


@pytest.mark.skipif(not fa.OUT.exists(), reason="report not built")
def test_frontier_agents_report():
    report = json.loads(fa.OUT.read_text())
    assert set(report["agents"]) == {label for _, label in fa.MODELS}
    for label, a in report["agents"].items():
        assert a["n_runs"] == a["n_items"] >= 90
        t = a["table1"]["gpt-4o forced"]
        assert abs(t["tolerance"] - a["within_5pct"]["mean"]) < 0.05
        assert sum(p["points"] for p in a["corrected"]["parts"].values()) == pytest.approx(
            a["corrected"]["corrected_minus_tolerance"]["mean"])
    assert len(fa.table_rows(report)) == len(fa.REPORTED)
