"""The other references the forced-choice excess is measured against (reference_check.py): any number in
the answer, a log-scale band for p-values, and the table's rows from the shipped result."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import reference_check as rc  # noqa: E402
from answer_numbers import graded  # noqa: E402


def test_any_number_finds_the_number_the_last_number_misses():
    answer = "p = 0.03, below the 0.05 threshold"
    assert not graded(answer, "0.03", 0.05)
    assert rc.graded_any(answer, "0.03", 0.05)
    assert not rc.graded_any("no number here", "0.03", 0.05)


def test_any_number_keeps_the_per_cent_convention():
    # a fraction against a key written as a per cent is read as that fraction times 100, as the tolerance reads it
    assert rc.graded_any("about 0.35 of cells", "35%", 0.05) == bool(graded("0.35", "35%", 0.05))


def test_the_log_scale_band_is_a_factor_either_side():
    assert rc.graded_log("0.012", "0.01", 2.0) and rc.graded_log("0.006", "0.01", 2.0)
    assert not rc.graded_log("0.004", "0.01", 2.0)
    assert rc.graded_log("0.1", "0.01", 10.0) and not rc.graded_log("0.11", "0.01", 10.0)
    assert not rc.graded_log("-0.01", "0.01", 10.0) and not rc.graded_log("", "0.01", 10.0)


def test_the_table_has_a_row_per_grading_and_the_tolerance_column_matches_table_1():
    report = json.loads((ROOT / "results" / "reference_check.json").read_text())
    rows = rc.table_rows(report)
    assert len(rows) == len(rc.LATEX) == 6
    released = json.loads((ROOT / "results" / "score_decomposition.json").read_text())
    for group, reading, _, _ in rc.LATEX:
        block, name = group.split("|")
        source = released["v1.0"][name] if block == "v1.0" else released["v1.5"]["pooled"]
        got = report["groups"][group][reading]["tolerance"]["excess"]["mean"]
        assert abs(got - source[reading]["score_minus_tolerance"]["mean"]) < 0.05, (group, reading)
