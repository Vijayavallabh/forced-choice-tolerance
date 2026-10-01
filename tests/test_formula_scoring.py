"""The correction for guessing (formula_scoring.py): each answer's kind as the split counts it, and in the shipped
report the corrected score is $(S-1/4)/(3/4)$, the split sums exactly to the corrected score minus the tolerance,
and a grading that always selects an option differs in its omission-aware score only by the empty answers."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import formula_scoring as fs  # noqa: E402

OPTIONS = ["10", "5", "20", "40"]


@pytest.mark.parametrize("answer,kind", [("10.2", "correct"), ("12", "miss, key nearest"), ("18", "miss, other nearest"),
                                         ("no idea", "miss, neither"), ("", "empty"), (" ", "empty")])
def test_kind_of(answer, kind):
    assert fs.kind_of({"answer": answer, "options": OPTIONS}) == kind


def test_the_decomposition_is_exact_algebra():
    # (S - 1/k)/(1 - 1/k) - t = k/(k-1) [h(a_h - 1) + sum_j m_j (a_j - 1/k) - e/k], with h = t
    k, h, a_h, e = 4, 0.2, 0.9, 0.1
    m, a = [0.3, 0.25, 0.15], [0.5, 0.1, 0.3]
    s = h * a_h + sum(x * y for x, y in zip(m, a))
    lhs = (s - 1 / k) / (1 - 1 / k) - h
    rhs = k / (k - 1) * (h * (a_h - 1) + sum(x * (y - 1 / k) for x, y in zip(m, a)) - e / k)
    assert lhs == pytest.approx(rhs)


@pytest.mark.skipif(not fs.OUT.exists(), reason="report not built")
def test_report():
    report = json.loads(fs.OUT.read_text())
    assert report["reported"]["corrected"] == pytest.approx((64.4 - 25) / 0.75)
    for r in report["rows"]:
        assert r["corrected"] == pytest.approx((r["score"] - 25) / 0.75)
        assert sum(p["points"] for p in r["parts"].values()) == pytest.approx(r["corrected_minus_tolerance"]["mean"])
        assert sum(p["share_of_runs"] for p in r["parts"].values()) == pytest.approx(100.0)
        if r["misses_selected"] == pytest.approx(100.0):
            # a grading that always selects differs from the correction only by the empty answers, scored as
            # omissions (0) rather than as wrong (-1/(k-1))
            empty = r["parts"]["empty"]["share_of_runs"]
            assert r["omission_aware_minus_tolerance"] == pytest.approx(r["corrected_minus_tolerance"]["mean"] + empty / 3)
    assert len(fs.table_rows(report)) == len(report["rows"])
