"""The strong closed agent (strong_agent.py): its table rows come from the shipped report, and its rule
contrasts use the who-pays table's own definition."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import strong_agent as sa  # noqa: E402

REPORT = ROOT / "results" / "strong_agent.json"
pytestmark = pytest.mark.skipif(not REPORT.exists(), reason="strong_agent results not present")


def test_the_table_has_six_rows_each_a_quantity_of_the_report():
    report = json.loads(REPORT.read_text())
    rows = sa.table_rows(report)
    assert len(rows) == 6 and all(r.endswith("\\\\") for r in rows)
    assert f"${report['within_5pct']['mean']:.1f}$" in rows[0]
    assert f"${report['rule|repaired-placebo|all']['mean']:+.1f}$" in rows[3]


def test_a_decline_is_told_from_an_answer():
    assert sa.DECLINE.search("Cannot be determined in this environment because the .xls file cannot be read")
    assert sa.DECLINE.search("The Mann-Whitney U statistic could not be computed")
    assert not sa.DECLINE.search("The median saturation is about zero")


def test_the_rule_contrast_is_the_who_pays_definition():
    report = json.loads(REPORT.read_text())
    # run_set_scaling.one averages each item's runs, then items: the same means as the intervals here
    assert abs(report["who_pays"]["moved"] - report["rule|repaired-placebo|moved"]["mean"]) < 0.01
    assert abs(report["who_pays"]["all"] - report["rule|repaired-placebo|all"]["mean"]) < 0.01
    assert abs(report["who_pays"]["within_5pct"] - report["within_5pct"]["mean"]) < 0.01


def test_table_1_gives_gpt_51_forced_and_with_a_refusal_option_then_the_current_agents():
    import score_decomposition as sd
    report = json.loads((ROOT / "results" / "score_decomposition.json").read_text())
    rows = sd.table_rows(report)
    at = next(i for i, r in enumerate(rows) if r.startswith("v1.5, gpt-5.1 & forced"))
    assert rows[at + 1].startswith(" & with refusal")
    assert [r.split(" & ")[0] for r in rows[at + 2:]] == ["v1.5, gpt-6-luna", "v1.5, DeepSeek-V4-Pro"]
    s = report["v1.5 closed"]["gpt-5.1"]["gpt-4o forced"]
    shipped = json.loads(REPORT.read_text())
    # both reports read the same 210 runs, and the same share within 5% of the key
    assert s["n_runs"] == shipped["n_runs"] == 210
    assert abs(s["tolerance"] - shipped["within_5pct"]["mean"]) < 0.05
