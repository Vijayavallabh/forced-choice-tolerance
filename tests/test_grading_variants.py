"""The gradings of ``grading_variants.py``: a reply constrained to a letter is read from its JSON, else by BixBench's
own parser; the within-5% option replaces BixBench's refusal as a fifth shuffled option; and tab:readers2's rows
print the rule's weight as 1 and a grading with no $U-P$ (BixBench's own grades) with its weight alone."""
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import grading_variants as gv  # noqa: E402
import grading_variants_analysis as gva  # noqa: E402


def test_a_constrained_reply_is_read_from_its_json():
    assert gv.parse_letter(json.dumps({"analysis": "the nearest value", "answer": "c"}), True) == "C"
    # anything else falls back to BixBench's own parser, which scores a reply without a letter as "Z", no option
    assert gv.parse_letter(json.dumps({"answer": "AB"}), True) == "Z"
    assert gv.parse_letter("<answer>B</answer>", True) == "B"


def test_the_within_option_is_a_fifth_shuffled_option():
    options = ["2.0", "1.0", "3.0", "4.0"]
    formatted, correct, none_letter, shown = gv.with_none_within("How much?", options, random.Random(3))
    assert len(shown) == 5 and gv.NONE_WITHIN in shown and shown[ord(correct) - 65] == "2.0"
    assert shown[ord(none_letter) - 65] == gv.NONE_WITHIN
    assert formatted.count("\n") == 6 and "within 5%" in gv.NONE_WITHIN


def test_the_schema_admits_only_the_shown_letters():
    s = gv.schema("ABCD", analysis=True)["json_schema"]["schema"]
    assert s["properties"]["answer"]["enum"] == ["A", "B", "C", "D"] and s["required"] == ["analysis", "answer"]
    assert s["additionalProperties"] is False


def test_table2_rows_print_the_rule_and_bixbench_grades():
    iv = {"mean": 18.6, "lo": 8.7, "hi": 30.7}
    report = {"table2": [
        {"runs": "v1.0, published", "grader": "nearest option", "lambda": 1.0, "moved": iv, "p": 5e-6,
         "predicted": None, "all": 3.3},
        {"runs": "v1.0, published", "grader": "published", "lambda": 0.434, "moved": None, "p": None,
         "predicted": 13.9, "all": None}]}
    rows = gva.table2_rows(report)
    assert rows[0] == ("v1.0, published & nearest option & $1$ & $+18.6$ {\\scriptsize$[+8.7,+30.7]$} & "
                       "$<0.001$ & $+3.3$\\\\")
    assert rows[1] == " & published & $0.43$ & -- & -- & --\\\\"
