"""How a number is read from an answer (answer_numbers.py), and the camera-ready fixes that rest on it."""
import math
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import answer_numbers as an  # noqa: E402


@pytest.mark.parametrize("text", ["CD14", "m6A", "H3K27ac", "at2759", "v1.5", "IL6", "chr7_12"])
def test_digits_inside_an_identifier_are_not_numbers(text):
    assert an.numbers(text) == []


@pytest.mark.parametrize("text, found", [
    ("4. Extract the log2 fold change for GRIK5.", []), ("1. The p-value is 0.03\n2. Done", [0.03]),
    ("10) item", []), ("- 2. item", []), ("2. 0.05", [0.05]), ("3.", [3.0]), ("FINAL: 3.", [3.0]),
    ("3.5 units", [3.5])])
def test_a_list_marker_labels_a_line_and_is_not_a_number(text, found):
    assert [v for v, _ in an.numbers(text)] == found


@pytest.mark.parametrize("text, value", [
    ("7.29 × 10⁷", 7.29e7), ("2.1 x 10^-26", 2.1e-26), ("10⁻⁵", 1e-5), (r"5 \times 10^{-4}", 5e-4),
    ("3·10^(4)", 3e4), ("1.2*10**-3", 1.2e-3), ("4.5e-3", 4.5e-3), ("7.29e+07", 7.29e7),
    ("3,827 genes", 3827.0), ("−0.5", -0.5), ("105", 105.0)])
def test_scientific_notation_and_separators_read_as_one_number(text, value):
    got = an.numbers(text)
    assert len(got) == 1 and math.isclose(got[0][0], value, rel_tol=1e-12)


def test_a_per_cent_sign_is_read_on_either_side():
    assert an.graded("0.29%", "0.29%", 0.05)
    assert an.graded("29%", "0.29", 0.05)
    assert an.graded("0.29", "29%", 0.05)
    assert not an.graded("29", "0.29", 0.05)


def test_the_bound_is_inclusive():
    assert an.graded("0.076", "0.08", 0.05)
    assert an.graded("0.95", "1.0", 0.05)
    assert an.graded("19", "20", 0.05)
    assert not an.graded("0.0759", "0.08", 0.05)


def test_against_a_key_of_zero_the_tolerance_is_absolute():
    assert an.graded("0.04", "0", 0.05)
    assert not an.graded("0.06", "0", 0.05)


def test_the_last_number_is_read_unless_the_first_is_asked_for():
    answer = "The answer is 12, from 3 samples"
    assert not an.graded(answer, "12", 0.05)
    assert an.graded(answer, "12", 0.05, pick="first")
    assert an.graded("CD14+ monocytes: 0.31", "0.31", 0.05)
    assert an.graded("about 7.29 × 10⁷ reads", "7.3E+07", 0.05)
    assert an.graded("no number here", "3", 0.05) is False
    assert an.graded("12", "not a number", 0.05) is None


def test_the_distance_is_the_rule_s_distance_and_does_not_saturate():
    for a, v in ((2.0, 1.0), (0.3, 0.5), (-4.0, -1.0), (1e-6, 3e-6)):
        assert math.isclose(an.option_distance(a, v), abs(a - v) / (abs(a) + abs(v)), rel_tol=1e-12)
    assert an.option_distance(-1.0, 1.0) == 1.0 and an.option_distance(0.0, 1.0) == 1.0
    assert an.nearest_ranks(1e17, [2.0, 1.0, 0.5, 0.25]) == [0]
    assert an.nearest_ranks(1e-17, [2.0, 1.0, 0.5, 0.25]) == [3]


def test_ties_are_kept_and_an_answer_of_the_other_sign_ties_every_option():
    assert an.nearest_ranks(2.0, [1.0, 4.0]) == [0, 1]
    assert an.nearest_ranks(-3.0, [1.0, 2.0, 3.0, 4.0]) == [0, 1, 2, 3]
    assert an.nearest_ranks(0.0, [1.0, 2.0]) == [0, 1]


def test_an_answer_far_beyond_an_extreme_key_is_credited_to_it():
    import bixbench_withdata as bw
    assert bw.nearest_is_key("1e17", ["4", "3", "2", "1"]) == 1.0
    assert bw.nearest_is_key("1e17", ["1", "4", "3", "2"]) == 0.0


def test_a_constrained_reply_cut_off_after_its_answer_is_read():
    import grading_variants as gv
    assert gv.parse_letter('{"answer": "B"' + " " * 40, True) == "B"
    assert gv.parse_letter('{"answer": "C"}', True) == "C"
    assert gv.parse_letter("{" + " " * 60, True) == "Z"
    assert gv.parse_letter("<answer>D</answer>", False) == "D"


@pytest.mark.parametrize("question, kind", [
    ("What is the adjusted p-value for the enrichment of the top pathway?", "p-value"),
    ("How many genes are significantly differentially expressed (padj < 0.05)?", "count"),
    ("What percentage of genes have padj < 0.05?", "percentage or proportion"),
    ("What is the log2 fold change of gene X?", "fold change or ratio")])
def test_a_question_that_filters_on_a_p_value_is_not_a_p_value_question(question, kind):
    import leak_origin
    assert leak_origin.kind(question) == kind


def test_a_point_mass_at_zero_is_the_interval_s_end_when_it_exceeds_the_tail():
    import replication as rp
    sums = np.array([0.0] * 16 + [1.0, 2.0, 0.5, 1.5, 3.0])
    assert rp.zero_ends(sums, 0.9949, 0.012, 0.2) == (0.0, 0.2)       # (16/21)^21 = 0.33% > 0.255%
    assert rp.zero_ends(sums, 0.95, 0.012, 0.2) == (0.012, 0.2)        # 0.33% < 2.5%
    assert rp.zero_ends(-sums, 0.9949, -0.2, -0.012) == (-0.2, 0.0)
    mixed = np.array([0.0] * 16 + [1.0, -2.0, 0.5, 1.5, 3.0])
    assert rp.zero_ends(mixed, 0.9949, -0.1, 0.2) == (-0.1, 0.2)


def test_the_vectorised_tau_is_kendall_s_tau_b():
    import ranking_check as rc
    rng = np.random.default_rng(3)
    a = rng.integers(0, 4, size=(200, 7)).astype(float)
    b = rng.integers(0, 4, size=(200, 7)).astype(float)
    got = rc.taus(a, b)
    assert np.allclose(got, [rc.kendall(x, y) for x, y in zip(a, b)])


def test_a_packed_episode_keeps_its_truncated_turns():
    import degenerate_runs as dr
    import replication as rp
    full = {"answer": "3", "termination": "submitted",
            "log": [{"finish": "stop"}, {"finish": "length"}, {"finish": ["length"]}, {"reply": "x"}]}
    packed = {"answer": "3", "termination": "submitted", **rp.turn_counts(full)}
    assert rp.turn_counts(full) == {"turns": 3, "truncated_turns": 2}
    assert dr.flags_of(full) == dr.flags_of(packed)
    assert dr.flags_of(packed)[0]["truncated_reply"] is True
