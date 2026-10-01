"""Other designs (option_design.py): the generator places the key at the rank asked for, with distinct values; the
error-modelled distractors take the most frequent wrong numbers on each side of the key, never one within the
tolerance; a middle-rank policy never leaves the key extreme; the rank rule's gain in theory is $2/(k(k-2))$ for a
rank uniform over the middle ranks and zero for one uniform over all; and the shipped report agrees with it."""
import json
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import option_design as od  # noqa: E402


@pytest.mark.parametrize("k", od.KS)
@pytest.mark.parametrize("options", [["12.5", "10.1", "15.2", "20.0"], ["0.02", "-0.1", "0.3", "0.5"]])
def test_spaced_places_the_key_at_every_rank(k, options):
    rng = random.Random(7)
    for target in range(k):
        got = od.spaced(options, rng, k, target)
        assert got is not None and got[0] == options[0] and len(got) == k
        assert od.rank_of(got) == target
        assert len(set(od.values_of(got))) == k


def test_spaced_keeps_the_key_sign():
    rng = random.Random(11)
    for target in range(4):
        got = od.spaced(["-0.35", "-0.2", "-0.5", "-0.8"], rng, 4, target)
        assert got is not None and all(v < 0 for v in od.values_of(got))


def test_error_distractors_are_the_most_frequent_wrong_numbers():
    pool = [7.0] * 5 + [7.1] * 2 + [5.0] * 3 + [20.0] + [10.2] * 9 + [-3.0] * 6
    got, from_pool = od.from_errors(["10", "8", "12", "15"], pool, random.Random(1), 2)
    assert got == ["10", "7", "5", "20"] and from_pool == 3        # 10.2 is within 5%, -3 of the other sign
    assert od.rank_of(got) == 2
    assert od.cluster_numbers([7.0] * 5 + [7.1] * 2 + [5.0] * 3, 10.0) == [(7.0, 7), (5.0, 3)]


def test_error_distractors_fill_a_short_side_from_the_generator():
    got, from_pool = od.from_errors(["10", "8", "12", "15"], [20.0, 20.0], random.Random(1), 1)
    assert from_pool == 1 and od.rank_of(got) == 1 and len({od.parse_number(x) for x in got}) == 4


@pytest.mark.parametrize("k", od.KS)
def test_rank_policies(k):
    rng = random.Random(3)
    for _ in range(50):
        middle = od.draw_target("middle", k, rng, 0)
        assert sorted(middle) == list(range(1, k - 1))
        assert sorted(od.draw_target("uniform", k, rng, 0)) == list(range(k))
    assert od.draw_target("released", 4, rng, 2) == [2]


@pytest.mark.parametrize("k", od.KS)
def test_theory_leak(k):
    assert od.theory_leak(f"k{k}-middle", k) == pytest.approx(100 * 2 / (k * (k - 2)))
    assert od.theory_leak(f"k{k}-uniform", k) == 0.0


@pytest.mark.skipif(not od.OUT.exists(), reason="report not built")
def test_report_middle_designs_never_extreme():
    report = json.loads(od.OUT.read_text())
    for rel in ("v1.0", "v1.5"):
        for name, entry in report["releases"][rel]["designs"].items():
            shares = entry["leak"]["rank_shares"]
            assert sum(shares) == pytest.approx(1.0)
            if "middle" in name:
                assert shares[0] == shares[-1] == 0
