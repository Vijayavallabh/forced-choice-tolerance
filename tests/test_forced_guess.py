"""``forced_guess.py``: a reply saying the value cannot be determined states no number, whatever it quotes; a
number is read as the tolerance reads it and mapped to the released option nearest it; the forced prompt is the
free arm's with a best estimate required."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import forced_guess as fg  # noqa: E402
import free_response as fr  # noqa: E402

OPTIONS = ["0.25", "0.5", "0.1", "0.9"]          # key first


def test_a_decline_states_no_number():
    assert fg.read("The value cannot be determined without the data. FINAL: 0.25", OPTIONS) == (None, None, False)


def test_a_number_is_read_as_the_tolerance_reads_it():
    number, nearest, within = fg.read("FINAL: 0.26", OPTIONS)
    assert number == 0.26 and nearest == 1.0 and within is True
    number, nearest, within = fg.read("Probably about 0.45.\nFINAL: 0.45", OPTIONS)
    assert number == 0.45 and nearest == 0.0 and within is False


def test_the_forced_prompt_adds_a_best_estimate_to_the_free_arm():
    assert fg.PROMPTS["stated"] == fr.FREE_SYSTEM
    assert fg.PROMPTS["forced"].startswith(fr.FREE_SYSTEM) and "best estimate" in fg.PROMPTS["forced"]
    p = fg.payload(fg.PROMPTS["forced"], {"question": "How many?"})
    assert p["temperature"] == 0.0 and p["messages"][1]["content"] == "Question: How many?"
