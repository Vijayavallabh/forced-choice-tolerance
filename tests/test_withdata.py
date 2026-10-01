"""The GPU round of 23 Sep 2026, tested where it could fail quietly.

Each test is a failure that happened, or one the code would not announce:

  * agents call ``submit_answer`` and ``list_workdir`` from *inside* a code cell;
    without the kernel-side functions one agent spent 24 steps unable to submit;
  * an ``edit_cell idx=N`` directive written on the line *before* a block was
    appended as a new cell instead of replacing the old one;
  * BixBench's own MCQ parse is strict -- ``<answer> B </answer>`` is no pick --
    and a lenient parse here would score a different pipeline than the one
    published;
  * the published extraction flattens the capsule's Data folder and drops its
    Notebook folder and ``.ipynb``; getting that wrong hands the agent the
    reference analysis;
  * a wall-clock budget started at queue time cut 303 episodes off at step 0;
  * every number the paper prints from the round re-derives from shipped files.
"""
import gzip
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import bixbench_agent as ba  # noqa: E402
import bixbench_withdata as bw  # noqa: E402
import sandbox_repl as sr  # noqa: E402


# --- the agent's protocol -----------------------------------------------------

def test_one_tool_per_reply_and_a_submission_after_code_is_not_taken():
    kind, code, idx, n_blocks, also_submitted = ba.parse_action(
        "Load it.\n```python\nx = 1\n```\n```python\ny = 2\n```\nsubmit_answer(\"<answer>3</answer>\")")
    assert (kind, code, idx, n_blocks, also_submitted) == ("edit", "x = 1", None, 2, True)


def test_submission_and_listing_in_text_are_read():
    assert ba.parse_action('Done.\nsubmit_answer("<answer>0.0002</answer>")')[:2] == ("submit", "0.0002")
    assert ba.parse_action("list_workdir()") == ("list",)
    assert ba.parse_action("The answer is 5.") == ("none",)


def test_edit_directive_inside_or_just_before_the_block():
    assert ba.parse_action("```python\n# edit_cell idx=2\nprint(2)\n```")[1:3] == ("print(2)", 2)
    assert ba.parse_action("Fix it:\nedit_cell idx=4\n```python\nprint(4)\n```")[1:3] == ("print(4)", 4)
    assert ba.parse_action("Fix it:\n# edit_cell idx=3\n```python\nprint(3)\n```")[1:3] == ("print(3)", 3)
    # a directive written after the block is not one: the block is appended
    assert ba.parse_action("```python\nprint(5)\n```\nedit_cell idx=1")[2] is None


def test_bash_and_r_blocks_become_cell_magics():
    assert ba.parse_action("```bash\nls\n```")[1] == "%%bash\nls"
    assert ba.parse_action("```r\nx <- 1\n```")[1] == "%%R\nx <- 1"


def test_the_kernel_exposes_the_tools_and_translates_notebook_syntax():
    assert callable(sr.submit_answer) and callable(sr.list_workdir)
    out = sr._translate("!ls -la\n%matplotlib inline\n%timeit f()\n  !echo hi")
    lines = out.splitlines()
    assert lines[0] == "__sandbox_sh__('ls -la')"
    assert lines[1] == "pass"
    assert lines[2].startswith("print(") and "%timeit" in lines[2]
    assert lines[3] == "  __sandbox_sh__('echo hi')"


def test_a_dead_replica_fails_over_and_takes_no_new_episodes(monkeypatch):
    """A GPU fault killed one of two replicas mid-run. Least-loaded assignment
    then steered new episodes *to* it -- they failed fast, so it always looked
    the least loaded -- and 66 episodes failed. Two failures now move an episode
    to the other replica, and a replica whose last call failed takes no new ones."""
    import asyncio
    import httpx

    real_sleep = asyncio.sleep
    monkeypatch.setattr(ba.asyncio, "sleep", lambda *a, **k: real_sleep(0))
    monkeypatch.setattr(ba, "REPLICA_DOWN", set())

    class Reply:
        status_code, text = 200, "{}"

        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}], "usage": {}}

    class Client:
        seen = []

        async def post(self, url, json=None, timeout=None):
            self.seen.append(url.split("/v1")[0])
            if "dead" in url:
                raise httpx.ConnectError("refused")
            return Reply()

    route = {"base": "http://dead", "others": ["http://live"]}
    reply = asyncio.run(ba.chat(Client(), route, "m", [], 10, 1.0, 1, {}, []))
    assert reply[0] == "ok" and route["base"] == "http://live"
    assert Client.seen == ["http://dead", "http://dead", "http://live"]
    assert ba.REPLICA_DOWN == {"http://dead"}


def test_the_wall_clock_starts_when_the_sandbox_does():
    import inspect
    source = inspect.getsource(ba.episode)
    slot = source.index("async with container_slots:")
    clock = source.index("started = time.time()")
    assert clock > slot, "the episode's clock starts before it has a sandbox"


def test_capsule_extraction_matches_bixbench(tmp_path):
    zpath = tmp_path / "CapsuleFolder-x.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("CapsuleData-x/counts.csv", "a,b\n1,2\n")
        zf.writestr("CapsuleData-x/sub/meta.tsv", "k\tv\n")
        zf.writestr("CapsuleNotebook-x/CapsuleNotebook-x_executed.ipynb", "{}")
        zf.writestr("stray.ipynb", "{}")
    out = ba.extract_capsule(zpath, tmp_path / "CapsuleFolder-x")
    found = sorted(str(p.relative_to(out)) for p in out.rglob("*"))
    assert found == ["counts.csv", "sub", "sub/meta.tsv"]


def test_the_notebook_view_keeps_the_latest_cell_whole():
    # every output is under the 4,000-character cap, so only the budget shortens
    cells = [{"source": f"c{i}", "output": "x" * 3500} for i in range(6)]
    view = ba.view_notebook(cells, budget=12000)
    outputs = [view.split(f"### Output {i}:", 1)[1].split("### Cell", 1)[0] for i in range(6)]
    shortened = ["not shown" in o for o in outputs]
    # oldest first, only as many as the budget needs, never the latest
    assert shortened[0] and not shortened[5] and len(view) <= 12000
    assert shortened == sorted(shortened, reverse=True)


# --- BixBench's readings, as published -------------------------------------------

def test_the_mcq_parse_is_upstreams_strict_one():
    assert bw.xml_extract("analysis ...\n<answer>C</answer>") == "C"
    assert bw.xml_extract("<answer> C </answer>") == "Z"
    assert bw.xml_extract("no tag at all") == "Z"


def test_questions_to_mcq_keeps_the_key_and_the_refusal_letters():
    import random
    formatted, correct, refusal, shown = bw.questions_to_mcq(
        "Q?", ["0.2", "0.1", "0.3", "0.4"], True, random.Random(1))
    assert len(shown) == 5 and shown[ord(correct) - 65] == "0.2"
    assert shown[ord(refusal) - 65] == bw.REFUSE
    assert formatted.startswith("Q?\nA. ") and formatted.count("\n") == 6


def test_strict_numeric_rounds_to_the_keys_precision():
    assert bw.strict_numeric("The p-value is 0.00021", "0.0002") is True
    assert bw.strict_numeric("about 1.87e-5", "1.9E-05") is True
    assert bw.strict_numeric("3,827 genes", "3827") is True
    assert bw.strict_numeric("17", "11") is False
    assert bw.strict_numeric("109.5%", "109.53%") is False
    assert bw.strict_numeric(None, "11") is None


def test_identical_prompts_in_flight_share_one_request(tmp_path):
    """Two trajectories can hand the reader one prompt, and greedy decoding in
    different batches answered 37 of Llama's 57 such pairs differently, six with
    different letters; a rerun from the cache then gave both the reply kept, and
    a number moved. Sharing the request makes the first pass the rerun."""
    import asyncio

    async def run():
        client = bw.Client(tmp_path / "cache.jsonl", 4)
        calls = []

        async def fake(key, model, base, prompt, max_tokens):
            calls.append(prompt)
            await asyncio.sleep(0.01)
            client.cache[key] = f"reply to {prompt}"
            return client.cache[key]
        client._call = fake
        out = await asyncio.gather(*(client.ask("m", "b", "same", 10) for _ in range(5)),
                                   client.ask("m", "b", "other", 10))
        await client.http.aclose()
        return calls, out
    calls, out = asyncio.run(run())
    assert sorted(calls) == ["other", "same"] and len(set(out[:5])) == 1


def test_grade_parse_is_upstreams():
    assert bw.parse_grade("<grade> correct </grade>") is True
    assert bw.parse_grade("<grade>refused</grade>") is False
    assert bw.parse_grade("correct") is False


def test_the_nearest_option_is_read_on_a_relative_scale():
    """bix-1-q1's released options: the key 0.0002 among perturbations of it."""
    options = ["0.0002", "1.049135E-04", "0.0001", "2.874950E-04"]
    assert bw.nearest_is_key("0.00021", options) == 1.0
    assert bw.nearest_is_key("0.00027", options) == 0.0      # nearer the 2.87e-4 distractor
    assert bw.nearest_is_key("2e-4", options) == 1.0
    assert bw.nearest_is_key(None, options) == 0.0 and bw.nearest_is_key("unclear", options) == 0.0


def test_an_answer_equidistant_from_every_option_is_not_credited_to_the_key():
    """An answer of 0, or one of the other sign from every option, is at distance 1 from all
    four; a tie broken by position would credit the key, which sits first. Ties are split
    evenly, a random tie-break's expected score."""
    options = ["166", "190", "464", "137"]
    assert bw.nearest_is_key("0", options) == 0.25
    assert bw.nearest_is_key("-58", ["0.397", "0.18", "0.52", "0.73"]) == 0.25
    # a tie the key is not part of scores nothing (4 is as near 2 as 8 on a log scale),
    # and a clean win is unaffected
    assert bw.nearest_is_key("4", ["100", "2", "8", "50"]) == 0.0
    assert bw.nearest_is_key("170", options) == 1.0


def test_miss_bins_and_the_open_side():
    """tab:mechanism bins a newly credited answer by its distance from the key relative to
    the key, and asks whether it lies beyond an edge key where no distractor sits."""
    import bracketing as bk
    assert bk.miss_bin("104", "100") == "within 5%"
    assert bk.miss_bin("120", "100") == "5 to 25%"
    assert bk.miss_bin("-150", "-100") == "25 to 100%"
    assert bk.miss_bin("250", "100") == "over 100%"
    assert bk.miss_bin("0", "0") == "within 5%" and bk.miss_bin("1", "0") == "over 100%"
    # the key is options[0]: largest key, answer above it; smallest key, answer below it
    assert bk.open_side("10", ["8", "1", "2", "3"])
    assert not bk.open_side("5", ["8", "1", "2", "3"])
    assert bk.open_side("0.5", ["1", "2", "3", "4"])
    # a bracketed key has no open side
    assert not bk.open_side("100", ["2", "1", "3", "4"])


def test_the_reply_limit_census_counts_the_wrapper_loop():
    """A reasoning turn cut at the reply limit is a loop only if it repeats ldp's
    wrapper; a step with no reasoning call recorded is not a turn."""
    loop = "Thought: I need to look.\n\n" + (bw.WRAPPER_TAIL + "\n") * 6
    ts = [{"termination": "max_steps", "log": [{"finish": ["stop", "stop"], "reasoning": "a"},
                                               {"finish": ["length", "stop"], "reasoning": loop}]},
          {"termination": "submitted", "log": [{"finish": ["length", "stop"], "reasoning": "long " * 50},
                                               {"step": 2}]}]
    census = bw.reply_limit_census(ts)
    assert (census["turns"], census["capped"], census["loops"]) == (3, 2, 1)
    assert census["endings"] == {"max_steps": {"episodes": 1, "with_capped_turn": 1},
                                 "submitted": {"episodes": 1, "with_capped_turn": 1}}


# --- the numbers the paper prints ------------------------------------------------

def test_the_with_data_summary_reproduces_from_the_shipped_rows():
    shipped = sorted((ROOT / "results" / "agent_runs").glob("withdata_rows_*.json.gz"))
    target = ROOT / "results" / "bixbench_withdata.json"
    if not shipped or not target.exists():
        pytest.skip("with-data rows not shipped")
    rows = [r for p in shipped for r in json.loads(gzip.open(p, "rt").read())]
    assert bw.summarise(rows, None) == json.loads(target.read_text())


def test_the_shipped_runs_name_no_path_on_the_machine_they_ran_on():
    """The bundle is for a double-blind venue. Every trajectory's settings once
    recorded the absolute paths it ran with, and a path names its owner."""
    shipped = sorted((ROOT / "results" / "agent_runs").glob("*.gz"))
    if not shipped:
        pytest.skip("with-data runs not shipped")
    for path in shipped:
        text = gzip.open(path, "rt").read()
        # no path on the machines the runs were made on (make_public_export.py matches them
        # by hash); "/mnt/data" and its like are paths an agent guessed at in its own code
        import make_public_export as mpe
        hosts = [d for c, d in mpe.scan_text(path.name, text, {}, narration=False)[0] if c == "local path"]
        assert not hosts, (path.name, hosts)
        if path.name.startswith(("trajectories_", "superseded_")):
            # a path the agent wrote in its own code ("/home/user/...") is its own
            # invention; what must not ship is a path the run was *given*
            for line in text.splitlines():
                settings = json.loads(line)["settings"]
                assert not any(isinstance(v, str) and v.startswith("/") for v in settings.values()), path.name
    assert bw.anonymous({"settings": {"items": str(ROOT / "data" / "x.jsonl"), "work": "/scratch/w",
                                      "seed": 1}})["settings"] == \
        {"items": "data/x.jsonl", "work": "<work>", "seed": 1}


def test_the_v10_grid_reproduces_from_the_shipped_dumps():
    import arm_intervals as ai
    import bixbench_v10_grid as vg
    from mcq_audit import rank_of
    doc = json.loads((ROOT / "results" / "bixbench_v10_grid.json").read_text())
    items = ROOT / "build" / "bixbench_v10.jsonl"
    if not items.exists():
        pytest.skip("build/bixbench_v10.jsonl not built; run: python3 bixbench_v10_items.py")
    rows = [json.loads(l) for l in open(items)]
    numeric = [rank_of([r["ideal"], *r["distractors"]]) is not None for r in rows]
    assert doc["level"] == max(doc["levels_by_model"].values())
    for entry in doc["rows"]:
        dump = Path(entry["dump"])
        dump = ai.resolve(str(dump if dump.is_absolute() else ROOT / dump))
        if not dump.exists():
            dump = ai.resolve(str(ROOT / "build" / "dumps" / Path(entry["dump"]).name.replace(".gz", "")))
        f_all, c_all = ai.load(dump, "file"), ai.load(dump, "clean")
        for part, keep in (("all", lambda i: True), ("numeric", lambda i: numeric[i]),
                           ("other", lambda i: not numeric[i])):
            got = vg.margin([r for r in f_all if keep(r["item"])], [r for r in c_all if keep(r["item"])],
                            2000, 0, doc["level"])
            assert got == entry[part], f"{entry['model']} {part}"


def test_wrong_step_on_bixbench_reproduces_its_difficulty_draws():
    import frontier_calibrated as fc
    doc = json.loads((ROOT / "results" / "bixbench_wrong_step.json").read_text())
    level = doc["level"]["level_for_95"]
    for path, arm, label in (("results/bixbench_numeric_validity.json", "wrong-step", "wrong step (Qwen2.5-14B)"),
                             ("results/bixbench_numeric_validity_llamaws.json", "wrong-step-llama",
                              "wrong step (Llama-3.1-8B)")):
        validity = json.loads((ROOT / path).read_text())
        for model, cell in doc["arms"][label]["difficulty"].items():
            point, drawn = fc.difficulty_draws(validity, model, arm, 4000, 20260919)
            assert abs(point - cell["points"]) < 1e-9
            assert np.allclose(fc.quantiles(drawn, level), cell["covering"])


def test_the_70b_class_solvers_reproduce_beside_the_frontier():
    import frontier_calibrated as fc
    doc = json.loads((ROOT / "results" / "frontier_calibrated.json").read_text())
    level = doc["calibration"]["level_for_95"]
    for name in ("qwen72b", "llama70b"):
        path = ROOT / "results" / f"frontier_validity_items_{name}.json"
        if not path.exists():
            pytest.skip(f"{path.name} not present")
        items = json.loads(path.read_text())
        for model in items["models"]:
            for arm in ("key-marginal", "key-marginal-near", "wrong-step"):
                point, drawn = fc.difficulty_draws(items, model, arm, 4000, 20260919)
                cell = doc["rows"][arm]["difficulty_large"][model]
                assert abs(point - cell["points"]) < 1e-9
                assert np.allclose(fc.quantiles(drawn, level), cell["covering"])
