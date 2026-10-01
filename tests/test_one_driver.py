"""``bixbench_agent.py``: one driver per run, one run per episode, and a filtered episode recorded once.

Two failures of DeepSeek-V4-Pro's first with-data run, each tested where it happened:

  * two drivers ran the model at once on one ``--out``; each looked for an episode's output only when its
    coroutine started and wrote with a rename, so an episode both finished kept whichever finished last.
    A driver now holds its run's directories under ``--out`` and ``--work`` and a second is refused while
    it lives; each episode claims its output before it runs; a finished result is never replaced;
  * Azure AI Foundry refused three requests with its content filter in a shape ``openai_api`` did not
    know, so each episode crashed without a record and was run again. It now ends, recorded with
    termination ``content_filter`` and no answer, and a later run of the driver leaves it as it is.
"""
import asyncio
import json
import os
import signal
import socket
import subprocess
import sys
import textwrap
import time
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import bixbench_agent as ba  # noqa: E402
import openai_api as oa  # noqa: E402
from test_openai_api import FOUNDRY_BODY  # noqa: E402

BASE = "http://127.0.0.1:9/openai"            # served as Azure through the test hook; nothing listens there


@pytest.fixture(autouse=True)
def release_locks():
    yield
    while ba.HELD:
        os.close(ba.HELD.pop())


def _args(tmp_path, **over):
    base = dict(model="DeepSeek-V4-Pro", protocol="react", out=tmp_path / "out", work=tmp_path / "work",
                base=BASE, max_steps=40, wallclock=3600, max_model_len=128000, chars_per_token=3.0,
                max_tokens=32768, temperature=1.0, view_budget=200000, cpus="4", memory="24g",
                cell_timeout=600, extra_body=None, reasoning_effort=None, served_as=None, keep_workdirs=False,
                condition=["nodata"], tool_format=None, items=str(tmp_path / "items.jsonl"), only=None,
                only_file=None, limit=0, rollouts=1, first_rollout=0, seed=20260923, concurrency=2)
    base.update(over)
    return types.SimpleNamespace(**base)


def _other_process():
    """A live process that is not this one, and how a lock or a claim would name it."""
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    for _ in range(100):
        found = ba._process_start(proc.pid)
        if found is not None:
            break
        time.sleep(0.01)
    return proc, {"pid": proc.pid, "host": socket.gethostname(), "start": found[1]}


def _stop(proc):
    proc.kill()
    proc.wait()


# ---------------------------------------------------------------- a process is alive, or it is not

def test_a_pid_counts_as_alive_only_while_the_process_that_took_it_runs():
    me = ba.whoami()
    assert me["pid"] == os.getpid() and me["start"] is not None
    assert ba.alive(me)
    assert not ba.alive({**me, "start": me["start"] + 1}), "a pid reused by another process is not the holder"
    proc, other = _other_process()
    assert ba.alive(other)
    _stop(proc)
    assert not ba.alive(other)
    assert ba.alive({**other, "host": "some-other-host"}), "on another host a holder cannot be checked"
    assert not ba.alive(None) and not ba.alive({"pid": "x"})


# ---------------------------------------------------------------- a finished result is never replaced

def test_a_finished_result_is_never_replaced(tmp_path, capsys):
    out = tmp_path / "run" / "data" / "q1__r0.json"
    first = {"question_id": "q1", "answer": "3", "termination": "submitted"}
    assert ba.save_result(out, first) == first
    kept = ba.save_result(out, {"question_id": "q1", "answer": "7", "termination": "submitted"})
    assert kept == first and json.loads(out.read_text()) == first
    aside = [p for p in out.parent.iterdir() if p.name.startswith("q1__r0.json.duplicate-")]
    assert len(aside) == 1 and json.loads(aside[0].read_text())["answer"] == "7"
    # nothing a run's readers glob for is left beside it: only the result is a .json
    assert sorted(p.name for p in out.parent.glob("*.json")) == ["q1__r0.json"]
    assert not [p for p in out.parent.iterdir() if p.name.endswith(".tmp")]
    assert "already there and is kept" in capsys.readouterr().out


# ---------------------------------------------------------------- an episode is claimed once

def test_an_episode_is_claimed_once_and_a_dead_claim_is_taken_over(tmp_path):
    out = tmp_path / "run" / "data" / "q1__r0.json"
    claim = ba.claim_output(out)
    assert claim == ba.claim_path(out) and ba._mine(json.loads(claim.read_text()))
    assert ba.claim_output(out) is None, "claimed twice by one process"
    ba.release_claim(claim)
    assert not claim.exists()

    proc, other = _other_process()
    try:
        claim.write_text(json.dumps(other))
        assert ba.claim_output(out) is None, "a live process's claim was taken"
        ba.release_claim(claim)
        assert json.loads(claim.read_text()) == other, "another process's claim was removed"
    finally:
        _stop(proc)
    taken = ba.claim_output(out)                 # its process is gone: the claim is taken over
    assert taken == claim and ba._mine(json.loads(claim.read_text()))
    ba.release_claim(taken)

    claim.write_text("{not json")                # unreadable: left alone while it is new ...
    assert ba.claim_output(out) is None
    old = time.time() - 120
    os.utime(claim, (old, old))                  # ... and taken over once it is old
    assert ba.claim_output(out) == claim


# ---------------------------------------------------------------- one driver per run

HOLDER = textwrap.dedent("""
    import sys, time, types
    sys.path.insert(0, sys.argv[1])
    import bixbench_agent as ba
    ba.hold_run(types.SimpleNamespace(out=sys.argv[2], work=sys.argv[3], model="DeepSeek-V4-Pro", protocol="react"))
    print("held", flush=True)
    time.sleep(120)
""")


def _driver(out, work):
    proc = subprocess.Popen([sys.executable, "-c", HOLDER, str(ROOT), str(out), str(work)],
                            stdout=subprocess.PIPE, text=True)
    assert proc.stdout.readline().strip() == "held"
    return proc


def test_a_second_driver_on_a_held_run_is_refused_until_the_first_is_gone(tmp_path):
    out, work = tmp_path / "out", tmp_path / "work"
    first = _driver(out, work)
    try:
        for o, w in ((out, work), (out, tmp_path / "work-b"), (tmp_path / "out-b", work)):
            with pytest.raises(ba.Busy, match="held by a live driver"):
                ba.hold_run(_args(tmp_path, out=o, work=w))
        assert not ba.HELD
        # another model's run may share --out and --work, as the current agents' runs did
        assert len(ba.hold_run(_args(tmp_path, model="gpt-6-luna"))) == 2
        record = json.loads((out / "DeepSeek-V4-Pro-react" / ba.LOCK_NAME).read_text())
        assert record["pid"] == first.pid and "since" in record
    finally:
        first.send_signal(signal.SIGKILL)
        first.wait()
    # a driver refused on its second directory left no record naming it in its first
    assert not (tmp_path / "out-b" / "DeepSeek-V4-Pro-react" / ba.LOCK_NAME).read_text().strip()
    fds = ba.hold_run(_args(tmp_path))           # the first driver died: its lock is taken over
    record = json.loads((out / "DeepSeek-V4-Pro-react" / ba.LOCK_NAME).read_text())
    assert len(fds) == 2 and ba._mine(record) and record["argv"]
    assert json.loads((work / "DeepSeek-V4-Pro-react" / ba.LOCK_NAME).read_text()) == record


def test_a_lock_whose_live_process_does_not_hold_it_still_refuses(tmp_path):
    proc, other = _other_process()
    try:
        where = tmp_path / "out" / "DeepSeek-V4-Pro-react"
        where.mkdir(parents=True)
        (where / ba.LOCK_NAME).write_text(json.dumps({**other, "argv": ["bixbench_agent.py"]}))
        with pytest.raises(ba.Busy, match="live process"):
            ba.hold_run(_args(tmp_path))
    finally:
        _stop(proc)
    assert len(ba.hold_run(_args(tmp_path))) == 2


def test_the_command_line_refuses_a_second_driver(tmp_path):
    out, work = tmp_path / "out", tmp_path / "work"
    first = _driver(out, work)
    try:
        # --only names no question, so even a driver that got through would run nothing
        second = subprocess.run([sys.executable, str(ROOT / "bixbench_agent.py"), "--model", "DeepSeek-V4-Pro",
                                 "--base", BASE, "--protocol", "react", "--condition", "data", "--only", "none",
                                 "--out", str(out), "--work", str(tmp_path / "work-b")],
                                capture_output=True, text=True, timeout=120,
                                env={**os.environ, "AGENTICLS_OPENAI_TEST_BASE": BASE,
                                     "AGENTICLS_OPENAI_TEST_AUTH": "azure"})
    finally:
        first.send_signal(signal.SIGKILL)
        first.wait()
    assert second.returncode != 0 and "held by a live driver" in second.stderr
    assert not (tmp_path / "work-b" / "DeepSeek-V4-Pro-react" / "data").exists()


# ---------------------------------------------------------------- a filtered episode, recorded once

class _NoBox:
    """A sandbox that is never needed: the episode is refused before it runs any code."""

    def __init__(self, *args, **kwargs):
        self.restarts, self.submitted = 0, None

    async def run(self, code):
        raise AssertionError("code ran")

    async def restart(self):
        raise AssertionError("the kernel was restarted")

    async def kill(self):
        pass


def _filtering(monkeypatch, tmp_path):
    """Azure AI Foundry refusing every request with its content filter, as it refused bix-1-q2's."""
    import httpx
    monkeypatch.setenv("AGENTICLS_OPENAI_TEST_BASE", BASE)
    monkeypatch.setenv("AGENTICLS_OPENAI_TEST_AUTH", "azure")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "azure-key-0123456789")
    monkeypatch.setenv("AGENTICLS_OPENAI_BUDGET_USD", "5")
    monkeypatch.setenv("AGENTICLS_OPENAI_PRICES", "2,0.2,8")
    monkeypatch.setenv("AGENTICLS_OPENAI_USAGE_LOG", str(tmp_path / "usage.jsonl"))
    monkeypatch.setattr(ba, "Sandbox", _NoBox)
    monkeypatch.setattr(ba, "REPLICA_DOWN", set())
    seen = []

    def handle(request):
        seen.append(json.loads(request.content))
        return httpx.Response(400, text=FOUNDRY_BODY, headers={"content-type": "application/json"})
    return httpx.MockTransport(handle), seen


ITEM = {"question_id": "bix-1-q2", "question": "How many genes?", "capsule_uuid": "c1", "ideal": "1",
        "data_folder": "d.zip", "eval_mode": "range_verifier"}


def test_a_filtered_episode_ends_recorded_with_no_answer(tmp_path, monkeypatch):
    import httpx
    transport, seen = _filtering(monkeypatch, tmp_path)
    args = _args(tmp_path)

    async def go():
        async with httpx.AsyncClient(transport=transport) as client:
            return await ba.episode_react(ITEM, "nodata", 0, args, ba.fhda_prompts(), client, asyncio.Semaphore(1))
    res = asyncio.run(go())
    assert res["termination"] == "content_filter" and res["answer"] is None and res["raw_submit"] is None
    assert res["steps"] == 1 and "content_filter" in res["log"][0]["error"]
    assert len(seen) == 1 and seen[0]["tool_choice"] == "none", "the refused request was sent again"
    saved = ba.output_path(args, "bix-1-q2", "nodata", 0)
    assert json.loads(saved.read_text())["termination"] == "content_filter"
    ledger = [json.loads(l) for l in (tmp_path / "usage.jsonl").read_text().splitlines()]
    assert [r.get("content_filter") for r in ledger] == [True] and ledger[0]["cost_usd"] == 0.0


def test_the_driver_records_a_filtered_episode_once_and_never_runs_it_again(tmp_path, monkeypatch, capsys):
    import httpx
    transport, seen = _filtering(monkeypatch, tmp_path)
    real = httpx.AsyncClient
    monkeypatch.setattr(ba.httpx, "AsyncClient", lambda **kw: real(transport=transport, **kw))
    (tmp_path / "items.jsonl").write_text(json.dumps(ITEM) + "\n")
    args = _args(tmp_path)
    asyncio.run(ba.main_async(args))
    printed = capsys.readouterr().out
    assert "done DeepSeek-V4-Pro: 1 episodes, 0 submitted, 1 ended by the content filter" in printed
    assert "Rejected" not in printed and len(seen) == 1
    saved = ba.output_path(args, "bix-1-q2", "nodata", 0)
    first = saved.read_text()
    assert json.loads(first)["termination"] == "content_filter"
    assert not ba.claim_path(saved).exists(), "the claim outlived the episode"

    asyncio.run(ba.main_async(args))             # the driver again: the recorded episode is not rerun
    assert len(seen) == 1 and saved.read_text() == first
    assert "1 ended by the content filter" in capsys.readouterr().out


def test_an_episode_a_live_process_has_claimed_is_left_to_it(tmp_path, monkeypatch, capsys):
    import httpx
    transport, seen = _filtering(monkeypatch, tmp_path)
    real = httpx.AsyncClient
    monkeypatch.setattr(ba.httpx, "AsyncClient", lambda **kw: real(transport=transport, **kw))
    (tmp_path / "items.jsonl").write_text(json.dumps(ITEM) + "\n")
    args = _args(tmp_path)
    saved = ba.output_path(args, "bix-1-q2", "nodata", 0)
    saved.parent.mkdir(parents=True)
    proc, other = _other_process()
    try:
        ba.claim_path(saved).write_text(json.dumps(other))
        asyncio.run(ba.main_async(args))
    finally:
        _stop(proc)
    printed = capsys.readouterr().out
    assert "held by a live process; left to it" in printed and "1 left to another live process" in printed
    assert not seen and not saved.exists()
    assert json.loads(ba.claim_path(saved).read_text()) == other
