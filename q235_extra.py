#!/usr/bin/env python3
r"""Four more runs of Qwen3-235B-A22B with the data, to tell power from mechanism in its failed hypotheses.

The pre-specified test's two runs of Qwen3-235B-A22B on v1.5 (``qwen3-235b|r0``, ``|r1``) fail H2 and H4: its
unchanged keys gain under $U$, and its moved keys' $U-P$ does not exclude zero. Its misses are fewer than the
other agents', so the moved keys' contrast rests on few newly accepted runs. These are four more runs made after
the test was analysed, by the same agent, protocol and settings (rollouts 2 to 5, with the data, on the 105
numeric questions, served on four H100s as the registered runs were, under a 10,800 s wall clock; an episode
the clock cuts counts as unanswered, and the report lists each with its kernel restarts). They are outside the
pre-specified test and change none of its verdicts: this reports the test's statistics
(``replication.test_set``) on the four new runs, and on all six pooled, beside the registered two, with the
unchanged keys split by what $U$ does to their rank (``miss_anatomy.unchanged_split``).

    python3 q235_extra.py pack       # results/agent_runs_q235_extra/answers_qwen3-235b.jsonl.gz, from build/
    python3 q235_extra.py analyse    # results/q235_extra.json
"""
import argparse
import gzip
import json
import pathlib
from collections import defaultdict

import numpy as np

import bixbench_withdata as bw
import miss_anatomy as ma
import replication as rp
from answer_numbers import graded

ROOT = pathlib.Path(__file__).resolve().parent
BUILD = ROOT / "build" / "agent_runs_q235_extra"
PACKED = ROOT / "results" / "agent_runs_q235_extra"
OUT = ROOT / "results" / "q235_extra.json"
SEED = 20260928


def trajectories():
    paths = sorted(BUILD.glob("*/*/*.json"))
    if paths:
        return [json.loads(p.read_text()) for p in paths]
    out = []
    for path in sorted(PACKED.glob("answers_*.jsonl.gz")):
        with gzip.open(path, "rt") as fh:
            out.extend(json.loads(line) for line in fh)
    return out


# the packed fields, with each episode's kernel restarts and seconds, which say why the wall clock cut an episode
FIELDS = rp.PACK_FIELDS + ("kernel_restarts", "seconds")


def pack(_args):
    rows = sorted(({**{k: t[k] for k in FIELDS if k in t}, **rp.turn_counts(t)} for t in trajectories()),
                  key=lambda r: (r["condition"], r["question_id"], r["rollout"]))
    PACKED.mkdir(exist_ok=True)
    path = PACKED / "answers_qwen3-235b.jsonl.gz"
    with gzip.GzipFile(filename="", mode="wb", mtime=0, fileobj=path.open("wb")) as fh:
        fh.write("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows).encode())
    print(f"{len(rows)} trajectories to {path.relative_to(ROOT)}")


def within_5pct(runs, sets):
    """Each run's share of numeric questions whose submitted number, the last in the answer, is within 5% of the key."""
    return {name: 100.0 * float(np.mean([graded(a, sets[q]["released"][0], 0.05) if a else 0.0
                                         for q, a in t if q in sets]))
            for name, t in runs.items()}


def analyse(_args):
    rng = np.random.default_rng(SEED)
    sets = bw.option_sets()
    items = {json.loads(l)["question_id"]: json.loads(l) for l in open(ROOT / "data" / "bixbench.jsonl")}
    capsule = {q: it["capsule_uuid"] for q, it in items.items()}
    new = defaultdict(list)
    for t in trajectories():
        if t["condition"] == "data":
            new[f"{bw.run_key(t)}|r{t['rollout']}"].append((t["question_id"], t.get("answer")))
    registered = {n: c["data"] for n, c in rp.d2_runs().items() if n.startswith("qwen3-235b|") and c.get("data")}
    assert not set(new) & set(registered), "the new runs must not reuse a registered rollout"
    report = {"not_registered": True, "seed": SEED,
              "n_trajectories": {n: len(t) for n, t in new.items()},
              "terminations": {}, "within_5pct": within_5pct({**registered, **new}, sets)}
    for t in trajectories():
        key = f"{bw.run_key(t)}|r{t['rollout']}"
        report["terminations"].setdefault(key, defaultdict(int))[t.get("termination") or "none"] += 1
    # episodes the wall clock cut: scored as unanswered, as every episode without an answer is
    report["cut_by_wallclock"] = sorted(
        ({"run": f"{bw.run_key(t)}|r{t['rollout']}", "question_id": t["question_id"], "steps": t.get("steps"),
          "kernel_restarts": t.get("kernel_restarts"), "seconds": t.get("seconds"), "answer": t.get("answer")}
         for t in trajectories() if t.get("termination") == "wallclock"), key=lambda e: e["run"])
    for label, runs in (("new", dict(new)), ("all six", {**registered, **new})):
        res = rp.test_set(runs, sets, capsule, rng, heavy=True)
        # the unchanged keys by what U does to their rank, as miss_anatomy.py splits them
        res["unchanged_split"] = ma.unchanged_split(runs, sets, capsule)
        report[label] = res
        f = lambda v: "--" if not v else f"{v['mean']:+.1f} [{v['lo']:+.1f},{v['hi']:+.1f}]"
        print(f"{label:8s} runs {sorted(runs)}: U-R moved {f(res['gain|moved to an edge'])}, unchanged "
              f"{f(res['gain|kept'])}, inward {f(res['gain|moved inward'])}; U-P moved "
              f"{f(res['repaired-placebo|moved to an edge'])}; newly accepted {res['newly_credited']['n_gained']}; "
              f"verdicts {res['verdicts']}")
    print("within 5%:", {k: round(v, 1) for k, v in report["within_5pct"].items()})
    OUT.write_text(json.dumps(report, indent=1, default=dict) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("pack").set_defaults(fn=pack)
    sub.add_parser("analyse").set_defaults(fn=analyse)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
