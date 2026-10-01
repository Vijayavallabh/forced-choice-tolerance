#!/usr/bin/env python3
r"""Rebuild an arm's result entry from the rollouts it dumped.

``agentic_probe.py`` keyed its results by model and cell and not by file, so the
same cell run on two benchmarks into one ``--out`` was one key and the second
run deleted the first. Qwen2.5-72B's two MMLU-Pro cells went that way, replaced
by its two BixBench cells. The key includes the file now, but the entries it
already ate have to come back.

Nothing was actually lost: ``--dump`` is per-file, and every field ``summarise``
produces is a function of the dumped rollouts -- the answer, the key, the option
count, the cluster, the turns, the tool flag and the prompt text the
second-smallest rate is read off. So this recomputes the entry exactly rather
than approximating it, using the probe's own ``summarise`` and the same seeds,
and marks the entry ``rebuilt_from_dump`` so nobody has to take that on trust.

    python3 rebuild_arms.py --dump build/dumps/qwen72b_mmlupro_neutral_notools.jsonl \
        --model Qwen/Qwen2.5-72B-Instruct --items build/mmlu_pro_matched_released.jsonl \
        --condition withheld --no-tools --out agentic_arms_qwen72b.json
"""
from __future__ import annotations

import argparse
import gzip
import json
import pathlib

import numpy as np

import agentic_probe

RESULTS = pathlib.Path("results")


def load(path, arm):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as fh:
        rows = [json.loads(line) for line in fh]
    return [r for r in rows if r.get("arm", "file") == arm]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--items", required=True)
    ap.add_argument("--condition", default="withheld")
    ap.add_argument("--no-tools", action="store_true")
    ap.add_argument("--argmax", action="store_true")
    ap.add_argument("--prompt", default="agent", choices=("agent", "bixbench"))
    ap.add_argument("--draws", type=int, default=2)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    payload = {"model": args.model, "items": args.items,
               "condition": args.condition,
               "readout": "argmax" if args.argmax else "generated",
               "prompt": args.prompt,
               "tools": (not args.no_tools) and not args.argmax,
               "draws": args.draws, "dump": args.dump,
               "rebuilt_from_dump": True}
    payload["file"] = agentic_probe.summarise(
        load(args.dump, "file"), args.bootstrap, args.seed)
    payload["clean"] = agentic_probe.summarise(
        load(args.dump, "clean"), args.bootstrap, args.seed + 1)
    payload["n_items"] = payload["file"]["n_rollouts"] // max(args.draws, 1)
    payload["margin_over_clean"] = (payload["file"]["margin_over_chance"]
                                    - payload["clean"]["margin_over_chance"])
    difference = [a - b for a, b in zip(payload["file"].pop("bootstrap_draws"),
                                        payload["clean"].pop("bootstrap_draws"))]
    payload["margin_over_clean_ci95"] = [
        100.0 * float(np.percentile(difference, 2.5)),
        100.0 * float(np.percentile(difference, 97.5))]

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / args.out
    merged = json.loads(out.read_text()) if out.exists() else {}
    key = (f"{args.model}|{pathlib.Path(args.items).stem}|{args.condition}"
           + ("|argmax" if args.argmax else "|notools" if args.no_tools else "")
           + ("|bixprompt" if args.prompt == "bixbench" else ""))
    if key in merged:
        raise SystemExit(f"{key} already exists in {out}; refusing to overwrite "
                         "a measured entry with a rebuilt one")
    merged[key] = payload
    out.write_text(json.dumps(merged, indent=1) + "\n")
    print(f"{key}\n  margin {payload['margin_over_clean']:+.2f} "
          f"{[round(v, 2) for v in payload['margin_over_clean_ci95']]}  "
          f"unparsed {payload['file']['unparsed']:.2f}/"
          f"{payload['clean']['unparsed']:.2f}\nwrote {out}")


if __name__ == "__main__":
    main()
