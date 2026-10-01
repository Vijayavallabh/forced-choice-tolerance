"""Two checks Table 2 needs before "the corner is empty" can be read as a fact
about operators rather than about this item set.

  --holdout   Two of the twenty subjects hold 282 of the 464 items. Drop both and
              re-read every arm on the remaining 182, so the comparison is not a
              statement about high-school and elementary mathematics.

  --collision WRONG STEP's writer is shown the question and never the key, but on
              114 of 635 items it produced the key anyway and the assembler
              dropped that value. Splitting the arm on whether that happened
              separates "the question determines the key" from "this writer knew
              the answer": if the leak is carried only by the items where the
              writer hit the key, the claim is about the writer, not the question.

Both write subsetted arms to build/ and then run the same two readers the main
table uses, so nothing about the estimator changes between the table and these
rows.

    /opt/conda/bin/python frontier_robustness.py --holdout --collision
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import subprocess
import sys

BUILD = pathlib.Path("build")
RESULTS = pathlib.Path("results")
PY = sys.executable

ARMS = ("released_matched", "key_marginal", "key_marginal_near", "wrong_step",
        "rank_uniform", "exchangeable", "imitation")
DOMINANT = ("high_school_mathematics", "elementary_mathematics")


def read(path) -> list:
    return [json.loads(line) for line in pathlib.Path(path).open()]


def write(path, rows) -> pathlib.Path:
    path = pathlib.Path(path)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return path


def run_readers(files, labels, tag, n_options=4) -> dict:
    """The two readers Table 2 reports, on whatever arms are handed to them."""
    out = {}
    for script, key, extra in (("learned_probe.py", "set", ["--cluster-field", "cluster"]),
                               ("key_identity.py", "one_option", [])):
        target = RESULTS / f"frontier_{tag}_{key}.json"
        cmd = [PY, script, "--n-options", str(n_options),
               "--output", str(target)] + extra
        for path, label in zip(files, labels):
            cmd += ["--jsonl", str(path), "--label", label]
        print("  " + " ".join(cmd[1:4]) + f" ... -> {target.name}", flush=True)
        done = subprocess.run(cmd, capture_output=True, text=True)
        if done.returncode:
            raise SystemExit(f"{script} failed:\n{done.stdout[-3000:]}\n{done.stderr[-3000:]}")
        out[key] = json.loads(target.read_text())
    return out


def do_holdout() -> dict:
    files, labels, counts = [], [], {}
    for arm in ARMS:
        rows = read(BUILD / f"mmlu_frontier_{arm}_aligned.jsonl")
        kept = [r for r in rows if r["cluster"] not in DOMINANT]
        files.append(write(BUILD / f"mmlu_frontier_{arm}_holdout.jsonl", kept))
        labels.append(arm)
        counts[arm] = len(kept)
    subjects = {r["cluster"] for r in read(files[0])}
    print(f"holdout: {counts[ARMS[0]]} items in {len(subjects)} subjects "
          f"(dropped {', '.join(DOMINANT)})", flush=True)
    payload = run_readers(files, labels, "holdout")
    payload["n_items"] = counts[ARMS[0]]
    payload["n_subjects"] = len(subjects)
    payload["dropped"] = list(DOMINANT)
    return payload


def do_collision() -> dict:
    """Split WRONG STEP on whether the writer produced the key for that item."""
    generations = json.loads((BUILD / "mmlu_frontier_wrong_step_generations.json"
                              ).read_text()) if (
        BUILD / "mmlu_frontier_wrong_step_generations.json").exists() else None
    arm = read(BUILD / "mmlu_frontier_wrong_step_aligned.jsonl")
    released = read(BUILD / "mmlu_frontier_released_matched_aligned.jsonl")
    strict = {(r["question"], r["ideal"]) for r in
              read(BUILD / "mmlu_frontier_wrong_step_strict_nofilter.jsonl")}
    # An item is "clean" when it also survives the drop-item arm, which keeps only
    # items where no value had to be removed.
    groups = collections.defaultdict(list)
    for a, r in zip(arm, released):
        assert a["question"] == r["question"] and a["ideal"] == r["ideal"]
        groups["clean" if (a["question"], a["ideal"]) in strict else "collided"].append((a, r))
    payload = {"n_clean": len(groups["clean"]), "n_collided": len(groups["collided"])}
    print(f"collision split: {payload['n_clean']} clean, "
          f"{payload['n_collided']} where the writer produced the key", flush=True)
    for name, pairs in groups.items():
        if len(pairs) < 40:
            payload[name] = {"skipped": "fewer than 40 items"}
            continue
        files = [write(BUILD / f"mmlu_frontier_wrong_step_{name}.jsonl", [a for a, _ in pairs]),
                 write(BUILD / f"mmlu_frontier_released_{name}.jsonl", [r for _, r in pairs])]
        payload[name] = run_readers(files, ["wrong_step", "released"], f"wrongstep_{name}")
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--holdout", action="store_true")
    ap.add_argument("--collision", action="store_true")
    ap.add_argument("--out", default="frontier_robustness.json")
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    target = RESULTS / args.out
    payload = json.loads(target.read_text()) if target.exists() else {}
    if args.holdout:
        payload["subject_holdout"] = do_holdout()
    if args.collision:
        payload["collision_split"] = do_collision()
    target.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"wrote {target}")


if __name__ == "__main__":
    main()
