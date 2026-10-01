"""The same readers, on the option text of the science-agent benchmarks.

Everything else in this paper reads option *values*, which restricts it to the
numeric slice of each file: 105 of BixBench's 205 items, and none of DbQA,
SeqQA or ProtocolQA at all.  The channel argument never needed the options to be
numbers -- Gamma is defined over the option set, whatever the options are -- so
the hashed character n-gram reader from ``learned_probe_nonlinear`` runs on the
text directly, and this is where it runs.

The clean control has to change with it.  A synthetic numeric file is not a null
for a file of English sentences, so the control here resamples each item's k
options i.i.d. from the *file's own pool* of option strings -- every key and
every distractor it contains -- and then marks one of them uniformly as the key.
That control has the file's own text distribution, its own k, its own item and
fold counts, and Gamma exactly zero by construction, so a margin over it is a
margin over "options that look like this benchmark's and say nothing".

    /opt/conda/bin/python text_options.py --device cuda:2
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import random
import zlib

import numpy as np

import learned_probe_nonlinear as reader

RESULTS = pathlib.Path("results")
LABBENCH = pathlib.Path("data/external/labbench")

# Each file is grouped by the first of these fields that has enough distinct
# values. SeqQA and DbQA are built in subtasks of forty and ten, and an item's
# neighbours inside a subtask share a template, so holding out random items
# would let the reader learn the template on one half and apply it to the other.
# Where no field groups the file, items are binned by id and the row says so.
FILES = [
    ("BixBench", "data/bixbench.jsonl", ("capsule_uuid",)),
    ("LAB-Bench LitQA2", str(LABBENCH / "LitQA2.jsonl"), ("subtask", "sources")),
    ("LAB-Bench TableQA", str(LABBENCH / "TableQA.jsonl"), ("subtask", "source")),
    ("LAB-Bench FigQA", str(LABBENCH / "FigQA.jsonl"), ("subtask", "source")),
    ("LAB-Bench SuppQA", str(LABBENCH / "SuppQA.jsonl"), ("subtask", "source")),
    ("LAB-Bench DbQA", str(LABBENCH / "DbQA.jsonl"), ("subtask", "source")),
    ("LAB-Bench SeqQA", str(LABBENCH / "SeqQA.jsonl"), ("subtask", "source")),
    ("LAB-Bench ProtocolQA", str(LABBENCH / "ProtocolQA.jsonl"), ("subtask", "source")),
    ("LAB-Bench CloningScenarios", str(LABBENCH / "CloningScenarios.jsonl"),
     ("subtask", "source")),
]
MIN_CLUSTERS = 10
N_FOLDS = 20


def fold_of(label: str, folds: int) -> str:
    """A deterministic fold for a group label, so held-out groups stay whole.

    These files run from one source for every item (DbQA) to one source per item
    (LitQA2: 103 distinct over 106). Leaving out one group at a time is what the
    rest of the paper does, but with a group per item that is leave-one-item-out,
    which Appendix A records as the estimator trap that returns exactly zero.
    Binning the *groups* into at most twenty folds keeps every group whole inside
    a fold and bounds the number of fits.
    """
    return f"fold{zlib.crc32(str(label).encode('utf-8')) % folds}"


def _label(row, field):
    value = row.get(field)
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    return str(value)


def load(path, cluster_fields):
    """The modal-k slice of a file, grouped by the first field that groups it."""
    rows = [json.loads(line) for line in pathlib.Path(path).open()]
    counts = collections.Counter(1 + len(r["distractors"]) for r in rows)
    k = counts.most_common(1)[0][0]
    rows = [r for r in rows if 1 + len(r["distractors"]) == k]

    chosen, distinct = None, 0
    for field in cluster_fields:
        if field not in rows[0]:
            continue
        found = len({_label(r, field) for r in rows})
        if found >= MIN_CLUSTERS:
            chosen, distinct = field, found
            break
    items = []
    for r in rows:
        options = [str(r["ideal"])] + [str(d) for d in r["distractors"]]
        if len(set(options)) != k:
            continue
        group = _label(r, chosen) if chosen else str(r["id"])
        items.append({"options": options, "cluster": fold_of(group, N_FOLDS)})
    return items, k, chosen, distinct


def clean_like(items, k, seed):
    """Options resampled i.i.d. from the file's own pool; the key uniform among them."""
    pool = sorted({o for item in items for o in item["options"]})
    rng = random.Random(seed)
    out = []
    for item in items:
        for _ in range(40):
            drawn = rng.sample(pool, k)
            if len(set(drawn)) == k:
                break
        else:
            continue
        rng.shuffle(drawn)
        out.append({"options": drawn, "cluster": item["cluster"]})
    return out


def score(items, args, device):
    x, clusters = reader.build_matrix(items, "chars", args.context, args.dim, None)
    if x.shape[0] == 0:
        return None
    picks, _ = reader.cross_validate(x, clusters, args, device, args.seed)
    return picks


def run_file(name, path, cluster_fields, args, device) -> dict:
    items, k, chosen, distinct = load(
        path, () if args.ignore_groups else cluster_fields)
    if len(items) < 60:
        return {"skipped": f"only {len(items)} items at the modal k"}
    picks = score(items, args, device)
    observed = reader.accuracy(picks)
    controls = []
    for replicate in range(args.replicates):
        clean = clean_like(items, k, f"{args.seed}|{name}|{replicate}")
        clean_picks = score(clean, args, device)
        if clean_picks:
            controls.append(reader.accuracy(clean_picks))
    worst = max(controls) if controls else float("nan")
    draws = reader.bootstrap(picks, args.reps, args.seed)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {
        "n_items": len(items), "k": k, "chance": 100.0 / k,
        "fold_field": (f"{chosen} in at most {N_FOLDS} folds" if chosen
                       else f"item id in {N_FOLDS} folds (no field groups this file)"),
        "n_folds": len({i["cluster"] for i in items}),
        "distinct_in_field": distinct,
        "accuracy": 100.0 * observed,
        "accuracy_ci95": [100.0 * float(lo), 100.0 * float(hi)],
        "clean_worst": 100.0 * worst,
        "clean_mean": 100.0 * float(np.mean(controls)) if controls else float("nan"),
        "margin_over_clean": 100.0 * (observed - worst),
        "margin_ci95": [100.0 * (float(lo) - worst), 100.0 * (float(hi) - worst)]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--context", choices=("option", "set"), default="set")
    ap.add_argument("--scorer", choices=("linear", "mlp"), default="mlp")
    ap.add_argument("--dim", type=int, default=2048)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--l2", type=float, default=1e-4)
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--replicates", type=int, default=10)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=20260920)
    ap.add_argument("--only", action="append", default=None)
    ap.add_argument("--ignore-groups", action="store_true",
                    help="bin items by id instead of by the file's own grouping "
                         "field, which is what an audit that does not look for one "
                         "would do; SeqQA and DbQA are built in templated subtasks "
                         "and this is how much that is worth")
    ap.add_argument("--out", default="text_options.json")
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    target = RESULTS / args.out
    payload = json.loads(target.read_text()) if target.exists() else {}
    payload.setdefault("settings", {}).update(
        {"extractor": "chars", "scorer": args.scorer, "dim": args.dim})
    key = args.context + ("|groups ignored" if args.ignore_groups else "")
    for name, path, field in FILES:
        if args.only and name not in args.only:
            continue
        print(f"[{key}] {name}", flush=True)
        row = run_file(name, path, field, args, args.device)
        payload.setdefault(key, {})[name] = row
        print("   " + json.dumps(row), flush=True)
        target.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"wrote {target}")


if __name__ == "__main__":
    main()
