#!/usr/bin/env python3
r"""Cut every repair operator to the items all of them could produce.

The operators have different domains: an exchangeable rewriting does not exist
for every item, the generator sometimes writes fewer than three usable values,
and a subject with few keys cannot lend three. Comparing operators across
different item sets would confound the operator with its domain, so each arm is
cut here to the intersection, keyed on the item's own key and question text.

    python3 frontier_align.py
"""
import argparse
import json
from pathlib import Path

ARMS = ("released_matched", "imitation", "wrong_step", "key_marginal_near",
        "key_marginal", "rank_uniform", "exchangeable")


def read(path):
    return [json.loads(line) for line in
            Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def signature(row):
    return (str(row["ideal"]).strip(), str(row.get("question", "")).strip())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--indir", default="build")
    ap.add_argument("--prefix", default="mmlu_frontier_")
    ap.add_argument("--suffix", default="_aligned")
    ap.add_argument("--arm", action="append", default=None,
                    help="restrict the intersection to these arms; repeat. "
                         "The robustness arms are aligned separately so that "
                         "adding a second generator does not shrink Table 2.")
    ap.add_argument("--report", default="results/frontier_alignment.json")
    args = ap.parse_args()

    loaded = {}
    for arm in (args.arm or ARMS):
        path = Path(args.indir) / f"{args.prefix}{arm}.jsonl"
        loaded[arm] = read(path)
        print(f"{arm:<20} {len(loaded[arm]):>4} rows  ({path})")

    common = None
    for arm, rows in loaded.items():
        keys = {signature(r) for r in rows}
        common = keys if common is None else (common & keys)
    print(f"\n{len(common)} items every operator produced")

    written = {}
    for arm, rows in loaded.items():
        seen, kept = set(), []
        for row in rows:
            key = signature(row)
            if key in common and key not in seen:
                seen.add(key)
                kept.append(row)
        kept.sort(key=signature)
        path = Path(args.indir) / f"{args.prefix}{arm}{args.suffix}.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in kept), encoding="utf-8")
        written[arm] = str(path)
        print(f"  wrote {path}: {len(kept)} rows")

    sizes = {arm: len(read(path)) for arm, path in written.items()}
    if len(set(sizes.values())) != 1:
        raise SystemExit(f"arms did not align: {sizes}")

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(
        {"n_common": len(common), "arms": written,
         "n_before": {a: len(r) for a, r in loaded.items()}}, indent=2), encoding="utf-8")
    print(f"\nwrote {args.report}")


if __name__ == "__main__":
    main()
