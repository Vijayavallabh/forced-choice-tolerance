#!/usr/bin/env python3
r"""A clean control for a whole released file, options and all, not just its numbers.

\S4's agent runs on BixBench's numeric slice -- $105$ of $205$ items -- because
the clean control it is read against is built by redrawing numeric \emph{values}.
The other hundred items have option sets too, and $\Gamma$ is defined over an
option set whatever the options are, so the restriction is the control's and not
the quantity's.

This builds the control \texttt{text\_options.py} uses, for any file: each item's
$k$ options are drawn i.i.d.\ from the \emph{file's own pool} of option strings
and the key is then uniform among them. Every string a released item could show
is a string this control can show, so its vocabulary, its length distribution
and its register are the file's; what it does not have is any relation between
the key and the set it sits in, which makes $\Gamma=0$ by construction.

    python3 build_set_control.py --items build/bixbench_v15.jsonl \
        --out build/bixbench_all_clean.jsonl
"""
import argparse
import json
import pathlib
import random


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20260920)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.items)]
    pool = sorted({str(o) for r in rows
                   for o in [r["ideal"]] + list(r["distractors"])})
    rng = random.Random(args.seed)
    out, skipped = [], 0
    for row in rows:
        k = 1 + len(row["distractors"])
        for _ in range(40):
            drawn = rng.sample(pool, k)
            if len(set(drawn)) == k:
                break
        else:
            skipped += 1
            continue
        out.append({"ideal": drawn[0], "distractors": drawn[1:],
                    "question": row.get("question", ""),
                    "cluster": row.get("cluster") or row.get("capsule_uuid")})
    pathlib.Path(args.out).write_text("".join(json.dumps(r) + "\n" for r in out))
    print(f"{len(out)} control items from a pool of {len(pool)} option strings "
          f"({skipped} skipped)\nwrote {args.out}")


if __name__ == "__main__":
    main()
