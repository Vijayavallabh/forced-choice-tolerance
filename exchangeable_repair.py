"""Repair by exchangeability rather than by rank, and close the channels rank misses.

Section 2's guarantee is about *ranks*: make the key's rank uniform under an
ordering and no solver whose preference is a distribution over that ordering's
ranks can collect anything. Appendix "a learned solver" then measures the hole
in it. A learner given features that are not ranks -- how large an option is
against the largest in its set, how far it sits from the rest -- scores
**better on the repaired file than on the released one** (41.2% against 28.4%
on MMLU). The repair regenerates distractors by perturbing the key, so the key
is the centre of the option set by construction, and no uniformising of its
*rank* touches that.

The fix is to stop repairing coordinates one at a time and repair the thing they
are all functions of: make the key a uniformly random member of its own option
set. Draw ``k`` i.i.d. offsets and a slot, and read the offsets relative to the
slot's::

    z_0 .. z_{k-1} i.i.d. N(0,1)      m ~ Uniform{0..k-1}
    v_i = key * exp(sigma (z_i - z_m))            (multiplicative, so v_m = key)
    v_i = key + sigma (z_i - z_m)                 (additive, when signs differ)

Then the key's rank is ``#{i : z_i < z_m}``, which is uniform, and the same
argument runs for *every* equivariant statistic at once: the key's relative
magnitude is ``u_m / max(u)`` for a uniformly drawn ``m``, so it is distributed
exactly like a distractor's. Nothing here is searched for, so there is no target
distribution, no reachability frontier, no calibration pass and no hit rate.

**Rejection has to be symmetric or it puts the leak back.** Renderings collapse
(two options round to the same text), so draws must be retried -- and the naive
retry is wrong in a way that is easy to miss and easy to measure. Whether a draw
survives depends on which slot the key took: for a small integer key, the slots
that put distractors *below* it round into each other and get rejected, so the
key drifts to the bottom. Drawing one slot and retrying until it works gives a
value-rank histogram of 32.4%, 21.6%, 23.4%, 22.5% on MMLU -- seven points of
plug-in credit on the coordinate this closes by proof. ``naive_row`` keeps that
version and every run records its histogram, so the comparison is a number in
the report rather than an argument. A draw is therefore accepted only when
**all k slots** are legal, which makes acceptance a function of the offsets alone; the
slot is then chosen uniformly among k sets that are all known to work, and the
histogram is uniform by construction rather than by luck.

**Exchangeability is strictly harder than rank-uniformity, and the file says
so.** ``['3', '1', '2', '4']`` cannot be made exchangeable in its own written
style at all: to put the key at the top you need three distinct positive
integers below 3. Roughly a tenth of MMLU's numeric items are like this, and
they are reported rather than forced -- ``--only-feasible`` writes the matched
subset so the comparison against another repair is on the same items.

Rendering is the other cost, and it is a real edit to the file: the option texts
have to be exchangeable too, so every option is written in the *key's* style,
where the Section 5 repair copies the style of the option each distractor
replaces. The distractors also stop being perturbations of the answer -- they
are perturbations of a point the answer is itself a perturbation of. Whether a
benchmark will accept that is a judgement about the benchmark, not about the
guarantee.

    python3 exchangeable_repair.py --jsonl build/mmlu.jsonl --n-options 4 \
        --output build/mmlu_exchangeable.jsonl \
        --report results/exchangeable_mmlu.json
"""
import argparse
import gzip
import io
import json
import math
import random
import statistics
import sys
from collections import Counter
from pathlib import Path

from mcq_audit import (build_items, format_like, parse_number, rank_of,
                       read_rows, surface_channels)
from mcq_audit import ORDERINGS as AUDIT_ORDERINGS

# E|X - Y| for X, Y i.i.d. standard normal. The offsets enter as *differences*,
# so dividing the item's observed mean gap by this is what makes the regenerated
# spread match the released one rather than sqrt(2) times it.
MEAN_ABS_DIFF = 2 / math.sqrt(math.pi)
DEFAULT_LOG_SIGMA = 0.25
ATTEMPTS = 60
MULTIPLICATIVE_ATTEMPTS = 30


def spread(key, values, log=True):
    """The sigma that reproduces this item's observed spread."""
    if log:
        gaps = [abs(math.log(abs(v / key))) for v in values[1:]
                if v and key and v / key > 0]
    else:
        gaps = [abs(v - key) for v in values[1:] if abs(v - key) > 0]
    if not gaps:
        return DEFAULT_LOG_SIGMA if log else (abs(key) * 0.1 or 0.1)
    return max(statistics.fmean(gaps) / MEAN_ABS_DIFF, 1e-9)


def _legal(rendered, key, k, same_sign, slot):
    parsed = [parse_number(t) for t in rendered]
    if any(v is None for v in parsed) or len(set(parsed)) < k:
        return None
    if same_sign and any((v > 0) != (key > 0) for v in parsed):
        return None
    if parsed[slot] != key:
        return None            # the key's own text must survive the round trip
    return parsed


def exchangeable_row(options, rng, attempts=ATTEMPTS):
    """Regenerate the option set so the key is a uniform draw from it.

    Returns ``(options, slot)`` with ``options[0]`` still the key -- the file's
    convention -- and ``slot`` the draw the key was assigned, or ``None`` when
    the item admits no exchangeable rewriting in its own written style.
    """
    k = len(options)
    values = [parse_number(o) for o in options]
    if any(v is None for v in values) or len(set(values)) < k:
        return None
    key = values[0]
    same_sign = key != 0 and all((v > 0) == (key > 0) for v in values)
    template = str(options[0]).strip()

    for attempt in range(attempts):
        widen = 1.0 + 0.4 * (attempt % 20)
        multiplicative = same_sign and attempt < MULTIPLICATIVE_ATTEMPTS
        sigma = spread(key, values, log=multiplicative) * widen
        z = [rng.gauss(0.0, 1.0) for _ in range(k)]
        # Build the set the key would sit in under *every* slot, and take the
        # draw only if all k are legal. Acceptance then depends on z alone, so
        # the slot chosen below is uniform given the accepted draw.
        candidates = []
        for slot in range(k):
            if multiplicative:
                new = [key * math.exp(sigma * (x - z[slot])) for x in z]
            else:
                new = [key + sigma * (x - z[slot]) for x in z]
            rendered = [format_like(template, v) for v in new]
            if _legal(rendered, key, k, same_sign, slot) is None:
                candidates = None
                break
            candidates.append(rendered)
        if not candidates:
            continue
        slot = rng.randrange(k)
        rendered = candidates[slot]
        # The key keeps its released text; the round-trip check above is what
        # guarantees that text and rendered[slot] name the same number.
        return [options[0]] + [t for i, t in enumerate(rendered) if i != slot], slot
    return None


def naive_row(options, rng, attempts=ATTEMPTS):
    """The wrong retry, kept so the paper's number for it is regenerated.

    Draws the slot first and retries until *that* slot is legal. Acceptance then
    depends on which slot the key took, and the bias is not small: on MMLU the
    key's value rank comes out at roughly 31%, 27%, 23%, 18% instead of flat.
    ``exchangeable_row`` differs from this in one line -- it requires all k slots
    to be legal before choosing one -- and that line is the guarantee.
    """
    k = len(options)
    values = [parse_number(o) for o in options]
    if any(v is None for v in values) or len(set(values)) < k:
        return None
    key = values[0]
    same_sign = key != 0 and all((v > 0) == (key > 0) for v in values)
    template = str(options[0]).strip()
    for attempt in range(attempts):
        widen = 1.0 + 0.4 * (attempt % 20)
        multiplicative = same_sign and attempt < MULTIPLICATIVE_ATTEMPTS
        sigma = spread(key, values, log=multiplicative) * widen
        z = [rng.gauss(0.0, 1.0) for _ in range(k)]
        slot = rng.randrange(k)
        if multiplicative:
            new = [key * math.exp(sigma * (x - z[slot])) for x in z]
        else:
            new = [key + sigma * (x - z[slot]) for x in z]
        rendered = [format_like(template, v) for v in new]
        parsed = _legal(rendered, key, k, same_sign, slot)
        if parsed is None:
            continue
        return [options[0]] + [t for i, t in enumerate(rendered) if i != slot], slot
    return None


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffixes[-2:] == [".jsonl", ".gz"]:
        raw = gzip.GzipFile(filename="", mode="wb", mtime=0,
                            fileobj=path.open("wb"), compresslevel=9)
        handle = io.TextIOWrapper(raw, encoding="utf-8")
    else:
        handle = path.open("w", encoding="utf-8")
    with handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jsonl", type=Path, required=True)
    ap.add_argument("--key-field", default="ideal")
    ap.add_argument("--distractor-field", default="distractors")
    ap.add_argument("--question-field", default="question")
    ap.add_argument("--cluster-field", default="cluster")
    ap.add_argument("--n-options", type=int, default=None)
    ap.add_argument("--seed", type=int, default=20260920)
    ap.add_argument("--replicates", type=int, default=400,
                    help="cluster-bootstrap replicates for the audit of the result")
    ap.add_argument("--only-feasible", action="store_true",
                    help="write only the items an exchangeable rewriting exists "
                         "for, so another file can be restricted to the same rows")
    ap.add_argument("--restrict", nargs=2, action="append", metavar=("SRC", "OUT"),
                    default=[],
                    help="cut another file to the same rows and write it to OUT; "
                         "repeat per file. Comparing two repairs on different "
                         "item sets would confound the repair with the subset, "
                         "and a tenth of the items admit no exchangeable "
                         "rewriting, so the comparison is made on the matched "
                         "rows and this is what produces them")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--report", type=Path)
    args = ap.parse_args()

    rows = read_rows(args.jsonl)
    items, k = build_items(rows, args.key_field, args.distractor_field,
                           args.cluster_field, args.question_field, args.n_options)
    numeric = [it for it in items if it["rank"] is not None]
    if not numeric:
        raise SystemExit(f"no numeric items in {args.jsonl}")

    # The same construction with the wrong retry, measured first so the claim
    # that the all-slots rule is doing the work is a number and not an argument.
    naive_rng = random.Random(args.seed)
    naive = Counter()
    for it in numeric:
        got = naive_row(it["options"], naive_rng)
        if got is not None:
            rank = rank_of(got[0], 0)
            if rank is not None:
                naive[rank] += 1
    naive_total = sum(naive.values()) or 1
    naive_share = [naive[r] / naive_total for r in range(k)]

    rng = random.Random(args.seed)
    repaired = [dict(r) for r in rows]
    slots, feasible, infeasible = Counter(), [], []
    for it in numeric:
        result = exchangeable_row(it["options"], rng)
        if result is None:
            infeasible.append(it["row_index"])
            continue
        new_options, slot = result
        repaired[it["row_index"]][args.distractor_field] = new_options[1:]
        slots[slot] += 1
        feasible.append(it["row_index"])

    out_rows = ([repaired[i] for i in feasible] if args.only_feasible else repaired)
    _write(args.output, out_rows)
    print(f"wrote {args.output} ({len(out_rows)} rows)")
    keep = set(feasible)
    for source, target in args.restrict:
        other = read_rows(Path(source))
        if len(other) != len(rows):
            raise SystemExit(f"{source} has {len(other)} rows, not {len(rows)}; "
                             "the restriction is by row index, so the files have "
                             "to be two repairs of the same released file")
        subset = [row for index, row in enumerate(other) if index in keep]
        _write(Path(target), subset)
        print(f"wrote {target} ({len(subset)} rows cut from {source})")
    print(f"  {len(feasible)}/{len(numeric)} numeric items made exchangeable "
          f"({len(feasible) / len(numeric):.1%}); {len(infeasible)} admit no "
          f"exchangeable rewriting in their own written style")
    if not args.only_feasible and infeasible:
        print("  those keep their released options, so the audit below is of a "
              "mixed file; pass --only-feasible for the matched subset")
    print("  key slot drawn: "
          + ", ".join(f"{r}: {slots[r] / len(feasible):.1%}" for r in range(k)))
    print("  the same construction retrying one slot at a time would give a key "
          "value rank of " + ", ".join(f"{s:.1%}" for s in naive_share)
          + f" over {naive_total} items")

    # Audited with the same tool as every other file in the paper, over all four
    # audited orderings including the numeric one: one construction is supposed
    # to close them together, so measuring them separately is the test of it.
    new_items, _ = build_items(out_rows, args.key_field, args.distractor_field,
                               args.cluster_field, args.question_field, k)
    new_numeric = [it for it in new_items if it["rank"] is not None]
    channels = surface_channels(new_numeric, args.replicates, k=k,
                                orderings=tuple(AUDIT_ORDERINGS))
    print()
    for name, block in channels["orderings"].items():
        entry = block.get("numeric items") or next(iter(block.values()))
        bound = entry["credit_bound"]
        shares = [entry["key_by_rank_p"][str(r)] for r in range(k)]
        print(f"  {name:<20} {100 * bound['credit_lower_bound']:+5.1f} points  "
              f"(key rank " + ", ".join(f"{s:.1%}" for s in shares) + ")")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({
            "source": str(args.jsonl), "output": str(args.output),
            "seed": args.seed, "n_options": k, "n_numeric": len(numeric),
            "only_feasible": args.only_feasible,
            "n_feasible": len(feasible), "n_infeasible": len(infeasible),
            "feasible_row_indices": feasible,
            "infeasible_row_indices": infeasible,
            "key_slot_share": [slots[r] / len(feasible) for r in range(k)],
            "naive_retry_rank_share": naive_share, "naive_retry_n": naive_total,
            "channels": channels,
        }, indent=1) + "\n")
        print(f"\nwrote {args.report}")


if __name__ == "__main__":
    main()
