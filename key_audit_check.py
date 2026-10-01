#!/usr/bin/env python3
r"""The with-data contrasts without the items whose keys a concurrent audit re-derived.

A concurrent audit of BixBench (Dai, 2026; Zenodo 10.5281/zenodo.22151932, MIT licence)
re-derived answer keys from the capsules' raw data for three capsules in full (bix-8, bix-26,
bix-49) and three in a pilot (bix-1, bix-4, bix-43), and traced most scored failures there
to the key or the question's wording: on v1.5's numeric items it documents wrong keys for
bix-8-q3, q6 and q7, all five of bix-49's and bix-26-q5. Every distance in this paper is
measured from the released key, so a wrong key moves it. This drops every numeric item in
all six capsules the audit examined -- the flagged ones and their neighbours alike, which
needs no judgment of ours about which keys are wrong -- and restates the headline contrasts:

* the paper's seven v1.5 run sets, pooled: repaired - placebo and repaired - released on the
  keys the repair moved to an edge, under the nearest-option rule, the family reader and
  gemma-3-27b (``reader_split.py``);
* the registered data: D1's moved-key gain and repair-minus-placebo (v1.0 items in the same
  capsules), and D2's for the new seeds and Qwen3-235B-A22B (``replication.py``);
* the same with-data runs read by a model as BixBench reads a run, every reader and mode
  ``published_reads.py`` took, on whatever of them it read (``reads|...`` keys).

Intervals are plain 95% cluster bootstraps over capsules (``bixbench_withdata.cluster_interval``).
Not registered.

    python3 key_audit_check.py      # results/key_audit_check.json
"""
import json
import pathlib

import bixbench_withdata as bw
import reader_split as rs
import replication as rp

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / "results" / "key_audit_check.json"
AUDITED = ("bix-8", "bix-26", "bix-49", "bix-1", "bix-4", "bix-43")
FLAGGED = ("bix-8-q3", "bix-8-q6", "bix-8-q7", "bix-49-q1", "bix-49-q2", "bix-49-q3", "bix-49-q4",
           "bix-49-q5", "bix-26-q5")
LEVEL = 0.95


def main():
    items = {json.loads(l)["question_id"]: json.loads(l) for l in open(ROOT / "data" / "bixbench.jsonl")}
    caps = {items[q]["capsule_uuid"] for q in items if items[q]["short_id"] in AUDITED}
    sets15 = bw.option_sets()
    drop15 = {q for q in sets15 if items[q]["capsule_uuid"] in caps}
    assert set(FLAGGED) <= drop15, sorted(set(FLAGGED) - drop15)
    cap15 = {q: items[q]["capsule_uuid"] for q in items}
    doc = rp.load_extract()
    v10 = rp.v10_sets()
    cap10 = {q: e["capsule"].replace("CapsuleFolder-", "") for q, e in doc["items"].items()}
    drop10 = {q for q in v10 if cap10[q] in caps}
    report = {"audited_capsules": list(AUDITED), "flagged_numeric_v15": list(FLAGGED),
              "n_dropped": {"v1.5": len(drop15), "v1.0": len(drop10)},
              "n_left": {"v1.5": len(sets15) - len(drop15), "v1.0": len(v10) - len(drop10)},
              "level": LEVEL, "not_registered": True}
    group = {q: rp.group_of(s) for q, s in sets15.items()}
    report["n_moved_left"] = sum(1 for q in sets15 if q not in drop15 and group[q] == "moved to an edge")
    # where the repair's newly credited with-data answers fall: on keys the audit flags, or elsewhere
    # (the seven run sets, keys moved to an edge, as bracketing.mechanism counts them)
    import bracketing as br
    credited = {"flagged": 0.0, "not flagged": 0.0}
    for run in br.RUNS:
        for r in br.rows_of(run):
            q = r["question_id"]
            if not r["numeric"] or r["condition"] != "data" or group[q] != "moved to an edge":
                continue
            gain = bw.nearest_is_key(r["answer"], sets15[q]["repaired"]) - bw.nearest_is_key(r["answer"], sets15[q]["released"])
            if gain > 0:
                credited["flagged" if q in FLAGGED else "not flagged"] += gain
    report["newly_credited_with_data"] = credited
    report["moved_flagged"] = sorted(q for q in FLAGGED if group.get(q) == "moved to an edge")

    # the seven run sets, pooled
    for reader in rs.READERS:
        for arm, base in (("repaired", "placebo"), ("repaired", "released")):
            pool = rs.pooled(rs.per_item("data", reader, arm, base, sets15))
            for label, keep in (("all", lambda q: True), ("without audited", lambda q: q not in drop15)):
                qs = [q for q in pool if group[q] == "moved to an edge" and keep(q)]
                report[f"seven|{reader}|{arm}-{base}|moved|{label}"] = bw.cluster_interval(
                    [pool[q] for q in qs], [cap15[q] for q in qs], level=LEVEL)

    # the registered data sets
    def registered(runs, sets, capsule, drop, key):
        groups = {q: rp.group_of(s) for q, s in sets.items()}
        for arm, base in (("repaired", "released"), ("repaired", "placebo")):
            _, pooled = rp.gains(runs, {q: s for q, s in sets.items() if base in s and arm in s}, arm, base)
            for label, keep in (("all", lambda q: True), ("without audited", lambda q: q not in drop)):
                qs = [q for q in pooled if groups[q] == "moved to an edge" and keep(q)]
                report[f"{key}|{arm}-{base}|moved|{label}"] = bw.cluster_interval(
                    [pooled[q] for q in qs], [capsule[q] for q in qs], level=LEVEL)

    runs1 = {name: [(q, a) for q, a, _ in doc["open_runs"][name]] for name in rp.OPEN_RUNS}
    registered(runs1, v10, cap10, drop10, "D1")
    runs2 = rp.d2_runs()
    for fam, want in (("qwen3-235b", lambda n: n.startswith("qwen3-235b|")),
                      ("reruns", lambda n: not n.startswith("qwen3-235b|"))):
        sub = {n: runs2[n]["data"] for n in runs2 if want(n)}
        registered(sub, sets15, cap15, drop15, f"D2 {fam}")

    # the same runs read by a model as BixBench reads a run (published_reads.py): every reader and mode, on
    # whatever of the published runs, the seven run sets and the new runs it read with the data
    import published_reads as pr
    for reader, mode in (("gemma27b", "forced"), ("qwen72b", "forced"), ("llama70b", "forced"),
                         ("gemma27b", "decline")):
        path = pr.rows_path(reader, mode)
        if not path.exists():
            continue
        rows = pr.load_rows(path)
        subsets = (("D1", [r for r in rows if r["set"] == "D1"], v10, cap10, drop10),
                   ("D0", [r for r in rows if r["set"] == "D0" and r["condition"] == "data"], sets15, cap15, drop15),
                   ("D2 reruns", [r for r in rows if r["set"] == "D2" and r["condition"] == "data"
                                  and not r["run"].startswith("qwen3-235b|")], sets15, cap15, drop15),
                   ("D2 qwen3-235b", [r for r in rows if r["set"] == "D2" and r["condition"] == "data"
                                      and r["run"].startswith("qwen3-235b|")], sets15, cap15, drop15))
        for key, sub, sets, capsule, drop in subsets:
            if not sub:
                continue
            groups = {q: rp.group_of(s) for q, s in sets.items()}
            for arm, base in (("repaired", "released"), ("repaired", "placebo")):
                pooled = pr.gains(sub, arm, base)
                for label, keep in (("all", lambda q: True), ("without audited", lambda q: q not in drop)):
                    qs = [q for q in pooled if groups[q] == "moved to an edge" and keep(q)]
                    report[f"reads|{reader}|{mode}|{key}|{arm}-{base}|moved|{label}"] = bw.cluster_interval(
                        [pooled[q] for q in qs], [capsule[q] for q in qs], level=LEVEL)
    OUT.write_text(json.dumps(report, indent=1) + "\n")
    print(f"dropped {report['n_dropped']}; moved keys left on v1.5: {report['n_moved_left']}; newly credited "
          f"with the data {report['newly_credited_with_data']} (moved and flagged: {report['moved_flagged']})")
    for k, v in report.items():
        if isinstance(v, dict) and "mean" in v:
            print(f"   {k:62s} {v['mean']:+6.1f} [{v['lo']:+6.1f},{v['hi']:+6.1f}] ({v['n_items']} items, "
                  f"{v['n_clusters']} capsules)")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
