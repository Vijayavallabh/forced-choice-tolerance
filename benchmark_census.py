#!/usr/bin/env python3
"""Census of temporal-provenance metadata across public science-agent benchmarks.

For each released benchmark artifact we ask four questions that a reader needs
in order to decide whether a score reflects analysis of new evidence or recall
of familiar results:

  1. Does any released field carry a date at all?
  2. Can a reader identify the source study behind each task?
  3. Is the benchmark artifact itself versioned in the data?
  4. How many independent source studies do the tasks actually represent?

Only released metadata is inspected. A missing field is reported as missing,
never inferred. Absence of a source identifier is not evidence of novelty, and
presence of one is not evidence of exposure.
"""
import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

DATE_FIELD_RE = re.compile(r"date|time|year|release|publish|posted|cutoff|created|updated", re.I)
DOI_RE = re.compile(r"10\.\d{4,9}/[^\s\"'<>?,&#]+", re.I)
URL_RE = re.compile(r"https?://", re.I)
MISSING = {"", "none", "null", "n/a", "na", "not available", "nan", "[]", "{}"}
# Field names that plausibly carry a pointer to the underlying study.
SOURCE_FIELDS = ("paper", "source", "sources", "doi", "url", "reference", "citation",
                 "paper-title", "src_file_or_path", "github_name", "capsule_title")
DATE_VALUE_RE = re.compile(r"\b(19|20)\d{2}-\d{2}(-\d{2})?\b")

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


def is_missing(value):
    if value is None:
        return True
    if isinstance(value, (list, tuple, dict)):
        return len(value) == 0
    return str(value).strip().lower() in MISSING


def load_rows(path):
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else [data]
    if path.suffix == ".csv":
        with path.open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    raise ValueError(f"unsupported file type: {path}")


def source_key(row):
    """A best-effort study identifier from whichever source field exists."""
    for field in SOURCE_FIELDS:
        value = row.get(field)
        if isinstance(value, (list, tuple)):
            value = value[0] if value else None
        if is_missing(value):
            continue
        text = str(value)
        doi = DOI_RE.search(text)
        if doi:
            return ("doi", doi.group(0).rstrip(".,;").lower())
        if URL_RE.search(text):
            return ("url", text.strip().rstrip("/"))
        return ("label", text.strip()[:200])
    return None


def census(name, path, cluster_field=None, notes=None):
    rows = load_rows(path)
    if not rows:
        return None
    fields = sorted({k for r in rows for k in r})

    # 1. date fields, detected by field name and independently by field value
    date_named = [f for f in fields if DATE_FIELD_RE.search(f)]
    date_valued = []
    for f in fields:
        hits = sum(1 for r in rows if DATE_VALUE_RE.search(str(r.get(f, ""))))
        if hits >= max(2, 0.5 * len(rows)):
            date_valued.append(f)

    # 2. source identifier coverage
    keys = [source_key(r) for r in rows]
    identified = [k for k in keys if k is not None]
    kinds = Counter(k[0] for k in identified)

    # 3. artifact version recorded in the data itself
    version_fields = [f for f in fields if re.fullmatch(r"version|ver|v|revision", f, re.I)]
    version_values = {f: dict(Counter(str(r.get(f)) for r in rows)) for f in version_fields}

    # 4. dependence structure: how many independent studies do the tasks represent?
    if cluster_field:
        cluster_keys = [str(r.get(cluster_field)) for r in rows]
    else:
        cluster_keys = [str(k) if k else f"__unidentified_{i}" for i, k in enumerate(keys)]
    groups = Counter(cluster_keys)
    named_groups = {g: c for g, c in groups.items() if not g.startswith("__unidentified_")}

    return {
        "benchmark": name,
        "file": str(path),
        "n_items": len(rows),
        "n_fields": len(fields),
        "fields": fields,
        "date_fields_by_name": date_named,
        "date_fields_by_value": date_valued,
        "publishes_any_date": bool(date_named or date_valued),
        "source_identifier": {
            "n_with_identifier": len(identified),
            "coverage": len(identified) / len(rows),
            "kinds": dict(kinds),
        },
        "version_fields": version_values,
        "has_canary": any("canary" in f.lower() for f in fields),
        "dependence": {
            "cluster_field": cluster_field or "derived source identifier",
            "n_distinct_sources": len(named_groups),
            "n_items_without_source": len(rows) - len(identified),
            "max_items_per_source": max(named_groups.values()) if named_groups else 0,
            "mean_items_per_source": (sum(named_groups.values()) / len(named_groups)) if named_groups else 0,
        },
        "notes": notes or "",
    }


TARGETS = [
    ("BixBench v1.5", Path("data/bixbench.jsonl"), "capsule_uuid",
     "Pinned Hugging Face revision f8cc3bd...422dc95a; audited in detail elsewhere in this repository."),
    ("LAB-Bench LitQA2", Path("data/external/labbench/LitQA2.jsonl"), None, ""),
    ("LAB-Bench SuppQA", Path("data/external/labbench/SuppQA.jsonl"), None, ""),
    ("LAB-Bench DbQA", Path("data/external/labbench/DbQA.jsonl"), None, ""),
    ("LAB-Bench SeqQA", Path("data/external/labbench/SeqQA.jsonl"), None, ""),
    ("LAB-Bench TableQA", Path("data/external/labbench/TableQA.jsonl"), None, ""),
    ("LAB-Bench FigQA", Path("data/external/labbench/FigQA.jsonl"), None, ""),
    ("LAB-Bench ProtocolQA", Path("data/external/labbench/ProtocolQA.jsonl"), None, ""),
    ("LAB-Bench CloningScenarios", Path("data/external/labbench/CloningScenarios.jsonl"), None, ""),
    ("ScienceAgentBench", Path("data/external/other/ScienceAgentBench.csv"), None,
     "Source column is a GitHub repository name, not a dated publication record."),
    ("CORE-Bench (train split)", Path("data/external/other/core_train.json"), "capsule_id",
     "Public split only; the test split is released encrypted."),
    ("DiscoveryBench (real, answer key)", Path("data/external/other/discoverybench_answer_key_real.csv"), "dataset",
     "Answer key only; task metadata lives in per-domain files."),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=Path("results/benchmark_census.json"))
    args = parser.parse_args()

    records = []
    for name, path, cluster, notes in TARGETS:
        if not path.exists():
            print(f"skip (absent): {name}")
            continue
        rec = census(name, path, cluster, notes)
        if rec:
            records.append(rec)

    width = max(len(r["benchmark"]) for r in records)
    print(f"{'benchmark':{width}}  {'items':>6}  {'dates?':>6}  {'src cov':>7}  {'sources':>7}  {'max/src':>7}")
    for r in records:
        d = r["dependence"]
        print(f"{r['benchmark']:{width}}  {r['n_items']:>6}  "
              f"{('YES' if r['publishes_any_date'] else 'no'):>6}  "
              f"{r['source_identifier']['coverage']:>6.1%}  "
              f"{d['n_distinct_sources']:>7}  {d['max_items_per_source']:>7}")

    n_dated = sum(1 for r in records if r["publishes_any_date"])
    total_items = sum(r["n_items"] for r in records)
    summary = {
        "n_benchmark_files": len(records),
        "n_items_total": total_items,
        "n_files_publishing_any_date": n_dated,
        "n_files_with_canary": sum(1 for r in records if r["has_canary"]),
        "statement": (f"{n_dated} of {len(records)} released benchmark files covering {total_items} "
                      "tasks carry any date field for the source study, the underlying data, or the "
                      "benchmark artifact."),
    }
    print("\n" + summary["statement"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "benchmarks": records}, indent=2) + "\n",
                           encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
