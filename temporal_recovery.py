#!/usr/bin/env python3
r"""The dates the benchmarks do not publish, recovered from the identifiers they do.

``benchmark_census.py`` establishes that none of the twelve released files
carries a date. That is the premise of the no-data fallback, and on its own it
is weaker than it looks: five of the twelve give a resolvable DOI for every
item, or for two thirds of them, and a DOI resolves to a publication date in
one request. "Not published" is not "not recoverable", and a claim that the
temporal test cannot be run has to survive the obvious attempt to run it.

So this attempts it. Every source identifier in the census is resolved against
Crossref (DOIs) and the bioRxiv/medRxiv API (preprint DOIs Crossref dates only
at posting), the answer is cached on disk, and the coverage is reported per
benchmark -- including the items where the attempt fails, which is the number
that decides whether the temporal test is available in practice.

    python3 temporal_recovery.py --sleep 0.12

No credential is sent and no identifying header is set: Crossref's anonymous
pool is slower and sufficient at this scale.
"""
import argparse
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from benchmark_census import TARGETS, load_rows, source_key

USER_AGENT = "benchmark-date-recovery/1.0 (research; no contact configured)"
BIORXIV_PREFIX = re.compile(r"^10\.1101/", re.I)


def _get(url, timeout):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as handle:
        return json.loads(handle.read().decode("utf-8"))


def _date_parts(block):
    parts = (block or {}).get("date-parts") or []
    if not parts or not parts[0] or parts[0][0] is None:
        return None
    fields = (list(parts[0]) + [1, 1])[:3]
    return "-".join(f"{int(v):02d}" if i else f"{int(v):04d}"
                    for i, v in enumerate(fields))


def crossref_date(doi, timeout):
    """The earliest date Crossref knows for this DOI, and which field it came from."""
    url = "https://api.crossref.org/works/" + urllib.parse.quote(doi, safe="")
    message = _get(url, timeout)["message"]
    candidates = {}
    for field in ("issued", "published-print", "published-online", "published"):
        date = _date_parts(message.get(field))
        if date:
            candidates[field] = date
    created = (message.get("created") or {}).get("date-time")
    if created:
        candidates["created"] = created[:10]
    if not candidates:
        return None
    field = min(candidates, key=lambda f: candidates[f])
    return {"date": candidates[field], "field": field,
            "type": message.get("type"), "all": candidates}


def biorxiv_date(doi, timeout):
    """Preprint servers date the first posting, which Crossref records as `created`."""
    for server in ("biorxiv", "medrxiv"):
        try:
            payload = _get(f"https://api.{server}.org/details/{server}/{doi}", timeout)
        except (urllib.error.URLError, json.JSONDecodeError, TimeoutError):
            continue
        collection = payload.get("collection") or []
        dates = sorted(entry["date"] for entry in collection if entry.get("date"))
        if dates:
            return {"date": dates[0], "field": f"{server} first posting",
                    "type": "posted-content", "all": {"versions": len(dates)}}
    return None


def resolve(kind, value, cache, timeout, sleep):
    key = f"{kind}|{value}"
    if key in cache:
        return cache[key]
    answer = None
    if kind == "doi":
        try:
            if BIORXIV_PREFIX.match(value):
                answer = biorxiv_date(value, timeout) or crossref_date(value, timeout)
            else:
                answer = crossref_date(value, timeout)
        except urllib.error.HTTPError as error:
            answer = {"error": f"http {error.code}"}
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            answer = {"error": type(error).__name__}
        time.sleep(sleep)
    elif kind == "url":
        found = re.search(r"10\.\d{4,9}/[^\s\"'<>?,&#]+", value)
        if found:
            return resolve("doi", found.group(0).rstrip(".,;").lower(), cache,
                           timeout, sleep)
        answer = {"error": "no doi in url"}
    else:
        answer = {"error": f"identifier is a {kind}, not a record"}
    cache[key] = answer
    return answer


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default="results/source_dates.json")
    ap.add_argument("--timeout", type=float, default=25.0)
    ap.add_argument("--sleep", type=float, default=0.12)
    ap.add_argument("--output", default="results/temporal_recovery.json")
    ap.add_argument("--per-item", default="results/item_dates.json")
    args = ap.parse_args()

    cache_path = Path(args.cache)
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    before = len(cache)

    report, per_item = [], {}
    for name, path, _cluster, _notes in TARGETS:
        if not path.exists():
            continue
        rows = load_rows(path)
        dated, undated, no_identifier, dates = 0, 0, 0, []
        item_dates = {}
        for index, row in enumerate(rows):
            identifier = source_key(row)
            if identifier is None:
                no_identifier += 1
                continue
            answer = resolve(identifier[0], identifier[1], cache, args.timeout,
                             args.sleep)
            if answer and "date" in answer:
                dated += 1
                dates.append(answer["date"])
                item_dates[str(row.get("id") or row.get("question_id") or index)] = {
                    "source": identifier[1], "date": answer["date"],
                    "field": answer["field"],
                    "cluster": row.get("capsule_uuid") or identifier[1],
                }
            else:
                undated += 1
        per_item[name] = item_dates
        dates.sort()
        entry = {"benchmark": name, "file": str(path), "n_items": len(rows),
                 "n_dated": dated, "n_identifier_unresolved": undated,
                 "n_no_identifier": no_identifier,
                 "coverage": dated / len(rows) if rows else 0.0,
                 "earliest": dates[0] if dates else None,
                 "latest": dates[-1] if dates else None,
                 "median": dates[len(dates) // 2] if dates else None}
        report.append(entry)
        print(f"{name:<32} {dated:>4}/{len(rows):<5} dated ({entry['coverage']:>5.1%})  "
              f"{entry['earliest']} .. {entry['latest']}", flush=True)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, indent=1), encoding="utf-8")

    total_items = sum(e["n_items"] for e in report)
    total_dated = sum(e["n_dated"] for e in report)
    summary = {
        "n_files": len(report), "n_items_total": total_items,
        "n_items_dated": total_dated,
        "coverage": total_dated / total_items if total_items else 0.0,
        "n_files_with_any_date_recovered": sum(1 for e in report if e["n_dated"]),
        "n_new_lookups": len(cache) - before,
        "statement": (f"no released file publishes a date; resolving the identifiers "
                      f"they do publish recovers one for {total_dated} of {total_items} "
                      f"tasks in {sum(1 for e in report if e['n_dated'])} of "
                      f"{len(report)} files."),
    }
    print("\n" + summary["statement"])
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps({"summary": summary, "benchmarks": report},
                                            indent=2) + "\n", encoding="utf-8")
    Path(args.per_item).write_text(json.dumps(per_item, indent=1), encoding="utf-8")
    print(f"wrote {args.output} and {args.per_item}")


if __name__ == "__main__":
    main()
