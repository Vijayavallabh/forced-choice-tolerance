#!/usr/bin/env python3
"""Audit a pinned JSONL; cache bibliographic evidence without inferring exposure."""
import argparse
import calendar
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import urllib.error
import urllib.parse
import urllib.request

REVISION = "f8cc3bdcc6357c88b8c3648306522b9c422dc95a"
DATASET_URL = f"https://huggingface.co/datasets/futurehouse/BixBench/resolve/{REVISION}/BixBench.jsonl"
MISSING_TOKENS = {"", "not available", "none", "n/a", "na", "null"}
DOI_RE = re.compile(r"10\.\d{4,9}/[^\s\"'<>?,&#]+", re.I)


def load(path):
    with open(path, encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise ValueError("Empty dataset")
    return rows


def extract_dois(value):
    return sorted({m.rstrip(".,;").lower() for m in DOI_RE.findall(urllib.parse.unquote(str(value)))})


def extract_doi(value):
    return next(iter(extract_dois(value)), None)


def classify(value):
    if value is None or str(value).strip().lower() in MISSING_TOKENS:
        return "missing"
    if extract_doi(value):
        return "doi"
    if re.search(r"https?://", str(value), re.I):
        return "repository_url"
    return "other"


def source_identifiers(value):
    identifiers = set(extract_dois(value))
    for url in re.findall(r"https?://[^\s,]+", str(value)):
        url = url.rstrip(".;")
        if extract_doi(url):
            continue
        parsed = urllib.parse.urlsplit(url)
        zenodo = re.search(r"/records?/(\d+)", parsed.path)
        nature = re.search(r"/articles/(s[\w-]+)", parsed.path)
        if parsed.hostname in {"zenodo.org", "www.zenodo.org"} and zenodo:
            identifiers.add("10.5281/zenodo." + zenodo.group(1))
        elif parsed.hostname in {"nature.com", "www.nature.com"} and nature:
            identifiers.add("10.1038/" + nature.group(1).lower())
        else:
            identifiers.add(urllib.parse.urlunsplit(parsed._replace(fragment="")))
    return sorted(identifiers)


def collapse_to_capsules(rows):
    capsules = {}
    for row in rows:
        uuid = row["capsule_uuid"]
        capsule = capsules.setdefault(uuid, {"n_questions": 0, "paper_values": [], "categories": []})
        capsule["n_questions"] += 1
        paper = row.get("paper")
        if paper not in capsule["paper_values"]:
            capsule["paper_values"].append(paper)
        for category in row.get("categories") or []:
            if category not in capsule["categories"]:
                capsule["categories"].append(category)
    for uuid, capsule in capsules.items():
        if len(capsule["paper_values"]) != 1:
            raise ValueError(f"Conflicting paper fields within capsule {uuid}")
        capsule["paper"] = capsule.pop("paper_values")[0]
        capsule["class"] = classify(capsule["paper"])
        capsule["identifiers"] = source_identifiers(capsule["paper"])
    return capsules


def date_interval(value):
    """Preserve year/month precision instead of inventing a day."""
    parts = str(value).split("T")[0].split("-")
    year = int(parts[0])
    month = int(parts[1]) if len(parts) > 1 else 1
    day = int(parts[2]) if len(parts) > 2 else 1
    upper_month = month if len(parts) > 1 else 12
    upper_day = day if len(parts) > 2 else calendar.monthrange(year, upper_month)[1]
    return {"lower": f"{year:04d}-{month:02d}-{day:02d}",
            "upper": f"{year:04d}-{upper_month:02d}-{upper_day:02d}",
            "precision": ("year", "month", "day")[min(len(parts), 3) - 1]}


def fetch(url, cache_dir, offline=False):
    path = Path(cache_dir) / (hashlib.sha256(url.encode()).hexdigest() + ".json")
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    if offline:
        raise FileNotFoundError(f"No cached response for {url}")
    request = urllib.request.Request(url, headers={"User-Agent": "TATE-provenance-audit/2.0"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            record = {"url": url, "retrieved_at": datetime.now(timezone.utc).isoformat(),
                      "status": response.status, "body": json.load(response)}
    except urllib.error.HTTPError as exc:
        record = {"url": url, "retrieved_at": datetime.now(timezone.utc).isoformat(),
                  "status": exc.code, "error": str(exc)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def evidence_from_record(record, provider):
    evidence = []
    if record.get("status") != 200:
        return evidence
    if provider == "crossref":
        data = record["body"]["message"]
        for field in ("published-online", "published-print", "published", "posted"):
            parts = data.get(field, {}).get("date-parts", [[]])[0]
            if parts:
                value = "-".join(str(p) if i == 0 else f"{p:02d}" for i, p in enumerate(parts))
                evidence.append({"field": field, "value": value, "kind": "publication", **date_interval(value)})
    else:
        data = record["body"]["data"]["attributes"]
        if data.get("publicationYear"):
            value = str(data["publicationYear"])
            evidence.append({"field": "publicationYear", "value": value, "kind": "publication", **date_interval(value)})
        for item in data.get("dates", []):
            if item.get("dateType") in {"Available", "Issued"}:
                value = item["date"]
                try:
                    evidence.append({"field": item["dateType"], "value": value, "kind": "publication", **date_interval(value)})
                except (ValueError, IndexError):
                    pass
        if data.get("created"):
            value = data["created"][:10]
            evidence.append({"field": "created", "value": value, "kind": "registration_only", **date_interval(value)})
    for item in evidence:
        item.update(provider=provider, evidence_url=record["url"])
    return evidence


def resolve(identifier, cache_dir, offline):
    if not identifier.startswith("10."):
        return {"status": "unsupported_url", "evidence": []}
    evidence, attempts = [], []
    for provider, base in (("crossref", "https://api.crossref.org/works/"),
                           ("datacite", "https://api.datacite.org/dois/")):
        url = base + urllib.parse.quote(identifier, safe="/")
        record = fetch(url, cache_dir, offline)
        attempts.append({"url": url, "status": record["status"]})
        evidence.extend(evidence_from_record(record, provider))
        if any(e["kind"] == "publication" for e in evidence):
            break
    publication = [e for e in evidence if e["kind"] == "publication"]
    if not publication and identifier.startswith("10.5281/zenodo."):
        url = "https://zenodo.org/api/records/" + identifier.rsplit(".", 1)[1]
        record = fetch(url, cache_dir, offline)
        attempts.append({"url": url, "status": record["status"]})
        if record["status"] == 200:
            value = record["body"].get("metadata", {}).get("publication_date")
            if value:
                evidence.append({"field": "publication_date", "value": value,
                    "kind": "publication", "provider": "zenodo", "evidence_url": url,
                    **date_interval(value)})
        publication = [e for e in evidence if e["kind"] == "publication"]
    return {"status": "dated" if publication else "unresolved", "evidence": evidence,
            "attempts": attempts,
            "earliest_publication_upper": min((e["upper"] for e in publication), default=None),
            "first_public_appearance_verified": False}


def audit(rows, raw_bytes, resolved=None):
    capsules = collapse_to_capsules(rows)
    classes = {}
    for name in ("doi", "repository_url", "other", "missing"):
        group = [c for c in capsules.values() if c["class"] == name]
        classes[name] = {"capsules": len(group), "questions": sum(c["n_questions"] for c in group)}
    keys = sorted(set().union(*(row.keys() for row in rows)))
    date_keys = [k for k in keys if re.search(r"date|time|release|publish|cutoff", k, re.I)]
    identifiers = sorted({i for c in capsules.values() for i in c["identifiers"]})
    # Explicit shared identifiers imply dependence; disjoint identifiers do not prove independence.
    groups = []
    for uuid, capsule in capsules.items():
        ids = set(capsule["identifiers"])
        if not ids:
            continue
        overlap = [g for g in groups if ids & g["identifiers"]]
        merged = {"identifiers": ids, "capsules": {uuid}}
        for group in overlap:
            merged["identifiers"].update(group["identifiers"])
            merged["capsules"].update(group["capsules"])
            groups.remove(group)
        groups.append(merged)
    result = {"dataset_sha256": hashlib.sha256(raw_bytes).hexdigest(),
              "n_questions": len(rows), "n_capsules": len(capsules),
              "versions": dict(Counter(str(r.get("version")) for r in rows)),
              "eval_modes": dict(Counter(r["eval_mode"] for r in rows)),
              "top_level_fields": keys, "date_like_top_level_fields": date_keys,
              "classes": classes, "identifiers": identifiers,
              "explicit_source_groups": [{k: sorted(v) for k, v in g.items()} for g in groups],
              "capsules": capsules}
    if resolved is not None:
        dated, pre2025 = [], []
        for uuid, capsule in capsules.items():
            records = [resolved[i] for i in capsule["identifiers"]]
            dates = [r["earliest_publication_upper"] for r in records if r.get("earliest_publication_upper")]
            if dates:
                dated.append(uuid)
                if min(dates) < "2025-01-01":
                    pre2025.append(uuid)
        result["date_resolution"] = {"records": resolved, "dated_capsules": len(dated),
            "dated_questions": sum(capsules[u]["n_questions"] for u in dated),
            "publication_by_end_2024_capsules": len(pre2025),
            "publication_by_end_2024_questions": sum(capsules[u]["n_questions"] for u in pre2025)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", type=Path, default=Path("data/bixbench.jsonl"))
    parser.add_argument("--resolve-dates", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--cache-dir", default="sources/api_cache")
    parser.add_argument("--output", type=Path, default=Path("results/audit.json"))
    args = parser.parse_args()
    rows = load(args.jsonl)
    resolved = None
    if args.resolve_dates:
        identifiers = sorted({i for c in collapse_to_capsules(rows).values() for i in c["identifiers"]})
        resolved = {}
        for identifier in identifiers:
            resolved[identifier] = resolve(identifier, args.cache_dir, args.offline)
            print(identifier, resolved[identifier]["status"], flush=True)
    result = audit(rows, args.jsonl.read_bytes(), resolved)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in {"capsules", "date_resolution", "identifiers", "explicit_source_groups"}}, indent=2))
    print("Explicit source groups:", len(result["explicit_source_groups"]))
    if resolved is not None:
        print(json.dumps({k: v for k, v in result["date_resolution"].items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
