#!/usr/bin/env python3
"""Vendor the option sets of public multiple-choice benchmarks for the survey.

The question the survey asks of a benchmark is narrow: among items whose
options are all distinct numbers, where does the keyed answer sit in the sorted
order? Answering it needs the option values and which one is keyed, and nothing
else, so that is all this fetches. Questions, rationales and explanations are
dropped, which keeps the vendored evidence small enough to commit and keeps the
survey from becoming a redistribution of someone else's benchmark.

Each target writes ``data/external/survey/<name>.jsonl`` with one record per
item: the options with the keyed one first, the option count, and a cluster
label where the benchmark provides something to cluster on. A URL and SHA-256
go into ``data/external/survey/PROVENANCE.json``.

Needs ``pyarrow`` to read Hugging Face parquet, and nothing else; the survey
itself (``channel_survey.py``) reads the vendored JSONL with the standard
library alone.
"""
import argparse
import datetime
import gzip
import hashlib
import io
import json
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path("data/external/survey")
API = "https://huggingface.co/api/datasets/{ds}/parquet/{cfg}/{split}"


def get(url, timeout=180):
    request = urllib.request.Request(url, headers={"User-Agent": "option-geometry-survey/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


# --------------------------------------------------------------------------
# One extractor per benchmark. Each returns (options_with_key_first, cluster)
# or None for an item that carries no usable option set.
def from_fields(key_field, distractor_fields, cluster_field=None):
    def extract(row):
        key = row.get(key_field)
        distractors = [row.get(f) for f in distractor_fields]
        if key is None or any(d is None for d in distractors):
            return None
        return [str(key)] + [str(d) for d in distractors], _cluster(row, cluster_field)
    return extract


def from_list(options_field, index_field, cluster_field=None, offset=0):
    """Options as a list with an integer index into it."""
    def extract(row):
        options, index = row.get(options_field), row.get(index_field)
        if not options or index is None:
            return None
        try:
            index = int(index) - offset
        except (TypeError, ValueError):
            return None
        options = [str(o) for o in options if o is not None]
        if not 0 <= index < len(options):
            return None
        return [options[index]] + options[:index] + options[index + 1:], _cluster(row, cluster_field)
    return extract


def from_letter(options_field, letter_field, cluster_field=None, strip_label=False):
    """Options as a list or dict addressed by a letter."""
    def extract(row):
        options, letter = row.get(options_field), row.get(letter_field)
        if not options or not letter:
            return None
        letter = str(letter).strip().upper()
        if isinstance(options, dict):
            ordered = [options[k] for k in sorted(options)]
            labels = sorted(options)
        else:
            ordered, labels = list(options), None
        if strip_label:
            # AQuA-RAT writes options as "A)21"; keep the value, drop the label.
            labels = [str(o).split(")")[0].strip().upper() for o in ordered]
            ordered = [str(o).split(")", 1)[1] if ")" in str(o) else str(o) for o in ordered]
        if labels is None:
            labels = [chr(ord("A") + i) for i in range(len(ordered))]
        if letter not in labels:
            return None
        index = labels.index(letter)
        ordered = [str(o) for o in ordered]
        return [ordered[index]] + ordered[:index] + ordered[index + 1:], _cluster(row, cluster_field)
    return extract


def from_labelled_choices(cluster_field=None):
    """ARC and OpenBookQA: ``{"text": [...], "label": [...]}`` plus an answer key."""
    def extract(row):
        choices, key = row.get("choices"), row.get("answerKey")
        if not choices or not key:
            return None
        texts = list(choices.get("text") or [])
        labels = [str(label).strip().upper() for label in (choices.get("label") or [])]
        key = str(key).strip().upper()
        if key not in labels or len(texts) != len(labels):
            return None
        index = labels.index(key)
        texts = [str(t) for t in texts]
        return [texts[index]] + texts[:index] + texts[index + 1:], _cluster(row, cluster_field)
    return extract


def _cluster(row, field):
    if not field:
        return None
    value = row.get(field)
    return None if value in (None, "") else str(value)


TARGETS = [
    # name, dataset, config, split, extractor, row cap
    ("sciq", "allenai/sciq", "default", "train",
     from_fields("correct_answer", ["distractor1", "distractor2", "distractor3"]), 0),
    ("mmlu", "cais/mmlu", "all", "test",
     from_list("choices", "answer", cluster_field="subject"), 0),
    ("mmlu_pro", "TIGER-Lab/MMLU-Pro", "default", "test",
     from_list("options", "answer_index", cluster_field="src"), 0),
    ("aqua_rat", "deepmind/aqua_rat", "raw", "test",
     from_letter("options", "correct", strip_label=True), 0),
    ("medmcqa", "openlifescienceai/medmcqa", "default", "train",
     from_fields("opa", ["opb", "opc", "opd"], cluster_field="subject_name"), 30000),
    # No cluster field: MedQA's meta_info records the exam step, which is a
    # difficulty label rather than a shared-source grouping, and clustering on
    # it would leave the survey with three folds.
    ("medqa_usmle", "GBaker/MedQA-USMLE-4-options", "default", "train",
     from_letter("options", "answer_idx"), 0),
    ("openbookqa", "allenai/openbookqa", "main", "train", from_labelled_choices(), 0),
    ("arc_challenge", "allenai/ai2_arc", "ARC-Challenge", "test", from_labelled_choices(), 0),
]


def key_index_for(name, row):
    """MedMCQA keys by integer index into opa..opd rather than by content."""
    if name != "medmcqa":
        return None
    try:
        return int(row["cop"])
    except (KeyError, TypeError, ValueError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="*", help="fetch only these targets")
    args = parser.parse_args()

    import pyarrow.parquet as pq

    ROOT.mkdir(parents=True, exist_ok=True)
    entries = []
    for name, dataset, config, split, extract, cap in TARGETS:
        if args.only and name not in args.only:
            continue
        listing = API.format(ds=urllib.parse.quote(dataset), cfg=config, split=split)
        try:
            urls = json.loads(get(listing))
        except Exception as exc:                       # noqa: BLE001 - reported, not hidden
            print(f"  {name}: listing failed ({type(exc).__name__}); skipped")
            continue

        rows = []
        for url in urls:
            table = pq.read_table(io.BytesIO(get(url)))
            rows.extend(table.to_pylist())
            if cap and len(rows) >= cap:
                rows = rows[:cap]
                break

        records = []
        for row in rows:
            index = key_index_for(name, row)
            if index is not None:
                options = [row.get(f) for f in ("opa", "opb", "opc", "opd")]
                if any(o is None for o in options) or not 0 <= index < len(options):
                    continue
                options = [str(o) for o in options]
                pair = ([options[index]] + options[:index] + options[index + 1:],
                        _cluster(row, "subject_name"))
            else:
                pair = extract(row)
            if pair is None:
                continue
            options, cluster = pair
            if len(options) < 2 or len(set(options)) != len(options):
                continue
            records.append({"options": options, "key": 0, "n_options": len(options),
                            "cluster": cluster})

        # Gzipped: the vendored option sets are committed so the survey replays
        # offline, and compressed they are a few megabytes rather than twenty.
        path = ROOT / f"{name}.jsonl.gz"
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")
        blob = path.read_bytes()
        entries.append({"name": name, "local_path": str(path), "dataset": dataset,
                        "config": config, "split": split, "listing_url": listing,
                        "parquet_urls": urls, "n_items": len(records),
                        "n_rows_read": len(rows), "row_cap": cap or None,
                        "sha256": hashlib.sha256(blob).hexdigest(), "bytes": len(blob)})
        print(f"  {name:14s} {len(records):6d} items from {len(rows)} rows  "
              f"{len(blob) // 1024} KB")

    # A partial re-fetch must not drop the other files' provenance, and no re-fetch drops the licenses recorded
    # for each file (read from each dataset's card and kept by hand; nothing here fetches them).
    existing, licenses, previous = {}, {}, {}
    record_path = ROOT / "PROVENANCE.json"
    if record_path.exists():
        previous = json.loads(record_path.read_text(encoding="utf-8"))
        licenses = {e["name"]: e["license"] for e in previous.get("files", []) if e.get("license")}
        if args.only:
            existing = {e["name"]: e for e in previous.get("files", [])}
    for entry in entries:
        if entry["name"] in licenses:                 # after the split, where the record keeps it
            items = list(entry.items())
            at = [k for k, _ in items].index("split") + 1
            entry = dict(items[:at] + [("license", licenses[entry["name"]])] + items[at:])
        existing[entry["name"]] = entry
    entries = [existing[name] for name in sorted(existing)]

    doc = {"_comment": ("Option sets only, vendored from public multiple-choice benchmarks so "
                        "the option-geometry survey replays offline. Questions, rationales and "
                        "explanations are deliberately not retained. The SHA-256 is of the "
                        "gzipped file as written here, not of the upstream parquet."),
           "retrieved_at": datetime.datetime.now(datetime.timezone.utc)
                                   .strftime("%Y-%m-%dT%H:%M:%SZ"),
           "note": ("Idavidrein/gpqa is gated and returns HTTP 401 without credentials, so it "
                    "is absent. SciBench and TheoremQA are open-answer and carry no distractors."),
           **({"license": previous["license"]} if previous.get("license") else {}),
           "files": entries}
    record_path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    print(f"\nrecorded {len(entries)} benchmark files in {record_path}")


if __name__ == "__main__":
    main()
