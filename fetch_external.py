#!/usr/bin/env python3
"""Re-fetch the vendored external evidence and record its provenance.

The analyses in this repository read only from ``data/external/``, which is
committed so that every result reproduces offline. This script documents and
re-performs the retrieval, writing a URL and SHA-256 for each file into
``data/external/PROVENANCE.json``.

Run it only to refresh the evidence. Upstream files can change, so a refresh
that alters a digest will make ``validate_artifact.py`` fail until the new
bytes are reviewed and the manuscript's numbers are regenerated. That failure
is the point: it is the check the benchmarks under study do not provide for
themselves.

Parquet conversion needs ``pyarrow``; nothing else here does, and no other
script in the repository imports it. Image and table payloads are dropped, so
what lands in ``data/external/labbench`` is metadata only.
"""
import argparse
import datetime
import hashlib
import json
import urllib.request
from pathlib import Path

BIXBENCH_REPO = "https://api.github.com/repos/Future-House/BixBench/commits/main"
BIXBENCH_RAW = "https://raw.githubusercontent.com/Future-House/BixBench/{commit}/{path}"
HF = "https://huggingface.co/datasets/{repo}/resolve/{revision}/{path}"

ZERO_SHOT_V15 = [
    "claude-3-5-sonnet-latest-grader-mcq-refusal-False",
    "claude-3-5-sonnet-latest-grader-mcq-refusal-True",
    "claude-3-5-sonnet-latest-grader-openended",
    "gpt-4o-grader-mcq-refusal-False",
    "gpt-4o-grader-mcq-refusal-True",
    "gpt-4o-grader-openended",
]
ZERO_SHOT_V10 = [
    "bixbench_llm_baseline_refusal_False_mcq_claude-3-5-sonnet-latest_1.0",
    "bixbench_llm_baseline_refusal_False_mcq_gpt-4o_1.0",
    "bixbench_llm_baseline_refusal_True_openended_claude-3-5-sonnet-latest_1.0",
    "bixbench_llm_baseline_refusal_True_openended_gpt-4o_1.0",
]
LABBENCH_SUBSETS = ["LitQA2", "SuppQA", "DbQA", "SeqQA", "TableQA", "FigQA",
                    "ProtocolQA", "CloningScenarios"]
OTHER = [
    ("osunlp/ScienceAgentBench", "main", "ScienceAgentBench.csv",
     "other/ScienceAgentBench.csv"),
    ("siegelz/core-bench", "main", "core_train.json", "other/core_train.json"),
    ("allenai/discoverybench", "main", "answer_key/answer_key_real.csv",
     "other/discoverybench_answer_key_real.csv"),
]
# Columns whose values are encoded images or tables rather than metadata.
BLOB_COLUMNS = {"figure", "image", "images", "figures", "img", "table_image", "tables"}

ROOT = Path("data/external")


def get(url, timeout=120):
    request = urllib.request.Request(url, headers={"User-Agent": "TATE-provenance-audit/3.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def save(blob, relative):
    path = ROOT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(blob)
    return path


def parquet_to_jsonl(blob, relative):
    """Keep the text metadata; drop encoded figure and table payloads."""
    import io

    import pyarrow.parquet as pq

    table = pq.read_table(io.BytesIO(blob))
    keep = [c for c in table.column_names if c.lower() not in BLOB_COLUMNS]
    lines = []
    for row in table.select(keep).to_pylist():
        clean = {}
        for key, value in row.items():
            if isinstance(value, (bytes, bytearray)):
                continue
            if isinstance(value, list) and value and isinstance(value[0], (bytes, bytearray)):
                continue
            clean[key] = value
        if "table-path" in clean:
            clean["n_tables"] = len(clean.pop("table-path") or [])
        lines.append(json.dumps(clean, default=str))
    return save(("\n".join(lines) + "\n").encode("utf-8"), relative)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-labbench", action="store_true",
                        help="skip the parquet subsets (avoids the pyarrow dependency)")
    args = parser.parse_args()

    commit = json.loads(get(BIXBENCH_REPO))["sha"]
    print(f"BixBench results repository pinned at {commit}")
    entries = []

    def record(path, url):
        blob = path.read_bytes()
        entries.append({"local_path": str(path), "url": url,
                        "sha256": hashlib.sha256(blob).hexdigest(), "bytes": len(blob)})
        print(f"  {path}  {len(blob) // 1024} KB")

    for name in ZERO_SHOT_V15:
        remote = f"bixbench-v1.5_results/zero_shot_baselines/{name}.csv"
        url = BIXBENCH_RAW.format(commit=commit, path=remote)
        record(save(get(url), f"zero_shot_v15/{name}.csv"), url)
    for name in ZERO_SHOT_V10:
        remote = f"bixbench_results/baseline_eval_data/{name}.csv"
        url = BIXBENCH_RAW.format(commit=commit, path=remote)
        record(save(get(url), f"zero_shot_v10/{name}.csv"), url)
    for local, remote in (("zero_shot_summary_v15.json",
                           "bixbench-v1.5_results/zero_shot_baselines.json"),
                          ("zero_shot_summary_v10.json",
                           "bixbench_results/zero_shot_baselines.json")):
        url = BIXBENCH_RAW.format(commit=commit, path=remote)
        record(save(get(url), local), url)

    if not args.skip_labbench:
        for subset in LABBENCH_SUBSETS:
            url = HF.format(repo="futurehouse/lab-bench", revision="main",
                            path=f"{subset}/train-00000-of-00001.parquet")
            record(parquet_to_jsonl(get(url), f"labbench/{subset}.jsonl"), url)

    for repo, revision, remote, local in OTHER:
        url = HF.format(repo=repo, revision=revision, path=remote)
        record(save(get(url), local), url)

    doc = {
        "_comment": ("Evidence vendored from public benchmark repositories. Each entry records "
                     "the exact source URL and the SHA-256 of the retrieved bytes so the analysis "
                     "replays offline and can be re-verified against upstream."),
        "upstream_repository": "https://github.com/Future-House/BixBench",
        "upstream_commit": commit,
        "license": ("Each source keeps its own license; THIRD_PARTY_NOTICES.md has the details. "
                    "BixBench (zero_shot_*): Apache-2.0, Copyright 2025 FutureHouse. "
                    "LAB-Bench (labbench/): CC BY-SA 4.0, adapted, so the adapted files are "
                    "CC BY-SA 4.0 too; see labbench/LICENSE. ScienceAgentBench.csv: CC BY 4.0, "
                    "except that tasks adapted from rasterio and matminer keep those projects' "
                    "licenses. discoverybench_answer_key_real.csv: ODC-By. core_train.json: MIT."),
        "retrieved_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "note": ("LAB-Bench subsets are converted from parquet to JSONL with encoded figure and "
                 "table payloads removed, so the recorded SHA-256 is of the converted file and "
                 "the URL is of the parquet source."),
        "files": entries,
    }
    (ROOT / "PROVENANCE.json").write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    print(f"\nrecorded {len(entries)} files in {ROOT / 'PROVENANCE.json'}")


if __name__ == "__main__":
    main()
