#!/usr/bin/env python3
"""Export conservative temporal fields from the saved bibliographic audit."""
import hashlib
import json
from pathlib import Path
from audit_provenance import DATASET_URL, REVISION


def unknown():
    return {"public_by": None, "first_public": None,
            "first_appearance_verified": False, "evidence_urls": []}


def main():
    audit = json.loads(Path("results/audit.json").read_text())
    groups = {}
    for group in audit["explicit_source_groups"]:
        group_id = hashlib.sha256("\n".join(group["identifiers"]).encode()).hexdigest()[:12]
        groups.update({uuid: group_id for uuid in group["capsules"]})
    tasks = []
    for uuid, capsule in sorted(audit["capsules"].items()):
        analysis = unknown()
        records = audit.get("date_resolution", {}).get("records", {})
        evidence = [item for identifier in capsule["identifiers"]
                    for item in records.get(identifier, {}).get("evidence", [])
                    if item["kind"] == "publication" and item["provider"] == "crossref"]
        # Repository timestamps alone do not certify publication of a source analysis.
        if evidence:
            analysis["public_by"] = min(item["upper"] for item in evidence)
            analysis["evidence_urls"] = sorted({item["evidence_url"] for item in evidence})
        tasks.append({"task_id": uuid, "source_group": groups.get(uuid),
                      "analysis": analysis, "data": unknown(), "artifact": unknown()})
    manifest = {"schema_version": "1.0", "benchmark": {"name": "BixBench public v1.5",
                "revision": REVISION, "sha256": audit["dataset_sha256"], "source_url": DATASET_URL},
                "tasks": tasks}
    Path("results/temporal_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Exported {len(tasks)} tasks; no first-appearance or data-version certification inferred.")


if __name__ == "__main__":
    main()
