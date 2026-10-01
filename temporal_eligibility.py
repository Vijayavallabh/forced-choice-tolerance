#!/usr/bin/env python3
"""Conservative date-based pair eligibility; not a contamination detector."""
import argparse
import calendar
from datetime import date
import json


def add_months(value, months):
    offset = value.year * 12 + value.month - 1 + months
    year, month = divmod(offset, 12)
    month += 1
    return date(year, month, min(value.day, calendar.monthrange(year, month)[1]))


def before(material, cutoff):
    if not material.get("evidence_urls"):
        return "unknown"
    if material.get("first_appearance_verified") and material.get("first_public"):
        return "eligible" if date.fromisoformat(material["first_public"]) < cutoff else "ineligible"
    if not material.get("public_by"):
        return "unknown"
    # A later public-by bound cannot exclude an earlier, undocumented appearance.
    return "eligible" if date.fromisoformat(material["public_by"]) < cutoff else "unknown"


def after(material, boundary):
    public_by = material.get("public_by")
    if public_by and material.get("evidence_urls") and date.fromisoformat(public_by) <= boundary:
        return "ineligible"
    if not (material.get("first_appearance_verified") and material.get("first_public")
            and material.get("evidence_urls")):
        return "unknown"
    return "eligible" if date.fromisoformat(material["first_public"]) > boundary else "ineligible"


def eligibility(original, twin, cutoff, margin_months=3, novelty="data_novel"):
    if novelty not in {"data_novel", "analysis_novel"} or margin_months < 0:
        raise ValueError("Invalid novelty class or margin")
    if cutoff is None:
        return {"status": "unknown", "reason": "No documented model cutoff"}
    for material in (original["analysis"], twin["analysis"], twin["data"]):
        if material.get("first_appearance_verified") and material.get("first_public") and material.get("public_by"):
            if date.fromisoformat(material["first_public"]) > date.fromisoformat(material["public_by"]):
                raise ValueError("First appearance cannot follow an evidenced public-by date")
    cutoff = date.fromisoformat(cutoff)
    boundary = add_months(cutoff, margin_months)
    checks = {"original_analysis": before(original["analysis"], cutoff),
              "twin_analysis": after(twin["analysis"], boundary)}
    if novelty == "data_novel":
        checks["twin_data"] = after(twin["data"], boundary)
    status = "ineligible" if "ineligible" in checks.values() else (
        "eligible" if all(v == "eligible" for v in checks.values()) else "unknown")
    return {"status": status, "checks": checks, "boundary": boundary.isoformat(),
            "novelty": novelty, "scope": "Date eligibility only; artifact/tool/post-training exposure requires separate review"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pair_json", help="Object with original, twin, cutoff, and optional margin_months/novelty")
    args = parser.parse_args()
    with open(args.pair_json) as handle:
        result = eligibility(**json.load(handle))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
