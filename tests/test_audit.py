import unittest
from audit_provenance import audit, classify, collapse_to_capsules, date_interval, evidence_from_record, source_identifiers


def row(uuid, paper):
    return {"capsule_uuid": uuid, "paper": paper, "categories": [], "eval_mode": "str_verifier"}


class AuditTests(unittest.TestCase):
    def test_embedded_doi_is_not_counted_as_url(self):
        self.assertEqual(classify("https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0305268"), "doi")

    def test_all_sources_and_fragments(self):
        identifiers = source_identifiers("https://zenodo.org/records/8036465, https://www.science.org/doi/10.1126/sciadv.adf4950")
        self.assertEqual(identifiers, ["10.1126/sciadv.adf4950", "10.5281/zenodo.8036465"])
        self.assertEqual(source_identifiers("https://www.nature.com/articles/s41467-022-29205-8#Abs1"), ["10.1038/s41467-022-29205-8"])

    def test_conflicting_capsule_provenance_fails(self):
        with self.assertRaises(ValueError):
            collapse_to_capsules([row("a", None), row("a", "10.1234/example")])

    def test_created_is_not_publication(self):
        record = {"status": 200, "url": "fixture", "body": {"data": {"attributes": {
            "publicationYear": 2025, "created": "2024-06-21T00:00:00Z"}}}}
        evidence = evidence_from_record(record, "datacite")
        publication = [e for e in evidence if e["kind"] == "publication"]
        self.assertEqual([e["upper"] for e in publication], ["2025-12-31"])

    def test_partial_date_uncertainty(self):
        self.assertEqual(date_interval("2024")['upper'], "2024-12-31")
        self.assertEqual(date_interval("2024-02")['upper'], "2024-02-29")

    def test_transitive_source_dependence(self):
        rows = [row("a", "10.1234/one"), row("b", "10.1234/two"),
                row("c", "10.1234/one, 10.1234/two"), row("d", "Not Available")]
        result = audit(rows, b"fixture")
        self.assertEqual(len(result["explicit_source_groups"]), 1)
        self.assertEqual(result["explicit_source_groups"][0]["capsules"], ["a", "b", "c"])
        self.assertEqual(result["classes"]["missing"]["questions"], 1)

    def test_schema_is_inspected(self):
        record = row("a", "")
        record["source_date"] = "2024-01-01"
        self.assertEqual(audit([record], b"fixture")["date_like_top_level_fields"], ["source_date"])


if __name__ == "__main__":
    unittest.main()
