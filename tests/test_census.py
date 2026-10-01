import json
import tempfile
import unittest
from pathlib import Path

from benchmark_census import census, is_missing, load_rows, source_key


def write_jsonl(rows):
    handle = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8")
    for row in rows:
        handle.write(json.dumps(row) + "\n")
    handle.close()
    return Path(handle.name)


class MissingValueTests(unittest.TestCase):
    def test_recognised_missing_tokens(self):
        for value in (None, "", "  ", "none", "N/A", "not available", "null", [], {}):
            self.assertTrue(is_missing(value), value)

    def test_real_values_are_not_missing(self):
        for value in ("10.1234/x", "0", 0, ["a"], "nan-like-name"):
            self.assertFalse(is_missing(value), value)


class SourceKeyTests(unittest.TestCase):
    def test_doi_is_extracted_from_a_url(self):
        self.assertEqual(source_key({"paper": "https://doi.org/10.1172/jci.insight.167744"}),
                         ("doi", "10.1172/jci.insight.167744"))

    def test_list_valued_source_field_uses_its_first_entry(self):
        self.assertEqual(source_key({"sources": ["https://doi.org/10.1128/msphere.00109-24"]}),
                         ("doi", "10.1128/msphere.00109-24"))

    def test_absent_source_returns_none(self):
        self.assertIsNone(source_key({"question": "q", "ideal": "a", "source": None}))

    def test_bare_label_is_kept_but_marked_as_a_label(self):
        kind, _ = source_key({"github_name": "deepchem/deepchem"})
        self.assertEqual(kind, "label")


class CensusTests(unittest.TestCase):
    def test_a_file_with_no_dates_is_reported_as_undated(self):
        path = write_jsonl([{"id": i, "question": "q", "paper": "10.1234/abc"} for i in range(4)])
        try:
            rec = census("toy", path)
            self.assertFalse(rec["publishes_any_date"])
            self.assertEqual(rec["source_identifier"]["coverage"], 1.0)
        finally:
            path.unlink()

    def test_a_date_valued_field_is_detected_even_when_oddly_named(self):
        path = write_jsonl([{"id": i, "released": "2024-03-01"} for i in range(4)])
        try:
            rec = census("toy", path)
            self.assertTrue(rec["publishes_any_date"])
            self.assertIn("released", rec["date_fields_by_value"])
        finally:
            path.unlink()

    def test_dependence_counts_tasks_sharing_one_source(self):
        rows = [{"id": 1, "paper": "10.1/a"}, {"id": 2, "paper": "10.1/a"},
                {"id": 3, "paper": "10.1/a"}, {"id": 4, "paper": "10.1/b"}]
        path = write_jsonl(rows)
        try:
            dep = census("toy", path)["dependence"]
            self.assertEqual(dep["n_distinct_sources"], 2)
            self.assertEqual(dep["max_items_per_source"], 3)
        finally:
            path.unlink()

    def test_unidentified_tasks_are_not_merged_into_one_pseudo_source(self):
        path = write_jsonl([{"id": i, "question": "q"} for i in range(5)])
        try:
            dep = census("toy", path)["dependence"]
            self.assertEqual(dep["n_distinct_sources"], 0)
            self.assertEqual(dep["n_items_without_source"], 5)
        finally:
            path.unlink()


class ReleasedCensusTests(unittest.TestCase):
    """The headline census claim, checked against the generated result."""

    def setUp(self):
        path = Path("results/benchmark_census.json")
        if not path.exists():
            self.skipTest("census not generated")
        self.doc = json.loads(path.read_text())

    def test_no_released_file_publishes_a_date(self):
        self.assertEqual(self.doc["summary"]["n_files_publishing_any_date"], 0)
        for rec in self.doc["benchmarks"]:
            self.assertFalse(rec["publishes_any_date"], rec["benchmark"])

    def test_census_covers_the_reported_task_total(self):
        total = sum(r["n_items"] for r in self.doc["benchmarks"])
        self.assertEqual(total, self.doc["summary"]["n_items_total"])
        self.assertEqual(self.doc["summary"]["n_benchmark_files"], 12)

    def test_bixbench_row_matches_the_standalone_audit(self):
        audit = json.loads(Path("results/audit.json").read_text())
        row = next(r for r in self.doc["benchmarks"] if r["benchmark"] == "BixBench v1.5")
        self.assertEqual(row["n_items"], audit["n_questions"])
        self.assertEqual(row["dependence"]["n_distinct_sources"], audit["n_capsules"])


if __name__ == "__main__":
    unittest.main()
