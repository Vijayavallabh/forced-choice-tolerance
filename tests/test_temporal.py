"""Tests for the date recovery and the temporal test it makes possible.

Nothing here touches the network: the resolver is exercised through its cache,
which is the same path a reproduction takes once ``build/source_dates.json``
ships.
"""
import json
import tempfile
import unittest
from pathlib import Path

from subset_guarantee import key_rank_histogram
from temporal_recovery import _date_parts, resolve
from temporal_test import spearman


class DateParsingTests(unittest.TestCase):
    def test_a_year_only_record_is_read_as_its_first_day(self):
        self.assertEqual(_date_parts({"date-parts": [[2019]]}), "2019-01-01")

    def test_a_year_and_month_record_fills_the_day(self):
        self.assertEqual(_date_parts({"date-parts": [[2019, 7]]}), "2019-07-01")

    def test_a_full_record_is_zero_padded(self):
        self.assertEqual(_date_parts({"date-parts": [[2019, 7, 4]]}), "2019-07-04")

    def test_an_empty_record_is_not_a_date(self):
        self.assertIsNone(_date_parts({"date-parts": [[None]]}))
        self.assertIsNone(_date_parts(None))


class ResolverTests(unittest.TestCase):
    def test_a_cached_answer_is_returned_without_a_request(self):
        cache = {"doi|10.1234/abc": {"date": "2020-01-02", "field": "issued"}}
        # timeout 0 and sleep 0: if it tried the network this would raise.
        self.assertEqual(resolve("doi", "10.1234/abc", cache, 0.0, 0.0)["date"],
                         "2020-01-02")

    def test_a_label_identifier_is_recorded_as_unresolvable_rather_than_fetched(self):
        cache = {}
        answer = resolve("label", "some-repo-name", cache, 0.0, 0.0)
        self.assertIn("error", answer)
        self.assertNotIn("date", answer)

    def test_a_url_carrying_a_doi_is_resolved_as_that_doi(self):
        cache = {"doi|10.1128/msphere.00109-24": {"date": "2024-04-01",
                                                  "field": "issued"}}
        answer = resolve("url", "https://doi.org/10.1128/msphere.00109-24",
                         cache, 0.0, 0.0)
        self.assertEqual(answer["date"], "2024-04-01")


class SpearmanTests(unittest.TestCase):
    def test_a_perfect_increase_is_one(self):
        self.assertAlmostEqual(spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)

    def test_a_perfect_decrease_is_minus_one(self):
        self.assertAlmostEqual(spearman([1, 2, 3, 4], [40, 30, 20, 10]), -1.0)

    def test_ties_take_the_average_rank_rather_than_the_first(self):
        # Without average ranks this returns something other than zero.
        self.assertAlmostEqual(spearman([1, 1, 2, 2], [5, 5, 5, 5]), 0.0)

    def test_dates_sort_as_strings_in_this_format(self):
        dates = ["2019-12-31", "2020-01-01", "2020-01-02"]
        self.assertAlmostEqual(spearman(dates, [1.0, 2.0, 3.0]), 1.0)


class SubsetGuaranteeTests(unittest.TestCase):
    def test_the_histogram_counts_the_key_s_rank_among_the_values(self):
        rows = [{"ideal": "3", "distractors": ["1", "2", "4"], "cluster": "a",
                 "question": ""},
                {"ideal": "1", "distractors": ["2", "3", "4"], "cluster": "a",
                 "question": ""}]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "file.jsonl"
            path.write_text("".join(json.dumps(r) + "\n" for r in rows),
                            encoding="utf-8")
            n, histogram = key_rank_histogram(path, 4, "cluster")
        self.assertEqual(n, 2)
        self.assertEqual(histogram, [50.0, 0.0, 50.0, 0.0])

    def test_items_whose_options_are_not_distinct_numbers_are_skipped(self):
        rows = [{"ideal": "3", "distractors": ["3", "2", "4"], "cluster": "a",
                 "question": ""},
                {"ideal": "x", "distractors": ["y", "z", "w"], "cluster": "a",
                 "question": ""}]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "file.jsonl"
            path.write_text("".join(json.dumps(r) + "\n" for r in rows),
                            encoding="utf-8")
            n, _ = key_rank_histogram(path, 4, "cluster")
        self.assertEqual(n, 0)


if __name__ == "__main__":
    unittest.main()
