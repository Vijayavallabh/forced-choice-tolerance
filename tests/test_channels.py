"""Tests for the multi-channel view: several orderings of the same options.

The claim these support is that \\S2's bound is about *an* ordering, not about
numbers, so it applies to each coordinate separately -- and that a repair which
uniformises one coordinate has to be checked against the others.
"""
import json
import random
import unittest
from pathlib import Path

from channel_attribution import CHANNELS, credit, ranks_from_scores
from mcq_audit import (ORDERINGS, SURFACE_ORDERINGS, feature_rank, isolation_rank,
                       isolation_scores, length_scores, ordering_rank,
                       repair_row_multi, value_scores)

ATTRIBUTION = Path("results/channel_attribution_mmlupro.json")


class OrderingTests(unittest.TestCase):
    def test_the_attribution_measures_at_least_what_the_repair_fixes(self):
        """Measuring more than you repair is fine; the reverse is not."""
        self.assertTrue(set(ORDERINGS) <= set(CHANNELS),
                        f"unrepaired-but-audited: {set(ORDERINGS) - set(CHANNELS)}")
        self.assertNotIn("sorted value", SURFACE_ORDERINGS)

    def test_roundness_orders_hand_written_numbers_below_generated_ones(self):
        from mcq_audit import significant_digits as _significant_digits
        self.assertLess(_significant_digits("5.0"), _significant_digits("4.37"))
        self.assertLess(_significant_digits("100"), _significant_digits("103.6"))
        self.assertEqual(_significant_digits("0.05"), _significant_digits("5"))
        self.assertEqual(_significant_digits("-25%"), _significant_digits("25"))
        # U+2212, which MMLU-Pro writes and parse_number accepts. Counting the
        # sign as a digit made every such option read one digit less round.
        self.assertEqual(_significant_digits("\u221212.5"), _significant_digits("-12.5"))
        self.assertEqual(_significant_digits("\u22120.0400"), 1.0)

    def test_isolation_does_not_depend_on_the_python_version(self):
        """The score is a sum of floats, and ``sum`` changed under us.

        CPython 3.12 sums floats with Neumaier compensation and earlier versions
        do not. These four options score -1.5 exactly one way and
        -1.4999999999999998 the other, which moves the key's isolation rank --
        and with it a published channel number -- between interpreters. 815 of
        the 25999 option sets in this repository are on such a knife edge.
        ``math.fsum`` is exactly rounded everywhere, and agrees with 3.12.
        """
        from mcq_audit import isolation_rank, isolation_scores
        options = ["0.1126", "0.0867", "0.0713", "0.1433"]
        scores = isolation_scores(options)
        self.assertEqual(scores[0], -1.5)
        self.assertEqual(scores[1], -1.5)
        self.assertEqual(isolation_rank(options), 2)

    def test_the_rule_family_scores_the_same_way_the_audit_does(self):
        """text_artifacts._odd_one_out is the same statistic on the other side."""
        import math
        import text_artifacts
        options = ["0.1126", "0.0867", "0.0713", "0.1433"]
        for i, option in enumerate(options):
            others = [o for j, o in enumerate(options) if j != i]
            self.assertEqual(
                text_artifacts.RULES["least_like_other_options"](option, others, ""),
                -math.fsum(text_artifacts.similarity(option, other)
                           for other in others))

    def test_ranks_are_a_permutation(self):
        options = ["1.0", "22", "333", "4444"]
        for name, score in ORDERINGS.items():
            scores = score(options)
            self.assertIsNotNone(scores, name)
            self.assertEqual(sorted(ranks_from_scores(options, scores)), [0, 1, 2, 3], name)

    def test_ties_break_deterministically_on_the_text(self):
        """Two options of equal length must still get distinct ranks."""
        options = ["ab", "cd", "ef"]
        self.assertEqual(sorted(ranks_from_scores(options, length_scores(options))),
                         [0, 1, 2])
        self.assertEqual(ranks_from_scores(options, length_scores(options)),
                         ranks_from_scores(options, length_scores(options)))

    def test_value_ordering_refuses_a_non_numeric_set(self):
        self.assertIsNone(value_scores(["a", "b", "c", "d"]))
        self.assertIsNone(value_scores(["1", "1", "2", "3"]))
        self.assertIsNone(ordering_rank(["a", "b", "c", "d"], "sorted value"))

    def test_the_odd_option_out_ranks_last_on_isolation(self):
        options = ["0.101", "0.102", "0.103", "a totally different string"]
        self.assertEqual(isolation_rank(options, 3), 3)
        self.assertLess(feature_rank(options, isolation_scores(options), 0), 3)


class CreditTests(unittest.TestCase):
    def test_a_uniform_key_distribution_pays_nothing_on_any_channel(self):
        rng = random.Random(7)
        for k in (4, 10):
            uniform = [1 / k] * k
            for _ in range(40):
                weights = [rng.random() for _ in range(k)]
                total = sum(weights)
                b = [w / total for w in weights]
                self.assertAlmostEqual(credit(uniform, b), 0.0, places=12)

    def test_a_point_mass_attains_the_bound(self):
        p = [0.124, 0.514, 0.295, 0.067]
        best = [0.0, 1.0, 0.0, 0.0]
        self.assertAlmostEqual(credit(p, best), max(p) - 0.25, places=12)
        worst = [0.0, 0.0, 0.0, 1.0]
        self.assertAlmostEqual(credit(p, worst), min(p) - 0.25, places=12)


class MultiChannelRepairTests(unittest.TestCase):
    OPTIONS = ["5.0", "4.37", "6.21", "3.88", "7.02", "2.95", "8.14", "9.33", "1.76", "10.5"]

    def test_the_keyed_answer_is_never_altered(self):
        rng = random.Random(3)
        for _ in range(8):
            result = repair_row_multi(self.OPTIONS, rng)
            self.assertIsNotNone(result)
            self.assertEqual(result[0][0], self.OPTIONS[0])

    def test_it_reaches_the_value_rank_it_reports(self):
        rng = random.Random(4)
        for _ in range(8):
            options, value_rank, achieved, targets = repair_row_multi(self.OPTIONS, rng)
            self.assertEqual(ordering_rank(options, "sorted value"), value_rank)
            for name, rank in achieved.items():
                self.assertEqual(ordering_rank(options, name), rank)

    def test_it_reports_targets_it_missed_rather_than_claiming_them(self):
        """The surface ranks are a search result, not a guarantee."""
        rng = random.Random(5)
        missed = 0
        for _ in range(30):
            _, _, achieved, targets = repair_row_multi(self.OPTIONS, rng)
            missed += any(achieved[name] != targets[name] for name in achieved)
        self.assertGreater(missed, 0, "no target was ever missed; the test is not exercised")

    def test_the_search_places_the_surface_ranks_better_than_chance(self):
        """What the extra coordinates buy: the drawn rank is hit, not stumbled on.

        Whether the numeric-only repair *concentrates* a surface rank depends on
        the benchmark's own writing, so that comparison is checked against the
        released files in ``validate_artifact.py``. What can be checked here is
        the generator: a search over template schemes must hit a uniformly drawn
        surface rank more often than the 1/k it would get by accident.
        """
        from mcq_audit import repair_row
        rng = random.Random(6)
        k = len(self.OPTIONS)
        searched = accidental = trials = 0
        for _ in range(40):
            _, _, achieved, targets = repair_row_multi(self.OPTIONS, rng)
            searched += achieved["lexical isolation"] == targets["lexical isolation"]
            # the same target, against a repair that does not look at it
            accidental += isolation_rank(repair_row(self.OPTIONS, rng)) == \
                targets["lexical isolation"]
            trials += 1
        self.assertGreater(searched / trials, 1.5 / k,
                           "the search is no better than chance at its own target")
        self.assertGreater(searched, accidental)


class ReportedAttributionTests(unittest.TestCase):
    """Guard the manuscript's per-channel numbers against the results file."""

    @classmethod
    def setUpClass(cls):
        if not ATTRIBUTION.exists():
            raise unittest.SkipTest("attribution not generated")
        cls.doc = json.loads(ATTRIBUTION.read_text())

    @staticmethod
    def _collects(record):
        """Whether this solver's value-channel credit clears zero as released."""
        value = record["arms"]["original_stemless"]["channels"]["sorted value"]
        return value["credit_collected_ci95"][0] > 0

    def test_exactly_one_released_solver_collects_and_it_is_the_largest(self):
        """The paper's one positive exploitation result, kept honest.

        Two of the three ten-option solvers are at chance on every coordinate
        with the question withheld. Qwen2.5-32B is not: it takes about a fifth
        of the value channel. The claim is that the channel is *largely*
        unexploited, so the test is that the exception stays a single exception
        and stays the most capable model.
        """
        collecting = [r["model"] for r in self.doc["models"] if self._collects(r)]
        self.assertEqual(len(collecting), 1, collecting)
        self.assertIn("32B", collecting[0])
        value = next(r for r in self.doc["models"]
                     if self._collects(r))["arms"]["original_stemless"]["channels"]["sorted value"]
        self.assertLess(value["credit_collected"], 0.3 * value["credit_available"],
                        "the collector now takes more than a third of the channel")

    def test_the_other_released_solvers_are_at_chance_on_every_channel(self):
        for record in self.doc["models"]:
            if self._collects(record):
                continue
            arm = record["arms"]["original_stemless"]
            for name, channel in arm["channels"].items():
                self.assertLess(abs(channel["credit_collected"]), 0.02,
                                f"{record['model']} collects {channel['credit_collected']:+.3f} "
                                f"on {name} with the question withheld")

    def test_the_numeric_repair_leaves_the_solver_above_chance(self):
        """The finding that motivates repairing every ordering."""
        chance = 1.0 / self.doc["models"][0]["n_options"]
        for record in self.doc["models"]:
            released = record["arms"]["original_stemless"]["accuracy"]
            repaired = record["arms"]["repaired_stemless"]["accuracy"]
            self.assertGreater(repaired, chance + 0.02, record["model"])
            if self._collects(record):
                continue
            self.assertLess(abs(released - chance), 0.02, record["model"])
            self.assertGreater(repaired, released + 0.02, record["model"])

    def test_the_value_channel_is_the_one_the_repair_closes(self):
        for record in self.doc["models"]:
            before = record["arms"]["original_stemless"]["channels"]["sorted value"]
            after = record["arms"]["repaired_stemless"]["channels"]["sorted value"]
            self.assertLess(after["credit_available"], before["credit_available"] / 2,
                            record["model"])


class SearchTests(unittest.TestCase):
    """The repair's search, which turned out to matter more than its specification."""

    OPTIONS = ["5.0", "4.37", "6.21", "3.88", "7.02", "2.95", "8.14", "9.33", "1.76", "10.5"]

    def test_the_style_variants_span_written_lengths_for_one_value(self):
        from mcq_audit import _template_variants, format_like
        variants = _template_variants("4.37", "5.0")
        rendered = {format_like(template, 4.37) for template in variants}
        self.assertGreater(len({len(text) for text in rendered}), 2,
                           "one value renders at only one or two written lengths; "
                           "the search has no lever on the length coordinate")

    def test_rounding_a_value_lowers_its_significant_digits(self):
        from mcq_audit import _round_significant, significant_digits
        self.assertEqual(_round_significant(12345.678, 2), 12000.0)
        self.assertLess(significant_digits(str(_round_significant(12345.678, 2))),
                        significant_digits("12345.678"))

    def test_padding_a_round_value_does_not_raise_its_significant_digits(self):
        """Why the search needs to move values and not only renderings.

        Trailing zeros are not significant, so no template makes 15 read as six
        figures. Lowering the key's roundness rank needs distractors that are
        genuinely less round, which is what the jitter move supplies.
        """
        from mcq_audit import format_like, significant_digits
        self.assertEqual(significant_digits(format_like("1234.0000", 15.0)),
                         significant_digits("15"))

    def test_the_guided_search_meets_more_targets_than_the_sampler(self):
        met = {}
        for label, refine in (("guided", True), ("sample", False)):
            rng = random.Random(21)
            hits = 0
            for _ in range(12):
                _, _, achieved, targets = repair_row_multi(self.OPTIONS, rng,
                                                           refine=refine)
                hits += achieved["written length"] == targets["written length"]
            met[label] = hits
        self.assertGreater(met["guided"], met["sample"],
                           f"the guided search is no better on its hardest "
                           f"coordinate: {met}")

    def test_the_guided_search_still_never_touches_the_keyed_answer(self):
        """Rounding and jitter move distractor values; the key is not a distractor."""
        rng = random.Random(22)
        for _ in range(12):
            options, value_rank, _, _ = repair_row_multi(self.OPTIONS, rng)
            self.assertEqual(options[0], self.OPTIONS[0])
            self.assertEqual(ordering_rank(options, "sorted value"), value_rank)
            self.assertEqual(len(set(options)), len(self.OPTIONS))

    def test_the_tilt_asks_more_often_for_ranks_a_pass_under_produced(self):
        from collections import Counter

        from mcq_audit import _tilt
        landed = Counter({0: 70, 1: 10, 2: 10, 3: 10})
        weights = _tilt(None, landed, 4)
        self.assertAlmostEqual(sum(weights), 1.0, places=12)
        self.assertLess(weights[0], weights[1])
        self.assertAlmostEqual(weights[1], weights[2], places=12)

    def test_the_target_distribution_can_be_fitted_from_a_string_seed(self):
        """The probe seeds every stream from a formatted label, not an integer.

        It does that so ``PYTHONHASHSEED`` cannot reach the draws. An earlier
        version of the fit assumed an integer and raised ``TypeError`` the first
        time the probe asked for a calibrated repair.
        """
        from mcq_audit import fit_target_weights
        items = [{"options": self.OPTIONS, "rank": 0} for _ in range(6)]
        weights, trace = fit_target_weights(items, len(self.OPTIONS),
                                            "20260920|calibrate", steps=1, cap=6)
        self.assertEqual(sorted(weights), sorted(SURFACE_ORDERINGS))
        for row in weights.values():
            self.assertAlmostEqual(sum(row), 1.0, places=12)
        self.assertEqual(len(trace), 1)

    def test_the_tilt_compounds_rather_than_restarting(self):
        """It is a step, not a replacement: a second pass refines the first."""
        from collections import Counter

        from mcq_audit import _tilt
        first = _tilt(None, Counter({0: 70, 1: 10, 2: 10, 3: 10}), 4)
        second = _tilt(first, Counter({0: 40, 1: 20, 2: 20, 3: 20}), 4)
        self.assertLess(second[0], first[0])


class ConstructionTests(unittest.TestCase):
    """The repair that assigns its ranks instead of searching for them."""

    OPTIONS = ["5.0", "4.37", "6.21", "3.88", "7.02", "2.95", "8.14", "9.33", "1.76", "10.5"]
    SMALL = ["5.0", "4.37", "6.21", "3.88"]

    def _prepared(self, options, seed=3, value_target=1):
        from mcq_audit import _menus_for
        rng = random.Random(seed)
        for _ in range(8):
            prepared = _menus_for(options, rng, value_target)
            if prepared is not None:
                return rng, prepared
        self.skipTest("no seeded redraw placed the key at the value rank")

    def test_an_assignment_meets_both_counts_exactly_or_reports_none(self):
        from mcq_audit import _assign_ranks, _reachable_rank_pairs
        _, (_, _, menus, _) = self._prepared(self.SMALL)
        k = len(self.SMALL)
        reachable = _reachable_rank_pairs(menus, k - 1)
        for length in range(k):
            for digits in range(k):
                choices = _assign_ranks(menus, length, digits)
                if (length, digits) in reachable:
                    self.assertIsNotNone(choices, (length, digits))
                    self.assertEqual(sum(c["shorter"] for c in choices), length)
                    self.assertEqual(sum(c["rounder"] for c in choices), digits)
                else:
                    self.assertIsNone(choices, (length, digits))

    def test_the_frontier_is_smaller_than_the_grid_which_is_the_whole_point(self):
        """An independently drawn pair of targets often names a pair no
        rendering of any values reaches; that is why the searches missed."""
        from mcq_audit import _reachable_rank_pairs
        _, (_, _, menus, _) = self._prepared(self.OPTIONS)
        k = len(self.OPTIONS)
        reachable = _reachable_rank_pairs(menus, k - 1)
        self.assertGreater(len(reachable), 0)
        self.assertLess(len(reachable), k * k)

    def _check_invariant(self, options, budget=60):
        """Every isolation move leaves both assigned counts exactly where they were."""
        from mcq_audit import (_assign_ranks, _isolation_moves, _reachable_rank_pairs,
                               digit_scores, feature_rank, length_scores)
        rng, (_, _, menus, _) = self._prepared(options)
        k = len(options)
        pairs = sorted(_reachable_rank_pairs(menus, k - 1))
        self.assertTrue(pairs, "nothing reachable; the test is vacuous")
        pair = pairs[len(pairs) // 2]
        choices = _assign_ranks(menus, *pair)
        self.assertIsNotNone(choices)
        trial = [options[0]] + [c["text"] for c in choices]
        seen = 0
        for candidate in _isolation_moves(trial, menus, k, rng, partners=3):
            seen += 1
            self.assertEqual(feature_rank(candidate, length_scores(candidate), 0),
                             pair[0], candidate)
            scores = digit_scores(candidate)
            if scores is not None:
                self.assertEqual(feature_rank(candidate, scores, 0), pair[1], candidate)
            if seen > budget:
                break
        self.assertGreater(seen, 0, "the neighbourhood is empty")

    def test_isolation_moves_leave_both_assigned_counts_untouched(self):
        """The invariant that lets the third coordinate move at all."""
        self._check_invariant(self.SMALL)

    def test_the_invariant_holds_at_ten_options_too(self):
        """Where it matters: at four options the exchange rarely has to fire."""
        self._check_invariant(self.OPTIONS, budget=120)

    def test_the_value_coordinate_gets_the_same_treatment(self):
        """The lesson applied to the coordinate the paper started with.

        Drawing the value target uniformly and then walking a random
        permutation of the other ranks until one works does not give a uniform
        result; it gives the ranks the item can reach, weighted by nothing.
        """
        from mcq_audit import fit_pair_weights, placeable_value_ranks
        rng = random.Random(11)
        placeable = placeable_value_ranks(self.SMALL, rng, len(self.SMALL))
        self.assertTrue(placeable <= set(range(len(self.SMALL))))
        self.assertTrue(placeable, "no value rank is placeable; the test is vacuous")
        weights, info = fit_pair_weights([self.SMALL] * 40, len(self.SMALL), 13)
        self.assertIn("sorted value", weights)
        self.assertAlmostEqual(sum(weights["sorted value"]), len(self.SMALL), places=6)
        self.assertIn("sorted value", info["widest_attainable_rank_share"])

    def test_construction_meets_the_targets_it_returns(self):
        from mcq_audit import construct_row, ordering_rank
        rng = random.Random(9)
        built = 0
        for _ in range(12):
            result = construct_row(self.SMALL, rng)
            if result is None:
                continue
            built += 1
            options, value_rank, achieved, targets = result
            self.assertEqual(options[0], self.SMALL[0])
            self.assertEqual(ordering_rank(options, "sorted value"), value_rank)
            for name in ("written length", "significant digits"):
                self.assertEqual(achieved[name], targets[name],
                                 f"{name} was assigned, not searched for")
        self.assertGreater(built, 0, "nothing was constructible; the test is vacuous")


class ClusterFieldTests(unittest.TestCase):
    """A missing cluster field is the quietest way to narrow every interval."""

    ROWS = [{"ideal": "1", "distractors": ["2", "3", "4"], "subject": "algebra"},
            {"ideal": "5", "distractors": ["6", "7", "8"], "subject": "algebra"}]

    def test_the_field_is_used_when_it_is_there(self):
        from mcq_audit import build_items
        items, _ = build_items(self.ROWS, "ideal", "distractors", "subject",
                               "question", 4)
        self.assertEqual({item["cluster"] for item in items}, {"algebra"})

    def test_a_missing_field_is_reported_rather_than_absorbed(self):
        """Every item its own cluster is a real answer, and a loud one.

        The bootstrap resamples clusters, so falling back silently turns two
        correlated items into two independent ones and the interval comes out
        too narrow. We shipped a run like that.
        """
        import contextlib
        import io

        from mcq_audit import build_items
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            items, _ = build_items(self.ROWS, "ideal", "distractors", "capsule_uuid",
                                   "question", 4)
        self.assertEqual(len({item["cluster"] for item in items}), 2)
        self.assertIn("no row carries 'capsule_uuid'", stderr.getvalue())
        self.assertIn("too narrow", stderr.getvalue())

    def test_no_warning_when_no_cluster_field_was_asked_for(self):
        import contextlib
        import io

        from mcq_audit import build_items
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            build_items(self.ROWS, "ideal", "distractors", None, "question", 4)
        self.assertEqual(stderr.getvalue(), "")


class ReversedTieBreakTests(unittest.TestCase):
    """Written length under the other tie-break: a sixth coordinate, and the
    one the surface rule family actually reads."""

    OPTIONS = ["12.5", "7.25", "19.0", "3.4"]

    def test_the_two_tie_breaks_are_different_orderings(self):
        from mcq_audit import LENGTH_LATER, ordering_rank
        # All four render four characters wide, so the tie-break decides alone:
        # the audit's rank counts distractors whose text sorts earlier, the
        # reversed one counts those whose text sorts later.
        options = ["12.5", "8.68", "20.0", "25.0"]
        self.assertEqual(ordering_rank(options, "written length"), 0)
        self.assertEqual(ordering_rank(options, LENGTH_LATER), 3)

    def test_it_is_assignable_and_the_construction_meets_it(self):
        """Every reachable rank tuple, met exactly, including the new one.

        The regression this pins is not the dynamic program but the stage after
        it: tuning the key's isolation rank re-renders a distractor inside its
        class, which written length and roundness cannot see and the reversed
        tie-break can. With the class left at (shorter, rounder) this met the
        fourth target on 86% of a benchmark's items.
        """
        import random

        from mcq_audit import (LENGTH_LATER, SURFACE_ORDERINGS, _menus_for,
                               construct_row, legal_pairs)
        prepared = _menus_for(self.OPTIONS, random.Random(3), 1)
        self.assertIsNotNone(prepared)
        _, values, menus, same_sign = prepared
        legal = sorted(legal_pairs(self.OPTIONS, values, menus, same_sign, 1, 4,
                                   with_tie=True))
        self.assertTrue(legal, "no reachable triple; the test is vacuous")
        for length, rounder, later in legal:
            built = construct_row(
                self.OPTIONS, random.Random(3), value_target=1,
                targets={"written length": length, "significant digits": rounder,
                         LENGTH_LATER: later},
                orderings=SURFACE_ORDERINGS + (LENGTH_LATER,))
            self.assertIsNotNone(built, f"could not build a legal triple {legal}")
            achieved = built[2]
            self.assertEqual((achieved["written length"],
                              achieved["significant digits"],
                              achieved[LENGTH_LATER]), (length, rounder, later))

    def test_the_flag_needs_the_construction(self):
        import subprocess
        import sys
        out = subprocess.run(
            [sys.executable, "mcq_audit.py", "--search", "nearest",
             "--with-tie-break", "--repair", "/dev/null"],
            capture_output=True, text=True)
        self.assertNotEqual(out.returncode, 0)
        self.assertIn("--with-tie-break needs --search construct", out.stderr)


class LexicographicCoordinateTests(unittest.TestCase):
    """The fifth coordinate: assignable because the key's rank under it is a count."""

    def test_it_is_not_one_of_the_audited_orderings(self):
        from mcq_audit import LEXICOGRAPHIC, ORDERINGS, SURFACE_ORDERINGS
        self.assertNotIn(LEXICOGRAPHIC, ORDERINGS)
        self.assertNotIn(LEXICOGRAPHIC, SURFACE_ORDERINGS)

    def test_ordering_rank_reaches_it_anyway(self):
        from mcq_audit import LEXICOGRAPHIC, ordering_rank
        # "100" < "12.7" < "3" < "9.5" as text; the key is options[0]
        self.assertEqual(ordering_rank(["9.5", "12.7", "100", "3"], LEXICOGRAPHIC), 3)
        self.assertEqual(ordering_rank(["100", "12.7", "9.5", "3"], LEXICOGRAPHIC), 0)

    def test_every_rendering_is_tagged_with_its_side_of_the_key(self):
        import random
        from mcq_audit import _menus_for
        prepared = _menus_for(["12.5", "7.25", "19.0", "3.4"], random.Random(7), 1)
        self.assertIsNotNone(prepared)
        for menu in prepared[2]:
            for entry in menu:
                self.assertEqual(entry["before"], str(entry["text"]) < "12.5")

    def test_the_construction_meets_a_lexicographic_target_it_can_reach(self):
        import random
        from mcq_audit import (LEXICOGRAPHIC, SURFACE_ORDERINGS, _menus_for,
                               construct_row, legal_pairs, ordering_rank)
        options = ["12.5", "7.25", "19.0", "3.4"]
        prepared = _menus_for(options, random.Random(3), 1)
        triples = sorted(legal_pairs(options, prepared[1], prepared[2], prepared[3],
                                     1, 4, with_lexicographic=True))
        self.assertTrue(triples)
        for target in triples:
            built = construct_row(
                options, random.Random(3), value_target=1,
                targets={"written length": target[0], "significant digits": target[1],
                         LEXICOGRAPHIC: target[2]},
                orderings=SURFACE_ORDERINGS + (LEXICOGRAPHIC,))
            self.assertIsNotNone(built, target)
            self.assertEqual(ordering_rank(built[0], LEXICOGRAPHIC), target[2], target)

    def test_asking_for_it_does_not_disturb_the_two_coordinate_assignment(self):
        import random
        from mcq_audit import _assign_ranks, _menus_for
        prepared = _menus_for(["12.5", "7.25", "19.0", "3.4"], random.Random(11), 2)
        menus = prepared[2]
        for shorter in range(4):
            for rounder in range(4):
                two = _assign_ranks(menus, shorter, rounder)
                three = [_assign_ranks(menus, shorter, rounder, before)
                         for before in range(4)]
                if two is None:
                    self.assertTrue(all(x is None for x in three),
                                    (shorter, rounder))
                else:
                    self.assertTrue(any(x is not None for x in three),
                                    (shorter, rounder))


class CalibrationAndSearchResultsTests(unittest.TestCase):
    """Guard the two findings about the tool itself against their results files."""

    def _load(self, name):
        path = Path("results") / name
        if not path.exists():
            self.skipTest(f"{name} not generated")
        return json.loads(path.read_text())

    def test_no_single_audit_test_over_fires_on_files_that_cannot_leak(self):
        for name in ("audit_calibration.json", "audit_calibration_k10.json"):
            doc = self._load(name)
            rates = doc["false_positive_rate"]
            for test, rate in rates.items():
                if test == "any coordinate":
                    continue
                self.assertLessEqual(rate, 0.10, f"{name}: {test} fires on {rate:.1%}")

    def test_the_disjunction_over_fires_which_is_why_the_tool_has_no_verdict(self):
        r"""This measurement is why ``audit`` returns a list and not a headline.

        Six one-sided tests at one level have six chances to fire, so their OR
        beats every component. \S5 quotes the rate; the tool no longer ships the
        line that had it.
        """
        for name in ("audit_calibration.json", "audit_calibration_k10.json"):
            doc = self._load(name)
            rates = doc["false_positive_rate"]
            coordinates = [rate for test, rate in rates.items()
                           if test != "any coordinate" and "rule family" not in test]
            self.assertGreater(rates["any coordinate"], max(coordinates), name)

    def test_the_claimed_coordinates_clear_what_a_clean_file_can_produce(self):
        """The check that cost us a claim.

        BixBench's lexical-isolation bound is +0.8 points and clean four-option
        files reach +1.5 on that coordinate, so isolation is not established and
        the paper claims value, length and roundness. This test is what keeps
        that honest if either number moves.
        """
        clean = self._load("audit_calibration.json")["widest_bound_on_a_clean_file"]
        released = self._load("mcq_audit_multi.json")["before"]["surface_channels"]["orderings"]
        worst = {}
        for label, value in clean.items():
            name = label.split(" (")[0]
            worst[name] = max(worst.get(name, value), value)
        bounds = {name: block["numeric items"]["credit_bound"]["credit_lower_bound"]
                  for name, block in released.items()}
        for name in ("written length", "significant digits"):
            self.assertGreater(bounds[name], worst[name],
                               f"{name} no longer clears what a clean file produces")
        self.assertLess(bounds["lexical isolation"], worst["lexical isolation"],
                        "isolation now clears the clean-file range; the paper claims "
                        "three coordinates and would be understating it")

    def test_the_better_search_leaves_the_worse_file(self):
        doc = self._load("repair_search_ablation.json")
        cells = {(c["search"], c["draw"]): c for c in doc["cells"]}
        shipped, descent = cells[("sample", "uniform")], cells[("nearest", "uniform")]
        self.assertGreater(descent["first_draw_met_mean"]["written length"],
                           2 * shipped["first_draw_met_mean"]["written length"])
        self.assertGreater(descent["bound_worst"]["written length"], 0.05)
        self.assertLess(shipped["bound_worst"]["written length"], 0.05)

    def test_the_mechanism_is_the_hit_rate_varying_with_the_target_rank(self):
        doc = self._load("repair_search_ablation.json")
        cells = {(c["search"], c["draw"]): c for c in doc["cells"]}
        shipped, descent = cells[("sample", "uniform")], cells[("nearest", "uniform")]
        self.assertGreater(descent["hit_rate_spread_mean"]["written length"],
                           2 * shipped["hit_rate_spread_mean"]["written length"])
        self.assertGreater(descent["widest_realised_mean"]["written length"],
                           shipped["widest_realised_mean"]["written length"] + 0.05)


if __name__ == "__main__":
    unittest.main()
