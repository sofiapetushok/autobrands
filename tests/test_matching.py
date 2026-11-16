"""Regression checks for catalog matching safeguards.

All inputs are short test expressions, not an uploaded claims dataset.
"""
import csv
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from matching import Matcher


class MatchingRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with (ROOT / "data" / "vehicle_catalog.csv").open(encoding="utf-8-sig", newline="") as stream:
            cls.matcher = Matcher(list(csv.DictReader(stream)))

    def assert_identity(self, make, model, expected_make, expected_model, status=None):
        result = self.matcher.resolve(make, model)
        self.assertIn(result.match_status, {"EXACT_MATCH", "ALIAS_MATCH", "FUZZY_MATCH"})
        self.assertEqual((result.corrected_brand, result.corrected_model),
                         (expected_make, expected_model))
        if status:
            self.assertEqual(result.match_status, status)

    def assert_not_accepted(self, make, model, status=None):
        result = self.matcher.resolve(make, model)
        self.assertNotIn(result.match_status, {"EXACT_MATCH", "ALIAS_MATCH", "FUZZY_MATCH"})
        if status:
            self.assertEqual(result.match_status, status)

    def test_standard_and_variant_identity(self):
        self.assert_identity("Toyota", "Yaris", "Toyota", "Yaris", "EXACT_MATCH")
        self.assert_identity("Toyota", "Yaris Cross", "Toyota", "Yaris Cross", "EXACT_MATCH")

    def test_swapped_fields(self):
        self.assert_identity("Yaris", "Toyota", "Toyota", "Yaris", "EXACT_MATCH")
        self.assert_identity("Corolla", "Toyota", "Toyota", "Corolla", "EXACT_MATCH")

    def test_both_identifiers_in_either_field(self):
        self.assert_identity("Toyota Yaris", "", "Toyota", "Yaris")
        self.assert_identity("", "Toyota Yaris", "Toyota", "Yaris")
        self.assert_identity("ToyotaYaris", "", "Toyota", "Yaris")

    def test_misspelled_brand_with_exact_model(self):
        self.assert_identity("SUSUKI", "Swift", "Suzuki", "Swift", "FUZZY_MATCH")
        self.assert_identity("Peug", "3008", "Peugeot", "3008", "FUZZY_MATCH")
        self.assert_identity("Nissan", "Leeaf", "Nissan", "Leaf", "FUZZY_MATCH")

    def test_exact_alias_preserves_reported_model(self):
        for brand, model, canonical in (("Audi", " A3 Cabriolet ", "A3"),
                                         ("Audi", "A3 Sportback", "A3"),
                                         ("Audi", "A4 Avant", "A4"),
                                         ("Aixam", "500.4", "500")):
            with self.subTest(brand=brand, model=model):
                result = self.matcher.resolve(brand, model)
                self.assertEqual((result.match_status, result.corrected_model, result.extra_info),
                                 ("ALIAS_MATCH", canonical, model))
        rows = [
            {"brand": "Aster", "model": "Meteor", "vehicle_type": "car", "aliases": "Meteorr|Shared", "source": "test"},
            {"brand": "Aster", "model": "Comet", "vehicle_type": "car", "aliases": "Shared", "source": "test"},
        ]
        matcher = Matcher(rows)
        for brand, model in (("Aster", "Meteorr"), ("Meteorr", "Aster")):
            with self.subTest(brand=brand, model=model):
                decision = matcher.resolve(brand, model)
                self.assertEqual((decision.match_status, decision.corrected_model, decision.extra_info),
                                 ("ALIAS_MATCH", "Meteor", "Meteorr"))
        self.assertEqual(matcher.resolve("Aster", "Shared").match_status, "AMBIGUOUS")

    def test_distinctive_single_edit_and_competitors(self):
        rows = [{"brand": "Aster", "model": "Comet", "vehicle_type": "car", "aliases": "", "source": "test"}]
        matcher = Matcher(rows)
        for model in ("Comat", "Comot", "Commet", "Comt", "Comte"):
            with self.subTest(model=model):
                decision = matcher.resolve("Aster", model)
                self.assertEqual((decision.match_status, decision.match_method),
                                 ("FUZZY_MATCH", "RANKED_FUZZY_REVIEW_POLICY"))
                self.assertEqual(matcher.identities[decision.top_candidates[0].identity_id].model, "Comet")
        competing = Matcher(rows + [{"brand": "Aster", "model": "Comat", "vehicle_type": "car", "aliases": "", "source": "test"}])
        self.assertEqual(competing.resolve("Aster", "Comot").match_status, "AMBIGUOUS")
        self.assertEqual(self.matcher.resolve("Yamaha", "MT-08").match_status, "AMBIGUOUS")
        self.assert_identity("Toyota", "Corollla", "Toyota", "Corolla", "FUZZY_MATCH")
        self.assert_identity("Suzuki", "Swifft", "Suzuki", "Swift", "FUZZY_MATCH")
        self.assert_identity("Toyota", "Yariss", "Toyota", "Yaris", "FUZZY_MATCH")
        self.assert_identity("Toyota", "Yars", "Toyota", "Yaris", "FUZZY_MATCH")

    def test_contextual_candidates_and_unrelated_input(self):
        decision = self.matcher.resolve("Toyota", "Yaris Cros")
        self.assertEqual((decision.match_status, decision.corrected_model), ("AMBIGUOUS", ""))
        self.assertEqual(self.matcher.identities[decision.top_candidates[0].identity_id].model, "Yaris Cross")
        self.assertEqual(decision, self.matcher.resolve("Toyota", "Yaris Cros"))
        self.assertEqual(self.matcher.resolve("UnknownCo", "UnknownX").top_candidates, ())

    def test_protect_numeric_variants(self):
        self.assert_identity("Yamaha", "MT-07", "Yamaha", "MT-07")
        self.assert_identity("Yamaha", "MT-09", "Yamaha", "MT-09")

    def test_not_invent_missing_variant(self):
        self.assert_not_accepted("Mercedes-Benz", "SPRRINTER", "AMBIGUOUS")
        self.assert_not_accepted("Tesla", "Mode", "AMBIGUOUS")
        self.assert_not_accepted("Volkswagen", "GLF", "AMBIGUOUS")

    def test_explicit_conflict(self):
        self.assert_not_accepted("Toyota Corolla", "Honda", "CONFLICT")

    def test_insufficient_context(self):
        self.assert_not_accepted("", "650")
        self.assert_not_accepted("Toyota", "Corolla Crosss")

    def test_meaningful_punctuation_is_not_discarded(self):
        for brand, model in (("Toyota", "Yaris+"), ("Yamaha", "MT-07+"),
                             ("Toyota", "Yaris/Yaris Cross"), ("", "ToyotaYaris+")):
            with self.subTest(brand=brand, model=model):
                self.assert_not_accepted(brand, model)

    def test_single_trailing_brand_slash_requires_exact_identity(self):
        for brand, model, canonical in (("Yamaha/", "MT-09", "Yamaha"),
                                         ("Dacia/", "Sandero", "Dacia")):
            with self.subTest(brand=brand, model=model):
                self.assert_identity(brand, model, canonical, model, "EXACT_MATCH")
        for brand, model in (("Yamaha//", "MT-09"), ("Yamaha/ ", "MT-09"), ("Yama/ha", "MT-09"),
                             ("Yamaha/", "MT-09+"), ("Toyota", "Yaris/"),
                             ("Toyota", "Yaris/Yaris Cross"), ("Toyota", "Yaris+"),
                             ("Yamaha/", "MT-099"), ("Yamaha/", "Civic"),
                             ("Yamah/", "MT-09")):
            with self.subTest(brand=brand, model=model):
                self.assert_not_accepted(brand, model)
        matcher = Matcher([
            {"brand": "Aster", "model": "Comet", "vehicle_type": "car", "aliases": "Meteor", "source": "test"},
            {"brand": "Aster", "model": "350", "vehicle_type": "unknown", "aliases": "", "source": "test"},
            {"brand": "Aster", "model": "Star", "vehicle_type": "car", "aliases": "Shared", "source": "test"},
            {"brand": "Aster", "model": "Moon", "vehicle_type": "car", "aliases": "Shared", "source": "test"},
        ])
        alias = matcher.resolve("Aster/", "Meteor")
        self.assertEqual((alias.match_status, alias.corrected_model), ("ALIAS_MATCH", "Comet"))
        for model in ("350", "Shared"):
            with self.subTest(model=model):
                self.assertNotIn(matcher.resolve("Aster/", model).match_status,
                                 {"EXACT_MATCH", "ALIAS_MATCH", "FUZZY_MATCH"})

    def test_generic_numeric_across_field_layouts(self):
        matcher = Matcher([{"brand": "Aster", "model": "350", "vehicle_type": "unknown", "aliases": "", "source": "test"}])
        for brand, model in (("Aster", "350"), ("350", "Aster"), ("Aster 350", ""), ("", "Aster350")):
            with self.subTest(brand=brand, model=model):
                result = matcher.resolve(brand, model)
                self.assertEqual(result.match_status, "AMBIGUOUS")
                self.assertEqual((result.corrected_brand, result.corrected_model), ("", ""))

    def test_catalog_supported_punctuation_remains_identifying(self):
        matcher = Matcher([
            {"brand": "Aster", "model": "Comet+", "vehicle_type": "car", "aliases": "Comet/Plus", "source": "test"},
            {"brand": "Boreal", "model": "350", "vehicle_type": "motorcycle", "aliases": "", "source": "test"},
        ])
        for brand, model, expected in (("Aster", "Comet+", "Comet+"),
                                       ("Aster", "Comet/Plus", "Comet+"),
                                       ("Comet+", "Aster", "Comet+"),
                                       ("Boreal", "350", "350"),
                                       ("350", "Boreal", "350")):
            with self.subTest(brand=brand, model=model):
                result = matcher.resolve(brand, model)
                self.assertIn(result.match_status, {"EXACT_MATCH", "ALIAS_MATCH", "FUZZY_MATCH"})
                self.assertEqual(result.corrected_model, expected)
        self.assert_not_accepted("Toyota", "Yaris/", "AMBIGUOUS")

    def test_current_partial_and_ambiguity_outcomes(self):
        cases = [
            ("Toyota", "", "PARTIAL_MATCH", "Toyota", ""),
            ("", "Yaris", "PARTIAL_MATCH", "", "Yaris"),
            ("", "650", "AMBIGUOUS", "", ""),
            ("Toyota", "ΓΙΑΡΙΣ", "AMBIGUOUS", "", ""),
            ("ΤΟΓΙΟΤΑ", "ΓΙΑΡΙΣ", "AMBIGUOUS", "", ""),
            ("Nisssan", "Leeaf", "AMBIGUOUS", "", ""),
            ("Yamaha", "MT", "PARTIAL_MATCH", "Yamaha", ""),
            ("Yamaha", "MT-08", "AMBIGUOUS", "", ""),
            ("Toyota", "Yaris Cros", "AMBIGUOUS", "", ""),
            ("Toyota", "Yaris GT", "AMBIGUOUS", "", ""),
            ("Toyota", "Yaris 2020", "AMBIGUOUS", "", ""),
            ("Toyota", "Yaris 1.5", "AMBIGUOUS", "", ""),
            ("Yamaha", "MT-07 700 CC", "AMBIGUOUS", "", ""),
            ("Toyota Yaris", "Honda Civic", "CONFLICT", "", ""),
            ("Toyota", "Civic", "CONFLICT", "", ""),
            ("Toyota", "TotallyUnknown999", "PARTIAL_MATCH", "Toyota", ""),
            ("UnknownCo", "UnknownX", "AMBIGUOUS", "", ""),
            ("Mercedes-Benz", "Sprinter", "AMBIGUOUS", "", ""),
        ]
        for brand, model, status, expected_brand, expected_model in cases:
            with self.subTest(brand=brand, model=model):
                result = self.matcher.resolve(brand, model)
                self.assertEqual((result.match_status, result.corrected_brand, result.corrected_model),
                                 (status, expected_brand, expected_model))

    def test_additional_supported_paths(self):
        for brand, model, expected_brand, expected_model in (
            ("TOYΟTA", "Yaris", "Toyota", "Yaris"),
            ("", "ToyotaYaris", "Toyota", "Yaris"),
            ("Yamaha", "MT-07 07 CC", "Yamaha", "MT-07"),
        ):
            with self.subTest(brand=brand, model=model):
                self.assert_identity(brand, model, expected_brand, expected_model)
        fuel = self.matcher.resolve("Toyota", "Yaris DIESEL")
        self.assertEqual((fuel.match_status, fuel.corrected_model, fuel.extra_info),
                         ("FUZZY_MATCH", "Yaris", "DIESEL"))


if __name__ == "__main__":
    unittest.main()
