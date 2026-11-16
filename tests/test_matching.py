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


if __name__ == "__main__":
    unittest.main()
