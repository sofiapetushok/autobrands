import csv
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "data" / "vehicle_catalog.csv"
sys.path.insert(0, str(ROOT.parent))

from autobrands import normalize
from autobrands.matching import Matcher, weighted_similarity


class NormalizationTests(unittest.TestCase):
    def test_minimal_unicode_normalization_preserves_identity_information(self):
        cases = [
            ("  υαμαηα ", "ΥΑΜΑΗΑ"),
            (" βμW ", "ΒΜW"),
            (" ΟΡΕL ", "ΟΡΕL"),
            (" Golf + ", "GOLF +"),
            ("MT‐07", "MT-07"),
            ("e\u0301", "É"),
        ]
        for source, expected in cases:
            with self.subTest(source=source):
                self.assertEqual(normalize.normalize_text(source), expected)
        self.assertEqual(normalize.comparison_forms("ΒΜW"), ("ΒΜW", "BMW", "VMW"))
        self.assertIn("YAMAHA", normalize.comparison_forms("ΥΑΜΑΗΑ"))
        self.assertIn("OPEL", normalize.comparison_forms("ΟΡΕL"))
        self.assertIn("PENO", normalize.comparison_forms("ΡΕΝΟ"))
        self.assertIn("RENO", normalize.comparison_forms("ΡΕΝΟ"))
        self.assertEqual(normalize.normalize_text(pd.NA), None)
        self.assertEqual(weighted_similarity("Citroen", "CITROËN"), 100.0)
        self.assertLess(weighted_similarity("T o y o t a", "TOYOTA"), 88.0)


class MatcherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.matcher = Matcher.from_csv(CATALOG, top_k=5)

    def test_unicode_brand_alternatives_and_alias(self):
        cases = [
            ("ΒΜW", "BMW"),
            ("ΥΑΜΑΗΑ", "YAMAHA"),
            ("ΟΡΕL", "OPEL"),
            ("ΟΠΕΛ", "OPEL"),
            ("ΤΟΥΟΤΑ", "TOYOTA"),
            ("VW", "VOLKSWAGEN"),
        ]
        for value, expected in cases:
            with self.subTest(value=value):
                result = self.matcher.resolve(value, None)
                self.assertEqual(result["corrected_brand"], expected)
                self.assertEqual(result["match_status"], "MATCHED")

    def test_accent_equivalence_and_spacing_are_distinguished(self):
        for value in ("Citroen", "CITROE\u0308N"):
            with self.subTest(value=value):
                result = self.matcher.resolve(value, None)
                self.assertEqual(result["corrected_brand"], "CITROËN")
                self.assertEqual(result["match_score"], 100.0)
        spaced_letters = self.matcher.resolve("T o y o t a", None)
        self.assertEqual(spaced_letters["match_status"], "UNRESOLVED")
        self.assertEqual(spaced_letters["corrected_brand"], "T O Y O T A")

    def test_joint_brand_model_evidence_and_collisions(self):
        citroen = self.matcher.resolve("Citroen", "C3")
        self.assertEqual(citroen["match_status"], "MATCHED")
        self.assertEqual(citroen["corrected_brand"], "CITROËN")
        self.assertEqual(citroen["corrected_model"], "C3")
        aixam = [item for item in citroen["candidates"] if item["brand"] == "AIXAM"]
        self.assertFalse(aixam)

        yamaha = self.matcher.resolve("YAMHA", "MT07")
        self.assertEqual((yamaha["corrected_brand"], yamaha["corrected_model"]),
                         ("YAMAHA", "MT-07"))
        self.assertEqual(yamaha["match_status"], "FUZZY_MATCH")

        toyota = self.matcher.resolve("TOYTA", "YARSI")
        self.assertEqual(toyota["match_status"], "FUZZY_MATCH")
        self.assertEqual(toyota["corrected_brand"], "TOYOTA")
        self.assertEqual(toyota["corrected_model"], "YARIS")

        c3_only = self.matcher.resolve(None, "C3")
        self.assertEqual(c3_only["match_status"], "AMBIGUOUS")
        self.assertEqual(c3_only["match_score"], 100.0)

    def test_unordered_fields_and_combined_identity(self):
        cases = [
            self.matcher.resolve("TOYOTA", "YARIS"),
            self.matcher.resolve("YARIS", "TOYOTA"),
            self.matcher.resolve("TOYOTA YARIS", None),
            self.matcher.resolve(None, "TOYOTA YARIS"),
        ]
        self.assertEqual({(item["corrected_brand"], item["corrected_model"],
                           item["match_status"]) for item in cases},
                         {("TOYOTA", "YARIS", "EXACT_MATCH")})
        for column in ("brand", "model"):
            frame = pd.DataFrame({column: ["SUZUKI VITAR"]})
            result = self.matcher.match_dataframe(
                frame, brand_col=column if column == "brand" else None,
                model_col=column if column == "model" else None)
            self.assertEqual(result.loc[0, "match_status"], "AMBIGUOUS")

    def test_full_phrase_retrieval_and_alias_aware_guard(self):
        combined = self.matcher.resolve(None, "SUZUKI VITAR")
        self.assertEqual((combined["corrected_brand"], combined["corrected_model"]),
                         ("SUZUKI", "VITARA"))
        self.assertEqual(combined["match_status"], "FUZZY_MATCH")
        combined_in_brand = self.matcher.resolve("SUZUKI VITAR", None)
        self.assertEqual(combined_in_brand["match_status"], combined["match_status"])
        self.assertEqual(combined_in_brand["candidates"], combined["candidates"])

        separated = self.matcher.resolve("SUZUKI", "VITAR")
        self.assertEqual(separated["match_status"], "FUZZY_MATCH")
        self.assertEqual(separated["corrected_model"], "VITARA")

        damaged_alias = self.matcher.resolve(None, "Σκαραβα$ος")
        self.assertEqual((damaged_alias["corrected_brand"], damaged_alias["corrected_model"]),
                         ("VOLKSWAGEN", "BEETLE"))
        self.assertEqual(damaged_alias["match_score"], 90.0)

    def test_unordered_fields_and_combined_identity(self):
        cases = [
            self.matcher.resolve("TOYOTA", "YARIS"),
            self.matcher.resolve("YARIS", "TOYOTA"),
            self.matcher.resolve("TOYOTA YARIS", None),
            self.matcher.resolve(None, "TOYOTA YARIS"),
        ]
        self.assertEqual({(item["corrected_brand"], item["corrected_model"],
                           item["match_status"]) for item in cases},
                         {("TOYOTA", "YARIS", "EXACT_MATCH")})
        for column in ("brand", "model"):
            result = self.matcher.match_dataframe(
                pd.DataFrame({column: ["SUZUKI VITAR"]}),
                brand_col=column if column == "brand" else None,
                model_col=column if column == "model" else None,
            )
            self.assertEqual(result.loc[0, "match_status"], "FUZZY_MATCH")

    def test_catalog_model_aliases_and_numeric_context(self):
        cases = [
            (None, "Σκαραβαίος", "VOLKSWAGEN", "BEETLE"),
            (None, "MT-07", "YAMAHA", "MT-07"),
            (None, "MT07", "YAMAHA", "MT-07"),
            (None, "CBR 600 RR", "HONDA", "CBR600RR"),
            (None, "PCX125", "HONDA", "PCX125"),
            ("Volkswagen", "Σκαθάρι 1500", "VOLKSWAGEN", "BEETLE 1500"),
        ]
        for brand, model, expected_brand, expected_model in cases:
            with self.subTest(model=model):
                result = self.matcher.resolve(brand, model)
                self.assertEqual(result["corrected_brand"], expected_brand)
                self.assertEqual(result["corrected_model"], expected_model)
                self.assertIn(result["match_status"],
                              {"EXACT_MATCH", "ALIAS_MATCH", "MATCHED", "FUZZY_MATCH"})

    def test_uncertain_models_are_not_collapsed_or_forced(self):
        for brand, model in (("Toyota", "Yaris+"), ("Yamaha", "MT-09+")):
            with self.subTest(brand=brand, model=model):
                result = self.matcher.resolve(brand, model)
                self.assertEqual(result["corrected_model"], normalize.latin_fallback(model))
                self.assertNotIn(result["match_status"], {"EXACT_MATCH", "ALIAS_MATCH", "FUZZY_MATCH"})
                self.assertTrue(all(candidate["model"] != "GOLF"
                                    or candidate["score"] < 100
                                    for candidate in result["candidates"]))

    def test_trailing_plus_accepts_unique_catalog_base(self):
        matcher = Matcher([
            {"brand": "VOLKSWAGEN", "model": "GOLF", "vehicle_type": "car"},
        ])
        result = matcher.resolve(None, "Golf +")
        self.assertEqual((result["corrected_brand"], result["corrected_model"],
                          result["match_status"]), ("VOLKSWAGEN", "GOLF", "FUZZY_MATCH"))
        self.assertEqual(result["match_score"], weighted_similarity("Golf +", "GOLF"))

    def test_trailing_plus_stays_ambiguous_for_competing_identities(self):
        catalogs = (
            [
                {"brand": "VOLKSWAGEN", "model": "GOLF", "vehicle_type": "car"},
                {"brand": "VOLKSWAGEN", "model": "GOLF PLUS", "vehicle_type": "car"},
            ],
            [
                {"brand": "VOLKSWAGEN", "model": "GOLF", "vehicle_type": "car"},
                {"brand": "FORD", "model": "GOLF", "vehicle_type": "car"},
            ],
        )
        for rows in catalogs:
            with self.subTest(rows=rows):
                result = Matcher(rows).resolve(None, "Golf +")
                self.assertEqual(result["match_status"], "AMBIGUOUS")
                self.assertEqual(result["corrected_model"], "GOLF +")

        alias_matcher = Matcher([
            {"brand": "VOLKSWAGEN", "model": "GOLF", "vehicle_type": "car"},
            {"brand": "VOLKSWAGEN", "model": "GOLF PLUS", "model_aliases": "GOLF+",
             "vehicle_type": "car"},
        ])
        alias_result = alias_matcher.resolve(None, "Golf +")
        self.assertEqual((alias_result["corrected_model"], alias_result["match_status"]),
                         ("GOLF PLUS", "ALIAS_MATCH"))

        identified = Matcher(catalogs[1]).resolve("VOLKSWAGEN", "Golf +")
        self.assertEqual((identified["corrected_brand"], identified["corrected_model"],
                          identified["match_status"]),
                         ("VOLKSWAGEN", "GOLF", "FUZZY_MATCH"))

    def test_numeric_model_variants_are_not_collapsed(self):
        beetle = self.matcher.resolve("Volkswagen", "Beetle 1300")
        if any(identity.brand_key == "VOLKSWAGEN" and identity.model_key == "BEETLE 1300"
               for identity in self.matcher.identities):
            self.assertEqual(beetle["corrected_model"], "BEETLE 1300")
            self.assertIn(beetle["match_status"], {"EXACT_MATCH", "ALIAS_MATCH", "MATCHED"})
        else:
            self.assertNotEqual(beetle["corrected_model"], "BEETLE")
        wrong_number = self.matcher.resolve("Toyota", "Yaris 1300")
        self.assertNotEqual(wrong_number["corrected_model"], "YARIS")

    def test_collision_unknown_conflict_and_empty_inputs(self):
        rows = [
            {"brand": "Alpha", "model": "One", "vehicle_type": "car", "model_aliases": "Shared"},
            {"brand": "Beta", "model": "Two", "vehicle_type": "car", "model_aliases": "Shared"},
        ]
        matcher = Matcher(rows)
        ambiguous = matcher.resolve(None, "Shared")
        self.assertEqual(ambiguous["match_status"], "AMBIGUOUS")
        self.assertEqual(len(ambiguous["candidates"]), 2)
        self.assertEqual([item["brand"] for item in ambiguous["candidates"]], ["ALPHA", "BETA"])
        self.assertEqual(ambiguous["candidates"], matcher.resolve(None, "Shared")["candidates"])
        self.assertEqual(matcher.resolve("Unknown Maker", "Unknown Model")["match_status"], "UNRESOLVED")
        self.assertEqual(matcher.resolve(None, None)["match_status"], "UNRESOLVED")
        self.assertEqual(self.matcher.resolve("Toyota", "Civic")["match_status"], "CONFLICT")

    def test_confusable_alternative_collisions_remain_ambiguous(self):
        brand_matcher = Matcher([
            {"brand": "BETA", "model": "One", "vehicle_type": "car"},
            {"brand": "VETA", "model": "Two", "vehicle_type": "car"},
        ])
        brand_result = brand_matcher.resolve("ΒETA", None)
        self.assertEqual(brand_result["match_status"], "AMBIGUOUS")
        self.assertEqual([item["score"] for item in brand_result["candidates"][:2]],
                         [100.0, 100.0])

        model_matcher = Matcher([
            {"brand": "One", "model": "BETA", "vehicle_type": "car"},
            {"brand": "Two", "model": "VETA", "vehicle_type": "car"},
        ])
        model_result = model_matcher.resolve(None, "ΒETA")
        self.assertEqual(model_result["match_status"], "AMBIGUOUS")
        self.assertEqual([item["score"] for item in model_result["candidates"][:2]],
                         [100.0, 100.0])

    def test_dataframe_contract_duplicate_indices_and_repeated_rows(self):
        source = pd.DataFrame({
            "Κατασκευαστής": ["ΒΜW", None, "VW", "???", "VW"] ,
            "Vehicle name": ["R1250GS", "MT-07", "Golf +", "", "Golf +"],
            "note": ["a", "b", "c", "d", "e"],
        }, index=[4, 4, 7, 8, 8])
        snapshot = source.copy(deep=True)
        result = self.matcher.match_dataframe(
            source, brand_col="Κατασκευαστής", model_col="Vehicle name", top_k=3
        )
        self.assertEqual(len(result), len(source))
        self.assertTrue(result.index.equals(source.index))
        pd.testing.assert_frame_equal(source, snapshot)
        self.assertEqual(result["note"].tolist(), source["note"].tolist())
        self.assertTrue(all(len(items) <= 3 for items in result["candidates"]))
        self.assertEqual(result.loc[4].iloc[0]["corrected_brand"], "BMW")
        self.assertEqual(result.loc[4].iloc[1]["corrected_model"], "MT-07")
        self.assertEqual(result.loc[8, "corrected_brand"].tolist(), ["", "VOLKSWAGEN"])
        self.assertEqual(result.loc[8, "corrected_model"].tolist(), ["", "GOLF"])
        self.assertEqual(result.loc[8, "match_status"].tolist(), ["UNRESOLVED", "FUZZY_MATCH"])

    def test_brand_only_model_only_and_empty_dataframe_schemas(self):
        brand = self.matcher.match_dataframe(pd.DataFrame({"make": ["VW", "nope"]}), brand_col="make")
        self.assertIn("corrected_brand", brand)
        self.assertNotIn("corrected_model", brand)
        model = self.matcher.match_dataframe(pd.DataFrame({"name": ["PCX125"]}), model_col="name")
        self.assertNotIn("corrected_brand", model)
        self.assertEqual(model.loc[0, "corrected_model"], "PCX125")
        empty = self.matcher.match_dataframe(pd.DataFrame(columns=["make", "name"]),
                                             brand_col="make", model_col="name")
        self.assertTrue(empty.empty)
        self.assertEqual(list(empty.columns), ["make", "name", "corrected_brand",
                                               "corrected_model", "match_status",
                                               "match_score", "candidates"])

    def test_debug_explains_comparison_basis_without_changing_default_schema(self):
        source = pd.DataFrame({
            "brand": ["ΒΜW", "Citroen", "Toyota", "VW", "Yamaha", "Yamha"],
            "model": [None, "C3", "Yaris", "Golf", "MT07", "MT07"],
        })
        default = self.matcher.match_dataframe(source, brand_col="brand", model_col="model")
        debug = self.matcher.match_dataframe(source, brand_col="brand", model_col="model",
                                             debug=True)
        self.assertEqual(list(default.columns), ["brand", "model", "corrected_brand",
                                                 "corrected_model", "match_status",
                                                 "match_score", "candidates"])
        self.assertIn("ab_match_method", debug)
        self.assertEqual(debug.loc[0, "ab_match_method"], "UNICODE_EQUIVALENCE")
        self.assertEqual(debug.loc[1, "ab_match_method"], "UNICODE_EQUIVALENCE")
        self.assertEqual(debug.loc[2, "ab_match_method"], "EXACT_CANONICAL")
        self.assertEqual(debug.loc[3, "ab_match_method"], "EXPLICIT_ALIAS")
        self.assertEqual(debug.loc[4, "ab_match_method"], "FORMATTING_EQUIVALENCE")
        self.assertEqual(debug.loc[5, "ab_match_method"], "ORDINARY_FUZZY")

    def test_repeated_dataframe_uses_bounded_indexes_and_cache(self):
        frame = pd.DataFrame({"brand": ["ΒΜW"] * 100, "model": ["R1250GS"] * 100})
        before = self.matcher.stats["weighted_comparisons"]
        result = self.matcher.match_dataframe(frame, brand_col="brand", model_col="model")
        delta = self.matcher.stats["weighted_comparisons"] - before
        self.assertEqual(len(result), 100)
        self.assertLess(delta, 100 * 10)
        self.assertGreater(self.matcher.stats["score_cache_hits"], 0)
        self.assertGreater(self.matcher.stats["candidate_ids_retrieved"], 0)


if __name__ == "__main__":
    unittest.main()
