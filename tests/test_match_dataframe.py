"""Regression tests for the normalize-to-matcher DataFrame contract."""

from pathlib import Path
import unittest

import pandas as pd

import normalize
from matching import Matcher


ROOT = Path(__file__).resolve().parents[1]


class MatchDataFrameTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.matcher = Matcher.from_csv(ROOT / "data" / "vehicle_catalog.csv")

    def run_pipeline(self, frame, *, top_k=None, debug=False):
        original = frame.copy(deep=True)
        working = original.copy(deep=True)
        if "brand" not in working:
            working["brand"] = pd.NA
        if "model" not in working:
            working["model"] = pd.NA
        normalized = normalize.normalize_dataframe(
            working, brand_col="brand", model_col="model", prefix="ab_"
        )
        for field in ("brand", "model"):
            if field not in original and field in normalized:
                normalized = normalized.drop(columns=field)
        result = self.matcher.match_dataframe(
            normalized, prefix="ab_", top_k=top_k, debug=debug
        )
        pd.testing.assert_frame_equal(frame, original)
        self.assertTrue(result.index.equals(frame.index))
        return result

    def test_brand_only_exact_alias_fuzzy_and_unresolved(self):
        frame = pd.DataFrame({"brand": [
            "Toyota", "VW", "Mercedes-Ben", "ΤΟΥΟΤΑ", "Unknown Maker",
        ]})
        result = self.run_pipeline(frame)

        self.assertEqual(list(result.columns), [
            "brand", "corrected_brand", "match_status",
            "match_score", "candidates",
        ])
        self.assertEqual(result["corrected_brand"].tolist(), [
            "TOYOTA", "VOLKSWAGEN", "MERCEDES-BENZ", "TOYOTA", "UNKNOWN MAKER",
        ])
        self.assertEqual(result.loc[4, "match_status"], "UNRESOLVED")
        self.assertEqual(result.loc[2, "match_status"], "MATCHED")
        self.assertTrue(all(set(item) == {"brand", "score"}
                            for candidates in result["candidates"] for item in candidates))

    def test_model_only_and_full_pair_preserve_independent_values(self):
        model_only = self.run_pipeline(pd.DataFrame({"model": ["Yaris"]}))
        self.assertEqual(model_only.loc[0, "corrected_model"], "YARIS")
        self.assertEqual(model_only.loc[0, "match_status"], "MATCHED")
        self.assertNotIn("corrected_brand", model_only)
        self.assertEqual(set(model_only.loc[0, "candidates"][0]), {"brand", "model", "score"})

        full = self.run_pipeline(pd.DataFrame({"brand": ["Toyota"], "model": ["Yaris"]}))
        self.assertEqual(full.loc[0, "corrected_brand"], "TOYOTA")
        self.assertEqual(full.loc[0, "corrected_model"], "YARIS")
        self.assertEqual(full.loc[0, "match_status"], "EXACT_MATCH")
        self.assertEqual(list(full.columns), [
            "brand", "model", "corrected_brand", "corrected_model",
            "match_status", "match_score", "candidates",
        ])

    def test_model_only_partial_match_does_not_drop_unverified_plus(self):
        result = self.run_pipeline(pd.DataFrame({"model": ["Golf+", "Golf +"]}))
        self.assertEqual(result["corrected_model"].tolist(), ["GOLF+", "GOLF +"])

    def test_ambiguous_input_retains_normalized_values_not_candidate_identity(self):
        result = self.run_pipeline(pd.DataFrame({
            "brand": ["Volkswagen"], "model": ["Σκαθάρι 1700"],
        }))
        self.assertEqual(result.loc[0, "corrected_brand"], "VOLKSWAGEN")
        self.assertEqual(result.loc[0, "corrected_model"], "SKATHARI 1700")
        self.assertEqual(result.loc[0, "match_status"], "AMBIGUOUS")

    def test_candidates_are_stable_ranked_and_top_k_only_limits_output(self):
        frame = pd.DataFrame({"brand": ["Volkswagen"], "model": ["Skaravaios"]})
        top_five = self.run_pipeline(frame, top_k=5)
        repeated = self.run_pipeline(frame, top_k=5)
        top_two = self.run_pipeline(frame, top_k=2)

        candidates = top_five.loc[0, "candidates"]
        scores = [item["score"] for item in candidates]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertEqual(candidates, repeated.loc[0, "candidates"])
        self.assertLessEqual(len(top_two.loc[0, "candidates"]), 2)
        self.assertEqual(top_two.loc[0, "match_status"], top_five.loc[0, "match_status"])


    def test_weighted_unicode_matching_across_input_schemas(self):
        brand_only = self.run_pipeline(pd.DataFrame({"brand": [
            "ΒΜW", "ΥΑΜΑΗΑ", "ΟΡΕL", "ΟΠΕΛ", "ΤΟΥΟΤΑ", "Unknown Maker",
        ]}))
        self.assertEqual(brand_only["corrected_brand"].tolist(), [
            "BMW", "YAMAHA", "OPEL", "OPEL", "TOYOTA", "UNKNOWN MAKER",
        ])
        self.assertEqual(brand_only["match_status"].tolist(), [
            "MATCHED", "MATCHED", "MATCHED", "MATCHED", "MATCHED", "UNRESOLVED",
        ])

        model_only = self.run_pipeline(pd.DataFrame({"model": [
            "Σκαραβαίος", "Σκαραβα$ος", "MT-07", "PCX125", "Golf +",
            "Unlisted Vehicle Model",
        ]}))
        self.assertEqual(model_only.loc[0, "corrected_model"], "BEETLE")
        self.assertEqual(model_only.loc[1, "corrected_model"], "BEETLE")
        self.assertEqual(model_only.loc[2, "corrected_model"], "MT-07")
        self.assertEqual(model_only.loc[3, "corrected_model"], "PCX125")
        self.assertEqual(model_only.loc[4, "match_status"], "PARTIAL_MATCH")
        self.assertNotIn(model_only.loc[5, "match_status"],
                         {"MATCHED", "EXACT_MATCH", "ALIAS_MATCH", "FUZZY_MATCH"})

        full = self.run_pipeline(pd.DataFrame({
            "brand": ["MERCEDES-ΒΕΝ", "Volkswagen"],
            "model": ["C-Class", "Σκαθάρι 1500"],
        }))
        self.assertEqual((full.loc[0, "corrected_brand"], full.loc[0, "corrected_model"]),
                         ("MERCEDES-BENZ", "C-CLASS"))
        self.assertEqual(full.loc[0, "match_status"], "FUZZY_MATCH")
        self.assertEqual((full.loc[1, "corrected_brand"], full.loc[1, "corrected_model"]),
                         ("VOLKSWAGEN", "BEETLE 1500"))

    def test_unicode_interpretation_collision_remains_ambiguous(self):
        collision_matcher = Matcher([
            {"brand": "Aster", "model": "BETA X", "vehicle_type": "car", "aliases": ""},
            {"brand": "Aster", "model": "VETA X", "vehicle_type": "car", "aliases": ""},
        ])
        decision = collision_matcher.resolve("Aster", "ΒΕΤΑ X")
        self.assertEqual(decision.match_status, "AMBIGUOUS")
        self.assertEqual(decision.corrected_model, "")
        self.assertEqual(
            {collision_matcher.identities[item.identity_id].model
             for item in decision.top_candidates},
            {"BETA X", "VETA X"},
        )

    def test_unicode_weighted_match_does_not_weaken_numeric_or_unknown_safeguards(self):
        self.assertEqual(self.matcher.resolve("Yamaha", "MT-0").match_status, "AMBIGUOUS")
        partial = self.matcher.resolve("Toyota", "ΜΗ ΓΝΩΣΤΟ")
        self.assertEqual((partial.match_status, partial.corrected_brand, partial.corrected_model),
                         ("PARTIAL_MATCH", "TOYOTA", ""))
        unknown = self.matcher.resolve("Unknown Maker", "Unknown Model")
        self.assertNotIn(unknown.match_status,
                         {"MATCHED", "EXACT_MATCH", "ALIAS_MATCH", "FUZZY_MATCH"})

    def test_debug_columns_are_opt_in(self):
        frame = pd.DataFrame({"brand": ["Toyota"], "model": ["Yaris"]})
        compact = self.run_pipeline(frame)
        debug = self.run_pipeline(frame, debug=True)
        self.assertEqual(len(compact.columns), 7)
        self.assertIn("ab_brand_normalized", debug)
        self.assertIn("ab_model_normalized", debug)
        self.assertIn("ab_match_method", debug)

    def test_confusable_comparison_alternatives_use_catalog_safely(self):
        brands = self.run_pipeline(pd.DataFrame({"brand": [
            "ΒΜW", "ΥΑΜΑΗΑ", "ΟΡΕL", "ΟΠΕΛ",
        ]}))
        self.assertEqual(brands.loc[1, "corrected_brand"], "YAMAHA")
        self.assertEqual(brands.loc[1, "match_status"], "MATCHED")
        self.assertEqual(brands.loc[2, "corrected_brand"], "OPEL")
        self.assertEqual(brands.loc[2, "match_status"], "MATCHED")
        self.assertEqual(brands.loc[3, "corrected_brand"], "OPEL")
        self.assertEqual(brands.loc[3, "match_status"], "MATCHED")
        # In this mixed-script token, the bounded visual reading is stronger
        # than the unrelated phonetic reading that previously selected VMW.
        self.assertEqual(brands.loc[0, "corrected_brand"], "BMW")
        self.assertEqual(brands.loc[0, "match_status"], "MATCHED")
        self.assertEqual(brands.loc[0, "candidates"][0]["brand"], "BMW")

        models = self.run_pipeline(pd.DataFrame({"model": [
            "ΣΚΑΘΑΡΙ", "Σκαραβαίος", "Golf +", "MT-07", "PCX125",
        ]}))
        self.assertEqual(models["corrected_model"].tolist(), [
            "BEETLE", "BEETLE", "GOLF +", "MT-07", "PCX125",
        ])
        self.assertEqual(models["match_status"].tolist(), [
            "MATCHED", "MATCHED", "PARTIAL_MATCH", "MATCHED", "MATCHED",
        ])

        mixed_brand = self.run_pipeline(pd.DataFrame({
            "brand": ["MERCEDES-ΒΕΝ"], "model": ["C-Class"],
        }))
        self.assertEqual(mixed_brand.loc[0, "corrected_brand"], "MERCEDES-BENZ")
        self.assertEqual(mixed_brand.loc[0, "corrected_model"], "C-CLASS")
        self.assertEqual(mixed_brand.loc[0, "match_status"], "FUZZY_MATCH")


if __name__ == "__main__":
    unittest.main()
