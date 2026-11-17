"""Regression and workflow tests for the unified matcher."""

import importlib.util
import json
import sys
import csv
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("autobrands_run", HERE / "run.py")
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)
FULL_MATCHES = {"EXACT_MATCH", "ALIAS_MATCH", "FUZZY_MATCH"}


class MatchingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        catalog = module.read_csv(HERE / "data" / "vehicle_catalog.csv",
                                  {"brand", "model", "vehicle_type", "aliases"})
        cls.matcher = module.Matcher(catalog)

    def resolve(self, brand, model):
        return self.matcher.resolve(brand, model)

    def test_exact_distinct_car_families(self):
        yaris = self.resolve("TOYOTA", "YARIS")
        cross = self.resolve("TOYOTA", "YARIS CROSS")
        civic = self.resolve("HONDA", "CIVIC")
        self.assertEqual((yaris.match_status, yaris.corrected_model), ("EXACT_MATCH", "Yaris"))
        self.assertEqual((cross.match_status, cross.corrected_model),
                         ("EXACT_MATCH", "Yaris Cross"))
        self.assertEqual((civic.corrected_brand, civic.corrected_model, civic.vehicle_type),
                         ("Honda", "Civic", "car"))

    def test_explicit_brand_and_model_aliases(self):
        model_alias = self.resolve("AUDI", "A3 SPORTBACK")
        greek = self.resolve("ΤΟΥΟΤΑ", "YARIS")
        self.assertEqual((model_alias.match_status, model_alias.corrected_model),
                         ("ALIAS_MATCH", "A3"))
        self.assertEqual((greek.corrected_brand, greek.corrected_model),
                         ("Toyota", "Yaris"))
        folded = self.resolve("CITROEN", "C3")
        self.assertEqual((folded.match_status, folded.corrected_brand,
                          folded.match_method),
                         ("EXACT_MATCH", "Citroën", "FOLDED_BRAND_EXACT"))
        combined = self.resolve("NULL", "CITROEN C3")
        self.assertEqual((combined.match_status, combined.corrected_brand),
                         ("EXACT_MATCH", "Citroën"))

    def test_cross_field_and_embedded_names(self):
        for brand, model in (("NULL", "TOYOTA YARIS"), ("YARIS", "TOYOTA"),
                             ("TOYOTA YARIS", "NULL"), ("TOYOTA", "TOYOTA YARIS")):
            with self.subTest(brand=brand, model=model):
                decision = self.resolve(brand, model)
                self.assertEqual((decision.match_status, decision.corrected_brand,
                                  decision.corrected_model),
                                 ("EXACT_MATCH", "Toyota", "Yaris"))
        repeated_make = self.resolve("VESPA", "ΒΕΣΠΑ PRIMAVERA 125")
        self.assertEqual((repeated_make.match_status, repeated_make.corrected_model),
                         ("EXACT_MATCH", "Primavera 125"))
        short_catalog_name = self.resolve("ΧΟΝΔΑ", "PCX 125")
        self.assertEqual((short_catalog_name.corrected_brand,
                          short_catalog_name.corrected_model), ("Honda", "PCX125"))

    def test_missing_and_partial(self):
        partial = self.resolve("TOYOTA", "NULL")
        self.assertEqual((partial.match_status, partial.corrected_brand,
                          partial.corrected_model), ("PARTIAL_MATCH", "Toyota", ""))
        self.assertEqual(self.resolve("NULL", "-000XXX").match_status, "UNRESOLVED")
        self.assertEqual(self.resolve("NULL", "125").match_status, "AMBIGUOUS")
        self.assertNotEqual(self.resolve("HONDA", "125").match_status, "EXACT_MATCH")
        self.assertEqual(self.resolve("PEUGEOT", "208").corrected_model, "208")
        self.assertEqual(self.resolve("NULL", "HONDA PCX 125").match_status,
                         "EXACT_MATCH")

    def test_conflict_keeps_corrections_empty(self):
        decision = self.resolve("TOYOTA", "CIVIC")
        self.assertEqual(decision.match_status, "CONFLICT")
        self.assertEqual((decision.corrected_brand, decision.corrected_model), ("", ""))
        self.assertNotEqual(self.resolve("VOLKSWAGEN", "GLF").match_status, "CONFLICT")
        self.assertEqual((self.resolve("SYM SYMPHONY", "ST 200").match_status,
                          self.resolve("SYM SYMPHONY", "ST 200").corrected_model),
                         ("EXACT_MATCH", "Symphony ST 200"))

    def test_fuzzy_suggestions_and_safe_acceptance(self):
        misspelled_brand = self.resolve("YAMHA", "MT07")
        self.assertEqual(misspelled_brand.match_status, "AMBIGUOUS")
        self.assertEqual(self.matcher.identities[misspelled_brand.top_candidates[0].identity_id].model,
                         "MT-07")
        cross = self.resolve("NULL", "PARIS CROSS")
        self.assertEqual(cross.match_status, "AMBIGUOUS")
        self.assertEqual(self.matcher.identities[cross.top_candidates[0].identity_id].model,
                         "Yaris Cross")
        accepted = self.resolve("HONDA", "PCX 125")
        self.assertEqual((accepted.match_status, accepted.corrected_model),
                         ("FUZZY_MATCH", "PCX125"))
        self.assertEqual(self.resolve("YAMAHA", "MT09").corrected_model, "MT-09")
        self.assertEqual(self.resolve("PEUGOT", "208").corrected_brand, "Peugeot")
        self.assertEqual(self.resolve("TOYOTA", "CROLLA CROSS").corrected_model,
                         "Corolla Cross")
        self.assertEqual(self.resolve("TOYTA", "YARSI").corrected_model, "")

    def test_same_make_different_types_and_similar_families(self):
        self.assertEqual(self.resolve("HONDA", "CIVIC").vehicle_type, "car")
        self.assertEqual(self.resolve("HONDA", "PCX125").corrected_model, "PCX125")
        self.assertEqual(self.resolve("BMW", "3 SERIES").vehicle_type, "car")
        self.assertEqual(self.resolve("BMW", "R1250GS").corrected_model, "R1250GS")
        self.assertEqual(self.resolve("YAMAHA", "MT-07").corrected_model, "MT-07")
        self.assertEqual(self.resolve("YAMAHA", "MT-09").corrected_model, "MT-09")
        self.assertEqual(self.resolve("YAMAHA", "MT-0").match_status, "AMBIGUOUS")

    def test_deterministic_ranking_and_json(self):
        first = self.resolve("NULL", "PARIS CROSS")
        second = self.resolve("NULL", "PARIS CROSS")
        self.assertEqual(first, second)
        serialized = self.matcher.serialize_candidates(first)
        self.assertEqual(json.loads(serialized)[0]["model"], "Yaris Cross")
        self.assertEqual(serialized, self.matcher.serialize_candidates(second))

    def test_fuzzy_guardrails(self):
        for brand, model in (("M", "COUNTRYMAN"),
                             ("MERCEDES-BENZ", "SPRINTER"),
                             ("MERCEDES-BENZ A", "180"),
                             ("HONDA", "AFRICA TWIN 1100")):
            with self.subTest(brand=brand, model=model):
                decision = self.resolve(brand, model)
                self.assertNotEqual(decision.match_status, "FUZZY_MATCH")
                self.assertEqual(decision.corrected_model, "")

    def test_file_preservation_and_repeatability(self):
        source_row = {
            "brand_raw": " Toyota ", "model_raw": "Yaris",
            "brand_normalized": "TOYOTA", "model_normalized": "YARIS",
            "brand_missing": "false", "model_missing": "false", "quality_flags": "",
        }
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "claims.csv"
            target = Path(directory) / "matched.csv"
            with source.open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=source_row)
                writer.writeheader()
                writer.writerow(source_row)
            args = (source, HERE / "data" / "vehicle_catalog.csv", None,
                    target, 94, 8, 88, 88)
            report = module.run(*args)
            first = target.read_bytes()
            with target.open(encoding="utf-8", newline="") as file:
                reader = csv.DictReader(file)
                self.assertEqual(tuple(reader.fieldnames), module.OUTPUT_COLUMNS)
                actual = next(reader)
            self.assertEqual(report["rows"], 1)
            self.assertTrue(all(actual[key] == value for key, value in source_row.items()))
            self.assertEqual(actual["match_status"], "EXACT_MATCH")
            self.assertEqual(json.loads(actual["top_candidates"])[0]["model"], "Yaris")
            module.run(*args)
            self.assertEqual(target.read_bytes(), first)

    def test_held_out_catalog_generalization(self):
        # These invented identities are absent from the development catalog
        # and claims. The engine must construct every candidate from this input.
        catalog = [
            {"brand": brand, "model": model, "vehicle_type": "motorcycle",
             "aliases": ""}
            for brand, model in (
                ("Aster", "Swift 125"), ("Aster", "Swift 400"),
                ("Aster", "Swift 125 GT"), ("Boreal", "Comet 125"),
                ("Volkan", "Meteor 700"),
            )
        ]
        matcher = module.Matcher(catalog)
        cases = (
            ("Aster", "Swift125", "FUZZY_MATCH", "Aster", "Swift 125"),
            ("ASTR", "Swift 400", "FUZZY_MATCH", "Aster", "Swift 400"),
            ("NULL", "BorealComet125", "EXACT_MATCH", "Boreal", "Comet 125"),
            ("ΒΟΛΚΑΝ", "Meteor700", "FUZZY_MATCH", "Volkan", "Meteor 700"),
            ("NULL", "ΒΟΛΚΑΝ Meteor700", "EXACT_MATCH", "Volkan", "Meteor 700"),
            ("NULL", "Swift 400", "PARTIAL_MATCH", "", "Swift 400"),
            ("Aster", "Comet125", "CONFLICT", "", ""),
            ("Aster", "Swift 12", "AMBIGUOUS", "", ""),
            ("NULL", "ASTR", "UNRESOLVED", "", ""),
        )
        for brand, model, status, corrected_brand, corrected_model in cases:
            with self.subTest(brand=brand, model=model):
                result = matcher.resolve(brand, model)
                self.assertEqual((result.match_status, result.corrected_brand,
                                  result.corrected_model),
                                 (status, corrected_brand, corrected_model))
                self.assertEqual(result, matcher.resolve(brand, model))

    def test_brand_aliases_require_explicit_input(self):
        row = {"brand": "Aster", "model": "Swift 125", "vehicle_type": "motorcycle",
               "aliases": ""}
        self.assertNotEqual(module.Matcher([row]).resolve("ASTER MOTORS", "SWIFT 125").match_status,
                            "ALIAS_MATCH")
        verified = module.Matcher([row], [{"brand": "Aster", "alias": "ASTER MOTORS"}])
        self.assertEqual(verified.resolve("ASTER MOTORS", "SWIFT 125").match_status,
                         "ALIAS_MATCH")

    def test_generic_short_designation_is_not_a_conflict(self):
        rows = [
            {"brand": brand, "model": model, "vehicle_type": "car",
             "aliases": ""}
            for brand, model in (("Aster", "Tourer"), ("Boreal", "C 200"))
        ]
        decision = module.Matcher(rows).resolve("Aster", "C 200")
        self.assertNotEqual(decision.match_status, "CONFLICT")
        self.assertEqual(decision.corrected_model, "")

    def test_required_schema_on_configurable_input(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "other_claims.csv"
            source.write_text("brand_normalized\nASTER\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing columns"):
                module.read_csv(source, set(module.OUTPUT_COLUMNS[:6]) | {"quality_flags"})


class LatinComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = [
            {"brand": brand, "model": model, "vehicle_type": "motorcycle",
             "aliases": ""}
            for brand, model in (("Aster", "Meteor700"), ("Boreal", "Orbit150"))
        ]
        cls.matcher = module.Matcher(cls.catalog)

    def test_greek_comparison_channel_and_safeguards(self):
        row = module.normalize.normalize_row(
            {"brand": "ΑΣΤΕΡ", "model": "ΜΕΤΕΟΡ700"})
        supplied = {name: row[name] for name in ("brand_latin", "model_latin")}
        result = self.matcher.resolve(row["brand_normalized"], row["model_normalized"], **supplied)
        self.assertEqual((result.match_status, result.corrected_brand, result.corrected_model),
                         ("FUZZY_MATCH", "Aster", "Meteor700"))
        self.assertEqual(result, self.matcher.resolve(row["brand_normalized"], row["model_normalized"]))
        for model in ("ΜΕΤΕΟΡ701", "ΜΕΤΕΟΡ700 GT"):
            self.assertNotIn(self.matcher.resolve("ΑΣΤΕΡ", model).match_status, FULL_MATCHES)
        self.assertEqual(self.matcher.resolve("ΑΣΤΕΡ", "ΟΡΜΠΙΤ150").match_status, "CONFLICT")
        self.assertEqual(self.matcher.resolve("", "ΜΕΤΕΟΡ700").match_status, "PARTIAL_MATCH")
        with self.assertRaisesRegex(ValueError, "Latin comparison"):
            self.matcher.resolve("ΑΣΤΕΡ", "ΜΕΤΕΟΡ701", model_latin="METEOR700")
        greek_catalog = module.Matcher([{**self.catalog[0], "model": "ΜΠΟΛΤ700"}])
        self.assertEqual(greek_catalog.resolve("Aster", "BOLT700").corrected_model, "ΜΠΟΛΤ700")

    def test_old_and_new_csvs_preserve_fields_and_match_identically(self):
        row = module.normalize.normalize_row(
            {"brand": "ΑΣΤΕΡ", "model": "ΜΕΤΕΟΡ700"})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog = root / "catalog.csv"
            with catalog.open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=self.catalog[0])
                writer.writeheader(); writer.writerows(self.catalog)
            outputs = []
            for include_latin in (False, True):
                source, target = root / "input.csv", root / "output.csv"
                fields = row if include_latin else {k: v for k, v in row.items()
                                                   if k not in {"brand_latin", "model_latin"}}
                with source.open("w", encoding="utf-8", newline="") as file:
                    writer = csv.DictWriter(file, fieldnames=fields)
                    writer.writeheader(); writer.writerow(fields)
                report = module.run(source, catalog, None, target, 94, 8, 88, 88)
                self.assertEqual(report["rows"], 1)
                with target.open(encoding="utf-8", newline="") as file:
                    actual = next(csv.DictReader(file))
                self.assertTrue(all(actual[k] == v for k, v in row.items()))
                outputs.append(target.read_bytes())
            self.assertEqual(*outputs)


class AdvancedV2Tests(unittest.TestCase):
    """Independent identities exercise new rules without claim-derived labels."""

    @classmethod
    def setUpClass(cls):
        rows = [
            {"brand": brand, "model": model, "vehicle_type": "car",
             "aliases": ""}
            for brand, model in (
                ("Aster", "Nova"), ("Aster", "Nova Cross"),
                ("Aster", "Nova Sport"), ("Boreal", "Runner 125"),
                ("Boreal", "Runner 150"), ("Volkan", "Meteor 700"),
                ("Volkswagen", "Golf"), ("Volkswagen", "Golf Variant"),
            )
        ]
        cls.matcher = module.Matcher(rows)

    def resolve(self, brand, model):
        return self.matcher.resolve(brand, model)

    def top(self, decision):
        return self.matcher.identities[decision.top_candidates[0].identity_id].model

    def test_a_context_noise_requires_catalog_identity(self):
        result = self.resolve("NULL", "ASTER NOVA ΑΤΥΧΗΜΑ")
        self.assertEqual((result.match_status, result.corrected_brand,
                          result.corrected_model, self.top(result)),
                         ("AMBIGUOUS", "", "", "Nova"))
        uncertain = self.resolve("NULL", "ASTER NOVA INCIDENT")
        self.assertEqual((uncertain.match_status, self.top(uncertain)),
                         ("AMBIGUOUS", "Nova"))
        trailing_make = self.resolve("Aster", "Nova Boreal")
        self.assertEqual(trailing_make.match_status, "AMBIGUOUS")

    def test_b_trim_does_not_erase_model_terms(self):
        result = self.resolve("Aster", "Nova 1.6 TDI Comfortline")
        self.assertEqual((result.match_status, result.corrected_model, self.top(result)),
                         ("AMBIGUOUS", "", "Nova"))
        self.assertEqual(self.resolve("Aster", "Nova Cross").corrected_model,
                         "Nova Cross")
        self.assertEqual(self.resolve("Aster", "Nova Cross Sport").match_status,
                         "AMBIGUOUS")
        self.assertEqual(self.resolve("NULL", "ASTER NOVA ΚΡΟΣ").match_status,
                         "AMBIGUOUS")

    def test_c_engine_unit_keeps_displacement(self):
        result = self.resolve("Boreal", "Runner 150CC")
        self.assertEqual((result.match_status, result.corrected_model),
                         ("FUZZY_MATCH", "Runner 150"))
        self.assertEqual(self.resolve("Boreal", "Runner 155CC").corrected_model, "")

    def test_d_numeric_formatting_keeps_model_number(self):
        self.assertEqual(self.resolve("Boreal", "Runner150").corrected_model,
                         "Runner 150")
        self.assertEqual(self.resolve("Boreal", "Runner125").corrected_model,
                         "Runner 125")

    def test_e_repeated_tokens(self):
        result = self.resolve("NULL", "ASTER ASTER NOVA")
        self.assertEqual((result.match_status, result.corrected_model),
                         ("EXACT_MATCH", "Nova"))

    def test_f_reordered_tokens_keep_family(self):
        result = self.resolve("NULL", "CROSS NOVA ASTER")
        self.assertEqual((result.match_status, result.corrected_model),
                         ("EXACT_MATCH", "Nova Cross"))

    def test_g_two_spelling_errors_improve_candidate_not_forced(self):
        result = self.resolve("NULL", "ASTERR NVOA")
        self.assertEqual((result.match_status, result.corrected_model,
                          self.top(result)), ("AMBIGUOUS", "", "Nova"))

    def test_h_phonetic_brand_requires_model_evidence(self):
        result = self.resolve("ΒΟΛΚΑΝ", "METEOR700")
        self.assertEqual((result.corrected_brand, result.corrected_model),
                         ("Volkan", "Meteor 700"))
        self.assertEqual(self.resolve("ΒΟΛΚΑΝ", "NULL").corrected_model, "")

    def test_i_unlisted_family_variant_is_not_invented(self):
        result = self.resolve("Aster", "Nova Cross GT")
        self.assertNotIn(result.match_status,
                         {"EXACT_MATCH", "ALIAS_MATCH", "FUZZY_MATCH"})
        self.assertEqual(result.corrected_model, "")

    def test_j_multiple_vehicle_mentions(self):
        different = self.resolve("NULL", "ASTER NOVA / BOREAL RUNNER 150")
        same_brand = self.resolve("NULL", "ASTER NOVA / ASTER NOVA CROSS")
        self.assertEqual(different.match_status, "CONFLICT")
        self.assertEqual(same_brand.match_status, "AMBIGUOUS")
        self.assertEqual((different.corrected_brand, same_brand.corrected_model), ("", ""))
        split_fields = self.resolve("ASTER NOVA", "BOREAL RUNNER 150")
        self.assertEqual(split_fields.match_status, "CONFLICT")

    def test_k_identity_inside_longer_text(self):
        result = self.resolve("NULL", "ΙΧ ASTER NOVA ΑΣΦΑΛΙΣΜΕΝΟΥ")
        self.assertEqual((result.match_status, result.corrected_model,
                          self.top(result)), ("AMBIGUOUS", "", "Nova"))

    def test_l_combined_corruption_keeps_year_uncertain(self):
        result = self.resolve("NULL", "ΑΣΤΕΡΝΟΒΑ 2020")
        self.assertEqual((result.match_status, result.corrected_model,
                          self.top(result)), ("AMBIGUOUS", "", "Nova"))

    def test_adversarial_families_unknowns_and_garbage(self):
        self.assertEqual(self.resolve("Volkswagen", "Golf Variant").corrected_model,
                         "Golf Variant")
        self.assertEqual(self.resolve("Volkswagen", "Golf Variant Plus").corrected_model,
                         "")
        self.assertEqual(self.resolve("ZXQ", "Nova").corrected_model, "")
        self.assertEqual(self.resolve("NULL", "&*@").match_status, "UNRESOLVED")
        self.assertEqual(self.resolve("NULL", "125").corrected_model, "")

    def test_real_catalog_numeric_families(self):
        catalog = module.read_csv(HERE / "data" / "vehicle_catalog.csv",
                                  {"brand", "model", "vehicle_type", "aliases"})
        real = module.Matcher(catalog)
        self.assertEqual(real.resolve("TOYOTA", "YARIS CROSS").corrected_model,
                         "Yaris Cross")
        self.assertEqual(real.resolve("YAMAHA", "MT-07").corrected_model, "MT-07")
        self.assertEqual(real.resolve("YAMAHA", "MT-09").corrected_model, "MT-09")
        self.assertEqual(real.resolve("KAWASAKI", "NINJA 400").corrected_model,
                         "Ninja 400")
        self.assertEqual(real.resolve("KAWASAKI", "NINJA 650").corrected_model,
                         "Ninja 650")


class FinalRefinementTests(unittest.TestCase):
    """Held-out identities exercise parsing rules without claim-derived labels."""

    @classmethod
    def setUpClass(cls):
        rows = [{"brand": brand, "model": model, "vehicle_type": "car",
                 "aliases": ""}
                for brand, model in (
                    ("Pentar", "Pioneer"), ("Pentar", "Pioneer Cross"),
                    ("Pentar", "Ranger 300"), ("Pentar", "Ranger 350"),
                    ("Pentar", "RX-07"), ("Pentar", "RX-09"),
                    ("Pentar", "CV-X"), ("Pentar", "1999"),
                    ("Pentar", "Q 240"),
                    ("Boreon", "Pioneer"), ("Boreon", "Diesel"),
                )]
        cls.matcher = module.Matcher(rows)

    def test_recognized_descriptions_require_complete_catalog_identity(self):
        cases = {
            "PIONEER ΒΕΝΖΙΝΗ": "CATALOG_MODEL_REPORTED_FUEL",
            "ΙΧ PIONEER": "CATALOG_SUPPLEMENTARY_TEXT",
            "PIONEER PETROL": "CATALOG_MODEL_REPORTED_FUEL",
        }
        for text, method in cases.items():
            with self.subTest(text=text):
                result = self.matcher.resolve("Pentar", text)
                self.assertEqual((result.match_status, result.corrected_model, result.match_method),
                                 ("FUZZY_MATCH", "Pioneer", method))
        result = self.matcher.resolve("NULL", "PENTAR ΙΧ PIONEER")
        self.assertEqual((result.corrected_brand, result.corrected_model), ("Pentar", "Pioneer"))

    def test_unfamiliar_or_catalog_defining_tokens_are_preserved(self):
        for text in ("PIONEER UNKNOWNTRIM", "PIONEER CROSS GT",
                     "PIONEER 1999 ΒΕΝΖΙΝΗ", "PIONEER / RANGER 300 ΒΕΝΖΙΝΗ"):
            with self.subTest(text=text):
                self.assertNotIn(self.matcher.resolve("Pentar", text).match_status, FULL_MATCHES)
        result = self.matcher.resolve("Pentar", "PIONEER CROSS ΒΕΝΖΙΝΗ")
        self.assertEqual(result.corrected_model, "Pioneer Cross")
        diesel = self.matcher.resolve("Pentar", "PIONEER DIESEL")
        self.assertEqual((diesel.match_status, diesel.corrected_model, diesel.extra_info),
                         ("FUZZY_MATCH", "Pioneer", "DIESEL"))
        # Catalog use of a vocabulary term takes precedence even in another make.
        self.assertIn(module.latin_fold("DIESEL"), self.matcher.catalog_tokens)

    def test_displacement_confirmation_preserves_distinct_numbers(self):
        for text in ("RANGER 300CC", "RANGER 300 300CC", "RANGER 300 300 CC"):
            self.assertEqual(self.matcher.resolve("Pentar", text).corrected_model, "Ranger 300")
        for text in ("RANGER 300 350CC", "RANGER 350 300CC", "RANGER 305CC"):
            self.assertNotIn(self.matcher.resolve("Pentar", text).match_status, FULL_MATCHES)
        self.assertEqual(self.matcher.resolve("Pentar", "RANGER 350 350CC").corrected_model,
                         "Ranger 350")

    def test_numeric_designation_order_is_not_discarded(self):
        for text in ("240 Q", "240 Q ΒΕΝΖΙΝΗ"):
            result = self.matcher.resolve("Pentar", text)
            self.assertEqual((result.match_status, result.match_method),
                             ("AMBIGUOUS", "NUMERIC_TOKEN_ORDER"))
        self.assertEqual(self.matcher.resolve("Pentar", "RANGER 300 ΒΕΝΖΙΝΗ").corrected_model,
                         "Ranger 300")
        self.assertEqual(self.matcher.resolve("Pentar", "300 RANGER").corrected_model, "Ranger 300")

    def test_year_like_numbers_cannot_certify_an_identity(self):
        for text in ("PIONEER 1999", "RANGER 350 2002", "PIONEER 2020"):
            result = self.matcher.resolve("Pentar", text)
            self.assertEqual((result.match_status, result.corrected_model), ("AMBIGUOUS", ""))
            self.assertEqual(result.match_method, "POTENTIAL_YEAR_WITHOUT_DATE_EVIDENCE")
        # Four-digit catalog model names are model evidence, never discarded.
        self.assertEqual(self.matcher.resolve("Pentar", "1999").corrected_model, "1999")
        self.assertNotIn(self.matcher.resolve("NULL", "1999").match_status, FULL_MATCHES)

    def test_visual_confusables_require_a_latin_majority_within_each_word(self):
        self.assertEqual(module.latin_fold("ΡIΟNEER"), "PIONEER")
        self.assertEqual(module.latin_fold("ΡENTAR"), "PENTAR")
        self.assertEqual(module.latin_fold("Ρ"), "R")
        self.assertEqual(module.latin_fold("ΡΕΝΤΑΡ"), "RENTAR")
        self.assertEqual(module.latin_fold("AΡ"), "AR")
        self.assertEqual(module.latin_fold("ΡΕNΤΑR"), "RENTAR")
        result = self.matcher.resolve("ΡENTAR", "ΡIΟNEER")
        self.assertEqual((result.corrected_brand, result.corrected_model), ("Pentar", "Pioneer"))

    def test_confusables_do_not_change_numbers_or_family_tokens(self):
        for text in ("ΡIΟNEER CROSS GT", "RX-08", "RANGER 305"):
            self.assertNotIn(self.matcher.resolve("ΡENTAR", text).match_status, FULL_MATCHES)
        result = self.matcher.resolve("ΡENTAR", "ΡIΟNEER CROSS")
        self.assertEqual(result.corrected_model, "Pioneer Cross")

    def test_short_catalog_model_evidence_stays_partial(self):
        partial = self.matcher.resolve("NULL", "CV-X")
        self.assertEqual((partial.match_status, partial.corrected_brand, partial.corrected_model),
                         ("PARTIAL_MATCH", "", "CV-X"))
        shared = self.matcher.resolve("NULL", "PIONEER")
        self.assertEqual((shared.match_status, shared.corrected_brand), ("AMBIGUOUS", ""))
        unique = self.matcher.resolve("NULL", "RANGER 350")
        self.assertEqual((unique.match_status, unique.corrected_brand, unique.corrected_model),
                         ("PARTIAL_MATCH", "", "Ranger 350"))

    def test_conflicts_and_multiple_vehicles_survive_description_handling(self):
        result = self.matcher.resolve("Pentar", "BOREON DIESEL ΒΕΝΖΙΝΗ")
        self.assertEqual(result.match_status, "CONFLICT")
        result = self.matcher.resolve("NULL", "PENTAR PIONEER / BOREON DIESEL ΙΧ")
        self.assertEqual(result.match_status, "CONFLICT")


class ConsolidatedWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog_path = HERE / "data" / "vehicle_catalog.csv"
        rows = module.read_csv(cls.catalog_path,
                               {"brand", "model", "vehicle_type", "aliases"})
        cls.matcher = module.Matcher(rows)

    @staticmethod
    def write_csv(path, columns, rows):
        with path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=columns, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)

    def test_alias_information_and_significant_punctuation(self):
        for model in ("A3 CABRIOLET", "A3 SPORTBACK", "A4 AVANT"):
            with self.subTest(model=model):
                result = self.matcher.resolve("AUDI", model)
                self.assertEqual(result.match_status, "ALIAS_MATCH")
                self.assertEqual(result.extra_info, model)
        for brand, model in (("Yamaha//", "MT-09"), ("Yamaha/", "MT-09+"),
                             ("Toyota", "Yaris/"), ("Toyota", "Yaris+")):
            with self.subTest(brand=brand, model=model):
                self.assertNotIn(self.matcher.resolve(brand, model).match_status, FULL_MATCHES)
        self.assertEqual(self.matcher.resolve("Yamaha/", "MT-09").corrected_model, "MT-09")
        self.assertEqual(self.matcher.resolve("Dacia/", "Sandero").corrected_model, "Sandero")

    def test_greek_localized_aliases_and_latin_transliterations(self):
        shared = (
            ("Volkswagen", "Σκαραβαίος", "SKARAVEOS", 6),
            ("Volkswagen", "Σκαθάρι", "SKATHARI", 6),
            ("Volkswagen", "Κατσαριδάκι", "KATSARIDAKI", 6),
            ("Volkswagen", "Χελώνα", "CHELONA", 6),
            ("Citroen", "Ντεσεβό", "DESEVO", 2),
            ("Citroen", "Ντε Σεβό", "DE SEVO", 2),
            ("Citroen", "Ασχημόπαπο", "ASCHIMOPAPO", 2),
            ("Fiat", "500αράκι", "500ARAKI", 2),
            ("Fiat", "Πεντακοσαράκι", "PEDAKOSARAKI", 2),
            ("Renault", "Κατρέλ", "KATREL", 3),
            ("Renault", "Κατρουλή", "KATRULI", 3),
        )
        for brand, greek, latin, candidate_count in shared:
            with self.subTest(brand=brand, alias=greek):
                normalized = module.normalize.normalize_text(greek)
                self.assertEqual(module.normalize.latin_comparison_text(normalized), latin)
                for alias in (greek, latin):
                    result = self.matcher.resolve(brand, alias)
                    self.assertEqual(result.match_status, "AMBIGUOUS")
                    self.assertEqual(result.candidate_count, candidate_count)

        for alias in ("Βάτραχος", "VATRACHOS"):
            with self.subTest(alias=alias):
                result = self.matcher.resolve("Citroen", alias)
                self.assertIn(result.match_status, FULL_MATCHES)
                self.assertEqual((result.corrected_brand, result.corrected_model),
                                 ("Citroën", "DS"))

    def test_shared_greek_alias_numeric_context_narrows_without_accepting(self):
        cases = (
            ("Volkswagen", "1300 Σκαραβαίος", {"1300 Beetle", "Fusca 1300"}),
            ("Volkswagen", "1300 SKARAVEOS", {"1300 Beetle", "Fusca 1300"}),
            ("Volkswagen", "1200 Σκαραβαίος", {"1200 Beetle"}),
            ("Fiat", "110F Πεντακοσαράκι", {"500-110F"}),
        )
        for brand, model, expected in cases:
            with self.subTest(brand=brand, model=model):
                result = self.matcher.resolve(brand, model)
                candidates = {self.matcher.identities[item.identity_id].model
                              for item in result.top_candidates}
                self.assertEqual((result.match_status, result.match_method),
                                 ("AMBIGUOUS", "SHARED_ALIAS_NUMERIC_CONTEXT"))
                self.assertEqual(candidates, expected)

        conflicting = self.matcher.resolve("Volkswagen", "1700 Σκαραβαίος")
        self.assertEqual((conflicting.match_status, conflicting.match_method,
                          conflicting.candidate_count),
                         ("AMBIGUOUS", "SHARED_ALIAS_NUMERIC_CONFLICT", 6))

    def test_existing_aliases_and_canonical_names_are_unchanged(self):
        cases = (
            ("Volkswagen", "Type 1", "ALIAS_MATCH", "Beetle"),
            ("Volkswagen", "Käfer", "ALIAS_MATCH", "Beetle"),
            ("Citroen", "2CV", "ALIAS_MATCH", "Az 2CV"),
            ("Fiat", "Nuova 500", "ALIAS_MATCH", "500"),
            ("Citroen", "DS", "EXACT_MATCH", "DS"),
            ("Volkswagen", "1200 Beetle", "EXACT_MATCH", "1200 Beetle"),
            ("Volkswagen", "1300 Beetle", "EXACT_MATCH", "1300 Beetle"),
            ("Citroen", "Azu 2CV", "EXACT_MATCH", "Azu 2CV"),
            ("Renault", "4 Tl", "EXACT_MATCH", "4 Tl"),
            ("Renault", "R4 Clan", "EXACT_MATCH", "R4 Clan"),
            ("Fiat", "500-110F", "EXACT_MATCH", "500-110F"),
        )
        for brand, model, status, canonical_model in cases:
            with self.subTest(brand=brand, model=model):
                result = self.matcher.resolve(brand, model)
                self.assertEqual((result.match_status, result.corrected_model),
                                 (status, canonical_model))

    def test_greek_alias_collisions_and_broad_nicknames_abstain(self):
        collision_rows = [
            {"brand": "Aster", "model": model, "vehicle_type": "car",
             "aliases": "Κοινό"}
            for model in ("Nova", "Nova Cross")
        ]
        collision = module.Matcher(collision_rows).resolve("Aster", "Κοινό")
        self.assertEqual((collision.match_status, collision.match_method),
                         ("AMBIGUOUS", "EXACT_NAME_COLLISION"))

        for brand, nickname in (("BMW", "Μπέμπα"),
                                ("Honda", "Παπί"), ("Honda", "Παπάκι"),
                                ("Honda", "Μπόμπος"), ("Honda", "Ζετάκι"),
                                ("Mercedes-Benz", "Καναδέζα"),
                                ("Volkswagen", "Πλοιαρχικό"),
                                ("Citroen", "Θεά"),
                                ("Fuldamobil", "Ασημένιος ψύλλος")):
            with self.subTest(brand=brand, nickname=nickname):
                self.assertNotIn(self.matcher.resolve(brand, nickname).match_status,
                                 FULL_MATCHES)
        brandless = self.matcher.resolve("", "Σκαραβαίος")
        self.assertEqual((brandless.match_status, brandless.corrected_brand,
                          brandless.corrected_model), ("AMBIGUOUS", "", ""))
        self.assertGreater(brandless.candidate_count, 1)

    def test_public_statuses(self):
        self.assertEqual(module.public_status(self.matcher.resolve("Toyota", "Yaris")), "MATCHED")
        self.assertEqual(module.public_status(self.matcher.resolve("Toyota", "Civic")), "REVIEW")
        self.assertEqual(module.public_status(self.matcher.resolve("", "&*@")), "UNRESOLVED")

    def test_verified_mapping_add_replace_revoke_and_no_match(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog = root / "catalog.csv"
            verified = root / "verified.csv"
            review = root / "review.csv"
            catalog_rows = [
                {"brand": "Aster", "model": "Nova", "vehicle_type": "car",
                 "aliases": ""},
                {"brand": "Aster", "model": "Nova Cross", "vehicle_type": "car",
                 "aliases": ""},
            ]
            self.write_csv(catalog, tuple(catalog_rows[0]), catalog_rows)

            def import_row(action, canonical_brand="", canonical_model=""):
                row = {"brand_input": "ASTER", "model_input": "NVOA",
                       "action": action, "canonical_brand": canonical_brand,
                       "canonical_model": canonical_model}
                self.write_csv(review, module.VERIFIED_COLUMNS, [row])
                return module.import_reviews(review, catalog, verified)

            self.assertEqual(import_row("MATCH", "Aster", "Nova"), 1)
            first = verified.read_bytes()
            self.assertEqual(import_row("MATCH", "Aster", "Nova"), 0)
            self.assertEqual(verified.read_bytes(), first)

            matcher = module.Matcher(catalog_rows)
            matcher.set_verified_mappings(module.read_csv(verified, set(module.VERIFIED_COLUMNS)))
            result = matcher.resolve("ASTER", "NVOA")
            self.assertEqual((result.corrected_model, result.match_method),
                             ("Nova", "HUMAN_VERIFIED"))

            with self.assertRaisesRegex(ValueError, "Conflicting prior verification"):
                import_row("MATCH", "Aster", "Nova Cross")
            self.assertEqual(verified.read_bytes(), first)

            self.assertEqual(import_row("REPLACE", "Aster", "Nova Cross"), 1)
            replaced = module.read_csv(verified, set(module.VERIFIED_COLUMNS))
            self.assertEqual((replaced[0]["action"], replaced[0]["canonical_model"]),
                             ("MATCH", "Nova Cross"))
            self.assertEqual(import_row("REVOKE"), 1)
            self.assertEqual(module.read_csv(verified, set(module.VERIFIED_COLUMNS)), [])

            self.assertEqual(import_row("NO_MATCH"), 1)
            matcher.set_verified_mappings(module.read_csv(verified, set(module.VERIFIED_COLUMNS)))
            no_match = matcher.resolve("ASTER", "NVOA")
            self.assertEqual((no_match.match_status, no_match.match_method),
                             ("UNRESOLVED", "HUMAN_NO_MATCH"))

    def test_pipeline_preserves_order_duplicates_and_review_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw.csv"
            normalized = root / "normalized.csv"
            matched = root / "matched.csv"
            review = root / "review.csv"
            rows = [
                {"brand": "Toyota", "model": "Yaris"},
                {"brand": "Toyota", "model": "Civic"},
                {"brand": "Toyota", "model": "Yaris"},
            ]
            self.write_csv(raw, ("brand", "model"), rows)
            module.normalize.normalize_file(raw, normalized)
            module.run(normalized, self.catalog_path, None, matched, 94, 8, 88, 88,
                       None, review)
            with matched.open(encoding="utf-8", newline="") as file:
                output = list(csv.DictReader(file))
            self.assertEqual([(row["brand_raw"], row["model_raw"]) for row in output],
                             [(row["brand"], row["model"]) for row in rows])
            self.assertEqual(output[0], output[2])
            self.assertEqual([row["result_status"] for row in output],
                             ["MATCHED", "REVIEW", "MATCHED"])
            with review.open(encoding="utf-8", newline="") as file:
                review_rows = list(csv.DictReader(file))
            self.assertEqual(len(review_rows), 1)
            self.assertTrue(review_rows[0]["top_candidates"])

if __name__ == "__main__":
    unittest.main()
