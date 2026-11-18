"""Regression tests for the standalone normalizer."""

import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path


SPEC = importlib.util.spec_from_file_location(
    "autobrands_normalize", Path(__file__).resolve().parent.parent / "normalize.py"
)
normalizer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(normalizer)


class TextNormalizationTests(unittest.TestCase):
    def test_explicit_missing_values_only(self):
        for value in ("", " \u00a0", "NULL", "null", "None", "N/A", "NA", "NAN",
                      "UNKNOWN", "ΑΓΝΩΣΤΟ", "άγνωστο", "-", "--", "???"):
            with self.subTest(value=value):
                self.assertIsNone(normalizer.normalize_text(value))
        self.assertEqual(normalizer.normalize_text("-000XXX"), "-000XXX")
        self.assertEqual(normalizer.normalize_text("N/A 150"), "N/A 150")

    def test_whitespace_unicode_and_case(self):
        self.assertEqual(normalizer.normalize_text(" Toyota\u00a0 \tYaris "), "TOYOTA YARIS")
        self.assertEqual(normalizer.normalize_text("Σιτροέν"), "SITROEN")
        self.assertEqual(normalizer.normalize_text("toyota"), "TOYOTA")
        self.assertEqual(normalizer.normalize_text("Ｃ－ＨＲ"), "C-HR")
        self.assertEqual(normalizer.normalize_text("Café"), "CAFÉ")

    def test_punctuation_and_numeric_models(self):
        for value, expected in (
            ("MT–07", "MT-07"), ("MT—07", "MT-07"),
            ("Peugeot 208", "PEUGEOT 208"), ("BMW 320", "BMW 320"),
            ("Honda SH 150", "HONDA SH 150"), ("Toyota C-HR", "TOYOTA C-HR"),
            ("Honda X-ADV", "HONDA X-ADV"),
        ):
            with self.subTest(value=value):
                self.assertEqual(normalizer.normalize_text(value), expected)

    def test_script_aware_comparison_views(self):
        cases = (
            ("Mercedes-Ben", "MERCEDES-BEN", "MERCEDES-BEN", False),
            ("Mercedes-Βen", "MERCEDES-BEN", "MERCEDES-BEN", True),
            ("Mercedes-ΒΕΝ", "MERCEDES-BEN", "MERCEDES-BEN", True),
            ("ΒΕΝ", "VEN", "VEN", False),
            ("Nte Sevo", "NTE SEVO", "NTE SEVO", False),
            ("MΑZDA", "MAZDA", "MAZDA", True),
            ("CITROΕN", "CITROEN", "CITROEN", True),
        )
        for raw, normalized, comparison, mixed in cases:
            with self.subTest(raw=raw):
                actual = normalizer.normalize_text(raw)
                self.assertEqual(actual, normalized)
                self.assertEqual(normalizer.latin_comparison_text(raw), comparison)
                self.assertEqual(normalizer.has_mixed_greek_latin(raw), mixed)

        self.assertEqual(normalizer.comparison_texts("MERCEDES-ΒEN"),
                         ("MERCEDES-BEN", "MERCEDES-VEN"))
        self.assertEqual(normalizer.comparison_texts("MERCEDES-ΒΕΝ"),
                         ("MERCEDES-BEN", "MERCEDES-VEN"))
        self.assertEqual(normalizer.comparison_texts("ΒΕΝ"), ("VEN", "BEN"))
        self.assertEqual(normalizer.comparison_texts(None), ())

    def test_latin_comparison_is_preserved_and_greek_comparison_is_script_aware(self):
        cases = (
            ("Opel", "OPEL", "OPEL", ("OPEL",)),
            ("Mercedes-Ben", "MERCEDES-BEN", "MERCEDES-BEN", ("MERCEDES-BEN",)),
            ("OΡEL", "OPEL", "OPEL", ("OPEL", "OREL")),
            ("Mercedes-ΒΕΝ", "MERCEDES-BEN", "MERCEDES-BEN",
             ("MERCEDES-BEN", "MERCEDES-VEN")),
            ("ΟΠΕΛ", "OPEL", "OPEL", ("OPEL",)),
            ("Golf+", "GOLF+", "GOLF+", ("GOLF+",)),
            ("Golf +", "GOLF +", "GOLF +", ("GOLF +",)),
        )
        for raw, normalized, primary, alternatives in cases:
            with self.subTest(raw=raw):
                value = normalizer.normalize_text(raw)
                self.assertEqual(value, normalized)
                self.assertEqual(normalizer.latin_comparison_text(value), primary)
                self.assertEqual(normalizer.comparison_texts(raw), alternatives)

        # The mixed spelling ΜΕΡΣΕDES uses Greek sigma (S). A generic
        # transliterator must preserve that evidence instead of guessing the
        # vehicle-specific spelling MERCEDES.
        self.assertEqual(normalizer.normalize_text("ΜΕΡΣΕDES-BEN"), "MEPSEDES-BEN")

    def test_confusable_comparison_alternatives_are_bounded_and_stable(self):
        cases = (
            ("ΒΜW", "VMW", "BMW"),
            ("ΥΑΜΑΗΑ", "YAMAIA", "YAMAHA"),
            ("ΟΡΕL", "OREL", "OPEL"),
            ("ΟΠΕΛ", "OPEL", "OPEL"),
            ("ΣΚΑΘΑΡΙ", "SKATHARI", "SKATHARI"),
            ("ΤΟΥΟΤΑ", "TOYOTA", "TOYOTA"),
        )
        for raw, normalized, required in cases:
            with self.subTest(raw=raw):
                alternatives = normalizer.comparison_texts(raw)
                self.assertEqual(normalizer.normalize_text(raw), normalized)
                self.assertEqual(alternatives[0], normalized)
                self.assertIn(required, alternatives)
                self.assertEqual(len(alternatives), len(set(alternatives)))
                self.assertLessEqual(len(alternatives), normalizer.MAX_COMPARISON_TEXTS)
                self.assertEqual(alternatives, normalizer.comparison_texts(raw))

        long_mixed = " ".join(["ΒΜW"] * 20)
        alternatives = normalizer.comparison_texts(long_mixed)
        self.assertLessEqual(len(alternatives), normalizer.MAX_COMPARISON_TEXTS)
        self.assertEqual(alternatives, normalizer.comparison_texts(long_mixed))

        punctuation_model = "ΟΡΕL MT-07 + PCX125"
        alternatives = normalizer.comparison_texts(punctuation_model)
        for alternative in alternatives:
            self.assertEqual("".join(c for c in punctuation_model if c.isdigit()),
                             "".join(c for c in alternative if c.isdigit()))
            self.assertEqual("".join(c for c in punctuation_model if c in "-+/"),
                             "".join(c for c in alternative if c in "-+/"))
            self.assertEqual(punctuation_model.count(" "), alternative.count(" "))

    def test_dataframe_latin_columns_keep_input_and_normalized_values(self):
        import pandas as pd

        source = pd.DataFrame({
            "brand": ["Opel", "OΡEL", "Mercedes-Ben", "Mercedes-ΒΕΝ", "ΟΠΕΛ"],
            "model": ["Golf+", "Golf +", "MT-07", "MT07", "CBR 600 RR"],
        }, index=["latin", "mixed", "latin-brand", "mixed-brand", "greek"])
        original = source.copy(deep=True)
        result = normalizer.normalize_dataframe(
            source, brand_col="brand", model_col="model", prefix="ab_"
        )
        self.assertEqual(list(result["ab_brand_normalized"]),
                         ["OPEL", "OPEL", "MERCEDES-BEN", "MERCEDES-BEN", "OPEL"])
        self.assertEqual(list(result["ab_brand_latin"]),
                         ["OPEL", "OPEL", "MERCEDES-BEN", "MERCEDES-BEN", "OPEL"])
        self.assertEqual(list(result["ab_model_normalized"]),
                         ["GOLF+", "GOLF +", "MT-07", "MT07", "CBR 600 RR"])
        self.assertEqual(list(result["ab_model_latin"]),
                         ["GOLF+", "GOLF +", "MT-07", "MT07", "CBR 600 RR"])
        pd.testing.assert_frame_equal(source, original)

    def test_meaningful_symbols_and_spacing_remain_distinct(self):
        cases = (
            ("Golf", "GOLF"), ("Golf+", "GOLF+"), ("Golf +", "GOLF +"),
            ("MT-07", "MT-07"), ("MT07", "MT07"),
            ("CBR 600 RR", "CBR 600 RR"), ("CBR600RR", "CBR600RR"),
            ("Yaris/Yaris Cross", "YARIS/YARIS CROSS"),
            ("Yaris / Yaris Cross", "YARIS / YARIS CROSS"),
        )
        normalized = []
        for raw, expected in cases:
            with self.subTest(raw=raw):
                actual = normalizer.normalize_text(raw)
                self.assertEqual(actual, expected)
                self.assertEqual(normalizer.latin_comparison_text(actual), expected)
                normalized.append(actual)
        self.assertEqual(len(normalized), len(set(normalized)))

    def test_comparison_does_not_overwrite_raw_or_normalized_values(self):
        row = normalizer.normalize_row({"brand": "Mercedes-Βen", "model": "Golf +"})
        self.assertEqual((row["brand_raw"], row["model_raw"]),
                         ("Mercedes-Βen", "Golf +"))
        self.assertEqual((row["brand_normalized"], row["model_normalized"]),
                         ("MERCEDES-BEN", "GOLF +"))
        self.assertEqual((row["brand_latin"], row["model_latin"]),
                         ("MERCEDES-BEN", "GOLF +"))

    def test_text_normalization_is_idempotent(self):
        for value in ("  Toyota\u00a0Yaris ", "Σιτροέν", "MT—07", "-000XXX",
                      "ΑΓΝΩΣΤΟ", "Café", "ΑB 150", "Mercedes-Ben",
                      "Mercedes-Βen", "Mercedes-ΒΕΝ", "Golf", "Golf+",
                      "Golf +", "MT07", "CBR 600 RR", "CBR600RR"):
            with self.subTest(value=value):
                once = normalizer.normalize_text(value)
                self.assertEqual(normalizer.normalize_text(once), once)


class QualityFlagTests(unittest.TestCase):
    def test_missing_and_equal_fields(self):
        self.assertEqual(normalizer.quality_flags(None, None),
                         ("BRAND_MISSING", "MODEL_MISSING", "BOTH_MISSING"))
        self.assertEqual(normalizer.quality_flags("BMW", "BMW"), ("BRAND_EQUALS_MODEL",))
        self.assertEqual(normalizer.quality_flags(None, "150"),
                         ("BRAND_MISSING", "NUMERIC_ONLY_MODEL"))

    def test_characteristic_flags_and_order(self):
        self.assertEqual(normalizer.quality_flags("ALFA ROMEO", "ΑΒ 150"),
                         ("MIXED_GREEK_LATIN", "MULTIWORD_BRAND"))
        self.assertEqual(normalizer.quality_flags("123", "456"),
                         ("NUMERIC_ONLY_BRAND", "NUMERIC_ONLY_MODEL"))
        self.assertEqual(normalizer.quality_flags("BMW", "@@@"),
                         ("UNUSUAL_SYMBOLS", "POSSIBLE_GARBAGE"))
        self.assertEqual(normalizer.quality_flags("BMW", "A@B"),
                         ("UNUSUAL_SYMBOLS",))
        self.assertEqual(normalizer.quality_flags("BMW", "X-ADV 150"), ())
        self.assertEqual(normalizer.quality_flags("-000XXX", "SH 150"), ())


class FileTests(unittest.TestCase):
    def test_preservation_and_deterministic_output(self):
        rows = [
            {"brand": " Toyota  ", "model": "Yaris\u00a0  150"},
            {"brand": "???", "model": "MT—07"},
            {"brand": "", "model": "NULL"},
            {"brand": "-000XXX", "model": "Σιτροέν"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.csv"
            target = Path(directory) / "output.csv"
            with source.open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=normalizer.INPUT_COLUMNS)
                writer.writeheader()
                writer.writerows(rows)
            stats = normalizer.normalize_file(source, target)
            first_bytes = target.read_bytes()
            with target.open(encoding="utf-8", newline="") as file:
                actual = list(csv.DictReader(file))
            self.assertEqual(stats["rows"], len(rows))
            self.assertEqual([(r["brand_raw"], r["model_raw"]) for r in actual],
                             [(r["brand"], r["model"]) for r in rows])
            self.assertEqual(actual[1]["model_normalized"], "MT-07")
            self.assertEqual(actual[2]["quality_flags"],
                             "BRAND_MISSING|MODEL_MISSING|BOTH_MISSING")
            normalizer.normalize_file(source, target)
            self.assertEqual(target.read_bytes(), first_bytes)

    def test_invalid_schema_does_not_replace_output(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.csv"
            target = Path(directory) / "output.csv"
            source.write_text("brand\nA\n", encoding="utf-8")
            target.write_text("existing", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "model"):
                normalizer.normalize_file(source, target)
            self.assertEqual(target.read_text(encoding="utf-8"), "existing")


if __name__ == "__main__":
    unittest.main()
