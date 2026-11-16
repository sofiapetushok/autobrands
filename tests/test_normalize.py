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
        self.assertEqual(normalizer.normalize_text("Σιτροέν"), "ΣΙΤΡΟΕΝ")
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

    def test_text_normalization_is_idempotent(self):
        for value in ("  Toyota\u00a0Yaris ", "Σιτροέν", "MT—07", "-000XXX",
                      "ΑΓΝΩΣΤΟ", "Café", "ΑB 150"):
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
