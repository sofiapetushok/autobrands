"""Integration smoke test: generate temporary inputs; no claim files are published."""
import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "data" / "vehicle_catalog.csv"
sys.path.insert(0, str(ROOT))
import normalize


class NormalizationRegressionTest(unittest.TestCase):
    def test_exact_values_and_original_text(self):
        cases = [
            ("  Τογιότα\t", "ＭＴ－０９", "ΤΟΓΙΟΤΑ", "MT-09", "TOGIOTA", "MT-09", "MIXED_GREEK_LATIN"),
            ("TOYΟTA", "Yaris Cross", "TOYΟTA", "YARIS CROSS", "TOYOTA", "YARIS CROSS", "MIXED_GREEK_LATIN"),
            ("Σούζουκι", "16+", "ΣΟΥΖΟΥΚΙ", "16+", "SUZUKI", "16+", "NUMERIC_ONLY_MODEL"),
            ("Toyota", "Land Cruiser/GT 2.0+", "TOYOTA", "LAND CRUISER/GT 2.0+", "TOYOTA", "LAND CRUISER/GT 2.0+", ""),
        ]
        for brand, model, normalized_brand, normalized_model, latin_brand, latin_model, flags in cases:
            with self.subTest(brand=brand, model=model):
                row = normalize.normalize_row({"brand": brand, "model": model})
                self.assertEqual((row["brand_raw"], row["model_raw"]), (brand, model))
                self.assertEqual((row["brand_normalized"], row["model_normalized"]),
                                 (normalized_brand, normalized_model))
                self.assertEqual((row["brand_latin"], row["model_latin"]), (latin_brand, latin_model))
                self.assertEqual(row["quality_flags"], flags)
                self.assertEqual((row["brand_missing"], row["model_missing"]), ("false", "false"))

    def test_missing_placeholders_and_flags(self):
        for value in ("", " NA ", "null", "N/A", "---", "???", "άγνωστο"):
            with self.subTest(value=value):
                row = normalize.normalize_row({"brand": value, "model": value})
                self.assertEqual((row["brand_raw"], row["model_raw"]), (value, value))
                self.assertEqual((row["brand_normalized"], row["model_normalized"]), ("", ""))
                self.assertEqual((row["brand_latin"], row["model_latin"]), ("", ""))
                self.assertEqual((row["brand_missing"], row["model_missing"]), ("true", "true"))
                self.assertEqual(row["quality_flags"], "BRAND_MISSING|MODEL_MISSING|BOTH_MISSING")

    def test_current_unresolved_character_behavior(self):
        # These assertions record existing behavior, not a product decision.
        cases = [("Ν/Α", "Ν/Α", "N/A"), ("é", "É", "É"), ("\u200b", "\u200b", "\u200b")]
        for value, normalized, latin in cases:
            with self.subTest(value=value):
                row = normalize.normalize_row({"brand": value, "model": ""})
                self.assertEqual((row["brand_normalized"], row["brand_latin"]), (normalized, latin))
                self.assertEqual(row["brand_missing"], "false")
        self.assertEqual(normalize.normalize_row({"brand": "\u200b", "model": ""})["quality_flags"],
                         "MODEL_MISSING|UNUSUAL_SYMBOLS|POSSIBLE_GARBAGE")

    def test_idempotence(self):
        for value in ("  Σούζουκι\t", "Σου\u0301ζουκι", "e\u0301", "ＭＴ－０９", "TOYΟTA", "MT‐09/16+", "\u200b", "---"):
            with self.subTest(value=value):
                once = normalize.normalize_text(value)
                self.assertEqual(normalize.normalize_text(once), once)

    def test_bom_duplicate_order_and_original_values(self):
        rows = [{"brand": "Σούζουκι ", "model": "MT‐09"},
                {"brand": " Toyota", "model": "Yaris"},
                {"brand": "Σούζουκι ", "model": "MT‐09"}]
        with tempfile.TemporaryDirectory() as temp:
            source, output = Path(temp) / "input.csv", Path(temp) / "output.csv"
            with source.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["brand", "model"])
                writer.writeheader()
                writer.writerows(rows)
            stats = normalize.normalize_file(source, output)
            with output.open(encoding="utf-8", newline="") as stream:
                result = list(csv.DictReader(stream))
            self.assertEqual(stats["rows"], 3)
            self.assertEqual([(r["brand_raw"], r["model_raw"]) for r in result],
                             [(r["brand"], r["model"]) for r in rows])
            self.assertEqual(result[0], result[2])
            self.assertEqual(result[0]["model_normalized"], "MT-09")

    def test_empty_csv_and_header_only(self):
        with tempfile.TemporaryDirectory() as temp:
            source, output = Path(temp) / "input.csv", Path(temp) / "output.csv"
            for contents, valid in (("", False), ("brand,model\n", True)):
                with self.subTest(contents=contents):
                    source.write_text(contents, encoding="utf-8")
                    if not valid:
                        with self.assertRaisesRegex(ValueError, "Missing input columns"):
                            normalize.normalize_file(source, output)
                        self.assertFalse(output.exists())
                    else:
                        self.assertEqual(normalize.normalize_file(source, output)["rows"], 0)
                        with output.open(encoding="utf-8", newline="") as stream:
                            reader = csv.DictReader(stream)
                            self.assertEqual(reader.fieldnames, list(normalize.OUTPUT_COLUMNS))
                            self.assertEqual(list(reader), [])

    def test_invalid_csv_preserves_existing_output_atomically(self):
        cases = [("brand\nToyota\n", "Missing input columns"),
                 ("brand,model\nToyota,Yaris\nHonda\n", "Malformed CSV row"),
                 ("brand,model\nToyota,Yaris\nHonda,Civic,extra\n", "Malformed CSV row"),
                 (b"brand,model\nToyota,Yaris\nHonda,\xff\n", None)]
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            source, output = directory / "input.csv", directory / "output.csv"
            for contents, message in cases:
                with self.subTest(contents=contents):
                    source.write_bytes(contents if isinstance(contents, bytes) else contents.encode("utf-8"))
                    output.write_bytes(b"existing output\n")
                    if message:
                        with self.assertRaisesRegex(ValueError, message):
                            normalize.normalize_file(source, output)
                    else:
                        with self.assertRaises(UnicodeDecodeError):
                            normalize.normalize_file(source, output)
                    self.assertEqual(output.read_bytes(), b"existing output\n")
                    self.assertEqual({p.name for p in directory.iterdir()}, {"input.csv", "output.csv"})


class PipelineSmokeTest(unittest.TestCase):
    def test_end_to_end(self):
        with CATALOG.open(encoding="utf-8-sig", newline="") as stream:
            candidate = next(
                (row for row in csv.DictReader(stream)
                 if row.get("brand") and row.get("model")),
                None,
            )
        self.assertIsNotNone(candidate, "The reference catalog is empty.")

        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            raw = directory / "input.csv"
            normalized = directory / "normalized.csv"
            matched = directory / "matched.csv"

            with raw.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["brand", "model"])
                writer.writeheader()
                writer.writerows([{
                    "brand": candidate["brand"],
                    "model": candidate["model"],
                }] * 2)

            subprocess.run(
                [sys.executable, str(ROOT / "explore.py"), str(raw)],
                check=True, capture_output=True, text=True,
            )
            subprocess.run(
                [sys.executable, str(ROOT / "normalize.py"), str(raw), str(normalized)],
                check=True, capture_output=True, text=True,
            )
            subprocess.run(
                [sys.executable, str(ROOT / "run.py"), "--input", str(normalized),
                 "--catalog", str(CATALOG), "--output", str(matched)],
                check=True, capture_output=True, text=True,
            )
            with matched.open(encoding="utf-8-sig", newline="") as stream:
                results = list(csv.DictReader(stream))

            self.assertEqual(len(results), 2)
            self.assertNotIn("claim_id", results[0])
            self.assertEqual(results[0], results[1])
            with normalized.open(encoding="utf-8", newline="") as stream:
                self.assertNotIn("claim_id", csv.DictReader(stream).fieldnames)
            self.assertIn("match_status", results[0])
            self.assertIn("extra_info", results[0])
            self.assertIn("brand_latin", results[0])



if __name__ == "__main__":
    unittest.main()
