"""Integration smoke test: generate temporary inputs; no claim files are published."""
import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "data" / "vehicle_catalog.csv"


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
