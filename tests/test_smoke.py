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
                writer = csv.DictWriter(stream, fieldnames=["claim_id", "brand", "model"])
                writer.writeheader()
                writer.writerow({
                    "claim_id": "TEMP-001",
                    "brand": candidate["brand"],
                    "model": candidate["model"],
                })

            subprocess.run(
                [sys.executable, str(ROOT / "01_explore.py"), str(raw)],
                check=True, capture_output=True, text=True,
            )
            subprocess.run(
                [sys.executable, str(ROOT / "02_normalize.py"), str(raw), str(normalized)],
                check=True, capture_output=True, text=True,
            )
            subprocess.run(
                [sys.executable, str(ROOT / "03_match.py"), "--input", str(normalized),
                 "--catalog", str(CATALOG), "--output", str(matched)],
                check=True, capture_output=True, text=True,
            )
            with matched.open(encoding="utf-8-sig", newline="") as stream:
                results = list(csv.DictReader(stream))

            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["claim_id"], "TEMP-001")
            self.assertIn("match_status", results[0])
            self.assertIn("extra_info", results[0])
            self.assertIn("brand_latin", results[0])

            dry_output = directory / "dry-run.csv"
            dry_run = subprocess.run(
                [sys.executable, str(ROOT / "03_match.py"), "--input", str(normalized),
                 "--catalog", str(CATALOG), "--output", str(dry_output), "--llm-dry-run"],
                check=True, capture_output=True, text=True,
            )
            self.assertFalse(dry_output.exists())
            self.assertIn("Actual API attempts: 0", dry_run.stdout)


if __name__ == "__main__":
    unittest.main()
