import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

class DemoPipelineTest(unittest.TestCase):
    def test_pipeline_preserves_rows_and_has_candidates(self):
        with tempfile.TemporaryDirectory() as td:
            norm = Path(td) / 'normalized.csv'
            matched = Path(td) / 'matched.csv'
            subprocess.run([sys.executable, str(ROOT/'02_normalize.py'), str(ROOT/'data/demo_claims.csv'), str(norm)], check=True, capture_output=True, text=True)
            subprocess.run([sys.executable, str(ROOT/'03_match.py'), '--input', str(norm), '--catalog', str(ROOT/'data/demo_catalog.csv'), '--output', str(matched)], check=True, capture_output=True, text=True)
            with (ROOT/'data/demo_claims.csv').open(encoding='utf-8-sig',newline='') as fp: source=list(csv.DictReader(fp))
            with matched.open(encoding='utf-8',newline='') as fp: rows=list(csv.DictReader(fp))
            self.assertEqual([x['claim_id'] for x in source],[x['claim_id'] for x in rows])
            self.assertTrue(any(x['match_status'] in {'EXACT_MATCH','ALIAS_MATCH','FUZZY_MATCH'} for x in rows))
            self.assertTrue(all('extra_info' in x and 'brand_latin' in x for x in rows))

if __name__=='__main__':
    unittest.main()
