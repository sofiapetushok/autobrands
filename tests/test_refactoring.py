"""Full decision snapshots from pre-refactor commit 72b0751; no claims data."""
import csv
import dataclasses
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from autobrands.matching import Matcher
from autobrands.llm import LLMDecisionCache

CASES = [
    ('Toyota', 'Yaris', 'fc2de88ecc6f3e1e36b17e96a148b4cd04060ca1cca94dc16389b419e92a51ed'),
    ('Toyota', 'Yaris Cross', '405e272fd6e1f40bfd7dd5ea31d1685f0caf35e6a77c6c5ecf20650fa510e66c'),
    ('Yaris', 'Toyota', 'b49ec8e8a91d26077a5c61170f3c418c73d91f0f3d2aee3b56f755ba8b02098c'),
    ('Corolla', 'Toyota', '0fa71beef0e678d2feb2b93100facb98dc7ae907df045ee8de5ae9a0aec2f173'),
    ('Toyota Yaris', '', '0e751dcf8a8c238a9d2153d1f6a27cbf78cf689e4fef4c0e1ca4d46d9a36b089'),
    ('', 'Toyota Yaris', '0e751dcf8a8c238a9d2153d1f6a27cbf78cf689e4fef4c0e1ca4d46d9a36b089'),
    ('ToyotaYaris', '', '127bb89306d573e2e30fc059c028f05042e5de08deb44dbe8ad40234fc38195a'),
    ('SUSUKI', 'Swift', '4692427f2118ec231e66b01b2db3d0d9538d050a3288437a31c3d4dbe0ab5ab5'),
    ('Peug', '3008', '0d276cf26574289a32730809ac728621802313333d3bd3c973d85ee95cbcfefa'),
    ('Nissan', 'Leeaf', '98840fe1189bc73e16ff0b448da4d39c3f71b66a5fdcd4f454832f61f8083d07'),
    ('Yamaha', 'MT-07', '8fd5e09d774f4e6edec02e65986f0ff6ff3f8726a471ebc031cb753c1b71559b'),
    ('Yamaha', 'MT-09', 'd7a6a9dc8180452cb9892c11d9c421e3c6224e2ab75af3743d882a991d9215b7'),
    ('Mercedes-Benz', 'SPRRINTER', 'b2411c25f8474ed034aa68afc0dbcbc54574642194b568b9eae76fe135213083'),
    ('Tesla', 'Mode', '7c393ccbe45236bcf30c86c0247029193ac4ec3e13f2b6f2fbdedfb83515b782'),
    ('Volkswagen', 'GLF', '6791fcceefd77a9cac9ad4afc7067d650a29cdbd4e1daf7c147a61d442f47a05'),
    ('Toyota Corolla', 'Honda', 'e413f3c9805308e8ec84397281b36315f014644d25272b35b74b809dd0dde41b'),
    ('', '650', 'a019bce6ccd553a2373b93822ac2b3a546a58ed4aa41d818605eb1baf373deba'),
    ('Toyota', 'Corolla Crosss', 'c6f7acdc03edeff04a9388a66d875e7813e8b362d6ca5c238bdc7fd9ed8b58ca'),
    ('Toyota', 'Land Cruiser DIESEL', 'c657bd442c186a8aa923cfd9edcf24f24320d711bc42ca37e165129724ed5941'),
    ('Toyota', 'Yaris Sport', '48e9f9f4fbd5164315f5f6e741d6e9998f6fad0e4d07ea8fe768893433d646e6'),
    ('Toyota', 'Yaris GT', '48e9f9f4fbd5164315f5f6e741d6e9998f6fad0e4d07ea8fe768893433d646e6'),
    ('Toyota', 'Yaris 350', '48e9f9f4fbd5164315f5f6e741d6e9998f6fad0e4d07ea8fe768893433d646e6'),
    ('Toyota', 'Yaris 16+', '48e9f9f4fbd5164315f5f6e741d6e9998f6fad0e4d07ea8fe768893433d646e6'),
    ('', '', 'afd87df01343e753eb1dd4634b4666a0b15a57befb9a035cc759f17085eed0c7'),
    ('\u03a4\u039f\u0393\u0399\u039f\u03a4\u0391', '\u0393\u0399\u0391\u03a1\u0399\u03a3', 'bbfd4a00caafc2974cc603ebf3976506c2027bc05219781e9674ef8e6bc4f702'),
    ('Toyota', '\u0393\u0399\u0391\u03a1\u0399\u03a3', '340797e5883b0067410c3ba080e0dc7a05a0da709b2675a004ce52804abb2335'),
    ('Toyota', 'Land Cruiser UNKNOWN', 'f56e3df1d3185c85d5a827a44781846dcaf90cb904355a1f81b6b05b3e0aed0b'),
]


class RefactoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with (ROOT / "data/vehicle_catalog.csv").open(encoding="utf-8-sig", newline="") as stream:
            cls.matcher = Matcher(list(csv.DictReader(stream)))

    def test_original_decisions_are_unchanged(self):
        self.assertTrue(CASES)
        for brand, model, expected in CASES:
            with self.subTest(brand=brand, model=model):
                decision = self.matcher.resolve(brand, model)
                payload = [dataclasses.asdict(decision), self.matcher.serialize_candidates(decision)]
                actual = hashlib.sha256(json.dumps(payload, ensure_ascii=True, sort_keys=True).encode()).hexdigest()
                self.assertEqual(actual, expected)

    def test_llm_cache_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "cache.json"
            cache = LLMDecisionCache(path)
            cache.responses["test-request"] = '{"decision": "ABSTAIN"}'
            cache.save()
            self.assertEqual(LLMDecisionCache(path).responses, cache.responses)
            self.assertEqual(list(path.parent.iterdir()), [path])

    def test_matching_import_does_not_load_llm_or_cli(self):
        subprocess.run([sys.executable, "-c",
                        "import sys; import autobrands.matching; "
                        "assert 'autobrands.llm' not in sys.modules; "
                        "assert '03_match' not in sys.modules"], cwd=ROOT, check=True)


if __name__ == "__main__":
    unittest.main()
