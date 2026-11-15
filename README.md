# AutoBrands

**Explainable, catalog-aware vehicle make/model entity resolution in Python.**

AutoBrands standardizes noisy vehicle names using Unicode normalization, Greek-to-Latin comparison, exact/alias matching, and guarded fuzzy ranking. It keeps uncertain identities unresolved rather than inventing vehicle variants.

## Capabilities

- CSV profiling: missingness, recurring values, writing systems
- Source-preserving text normalization and Greek/Latin comparison
- Exact, alias, fuzzy and cross-field identification against a reference catalog
- Variant and numeric-code safeguards; ranked alternatives for ambiguous inputs
- Match statuses, similarity scores and unverified additional text (`extra_info`)
- Optional LLM routing **dry-run only**; no inference API calls

## Quick start

Python 3.10+:

```bash
python -m pip install -r requirements.txt
python 01_explore.py path/to/input.csv
python 02_normalize.py path/to/input.csv output/normalized.csv
python 03_match.py --input output/normalized.csv --catalog data/vehicle_catalog.csv --output output/matched.csv
python -m unittest discover -s tests -v
```

Your input CSV must contain `claim_id,brand,model`. **No claim-level input records or sample claims are distributed in this repository.** Tests generate temporary input in memory/on disk during execution.

The reference catalog schema is `brand,model,vehicle_type,aliases,source`. It contains 14,742 deduplicated reference rows from the attributed open datasets below, not 14,742 unique manufacturer-confirmed identities. It may contain omissions, aliases and inconsistencies.

Match scores represent textual similarity, **not calibrated probabilities**. `extra_info` preserves reported text and does not verify fuel type, vehicle specifications or compatibility. Full-match coverage and accuracy are different measures.

## Reference data and attribution

The included `data/vehicle_catalog.csv` is a **modified, merged and deduplicated** make/model reference dataset sourced from:

1. [VehiclesDB](https://github.com/vehiclesdb/vehiclesdb), dataset version **2026.11.10**, **CC BY 4.0**. Credit: [Vehicle data by VehiclesDB](https://vehiclesdb.com). See its [upstream attribution notices](https://github.com/vehiclesdb/vehiclesdb/blob/main/ATTRIBUTION.md), which may include notices from the underlying official registers.
2. [MotorcycleName](https://github.com/ixiehao/MotorcycleName), version **1.1.1**, **CC BY-SA 4.0**.

The AutoBrands catalog is a transformed combination: it selects the relevant names, harmonizes vehicle types, merges overlapping make/model records and may incorporate aliases. The **combined dataset is distributed under CC BY-SA 4.0**, preserving source attribution and share-alike obligations. The Python source code is licensed separately under MIT (`LICENSE`). Source data is not relicensed under MIT.

See [data/README.md](data/README.md) for dataset provenance and required upstream notices. Third-party brand names remain the property of their respective owners.

## Limitations

- Catalog coverage is incomplete; similar names can indicate distinct vehicles.
- Fuzzy confidence is not calibrated and may produce wrong matches.
- Greeklish, shorthand and multiple simultaneous corruptions remain challenging.
- No independently established accuracy figure is claimed for this release.
- Not a VIN decoder or validated vehicle-specifications service.

## Files

| Path | Purpose |
| --- | --- |
| `01_explore.py` | Input profiling |
| `02_normalize.py` | Normalization |
| `03_match.py` | Catalog matching |
| `data/vehicle_catalog.csv` | Attributed open reference data |
| `tests/test_smoke.py` | Temporary-input end-to-end smoke test |

For reproducibility, use the exact catalog snapshot committed here; upstream datasets may change over time.
