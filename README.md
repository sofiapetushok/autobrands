# AutoBrands

**Multilingual vehicle make/model entity resolution with explainable fuzzy matching.**

AutoBrands is a Python pipeline that identifies and standardizes vehicle makes and models from inconsistent text. It combines Unicode normalization, Greek-to-Latin comparison, catalog-aware candidate retrieval, and conservative decision rules.

## Features

- **Text normalization:** case, spacing, Unicode and Greek/Latin comparison, while retaining the original text.
- **Matching:** exact, alias, fuzzy and cross-field candidate matching.
- **Safer decisions:** safeguards for model numbers, variants, conflicting evidence and close competitors.
- **Traceable output:** matched make/model, status, method, score, ranked alternatives and `extra_info`.
- **Optional LLM routing:** dry-run planning only; no API calls or LLM inference are included.

## Getting started

Requires Python 3.10+.

```bash
python -m pip install -r requirements.txt

# Inspect a local CSV
python 01_explore.py path/to/input.csv

# Normalize your own input
python 02_normalize.py path/to/input.csv output/normalized.csv

# Match against the included reference catalog
python 03_match.py --input output/normalized.csv --catalog data/vehicle_catalog.csv --output output/matched.csv

# Run the smoke test (generates temporary input)
python -m unittest discover -s tests -v
```

Input CSV columns: `claim_id,brand,model`. The catalog columns are `brand,model,vehicle_type,aliases,source`.

**No input records, claims dataset, or synthetic claims CSV is included in this repository.** Bring your own input file. The smoke test generates its input in a temporary directory. Generated outputs and local inputs are not intended for version control.

## Matching results

Full accepted matches use `EXACT_MATCH`, `ALIAS_MATCH` or `FUZZY_MATCH`. Other statuses, including `PARTIAL_MATCH`, `AMBIGUOUS`, `CONFLICT` and `UNRESOLVED`, should not be treated as verified full identities.

A similarity score is **not a calibrated probability**. The `extra_info` field preserves additional source text; it does not verify technical attributes such as fuel type. Catalog coverage and matching accuracy are separate questions, and this release does not claim independently verified real-world accuracy.

## Open reference catalog

`data/vehicle_catalog.csv` contains **14,742 reference rows**, merged and transformed from two publicly available GitHub datasets:

| Source | Version | Original license |
| --- | --- | --- |
| [VehiclesDB](https://github.com/vehiclesdb/vehiclesdb) | 2026.11.10 | CC BY 4.0 |
| [MotorcycleName](https://github.com/ixiehao/MotorcycleName) | 1.1.1 | CC BY-SA 4.0 |

The dataset is a modified compilation. Source records were selected, harmonized, merged and deduplicated, with alternative model names stored where available. It is not an authoritative list of all vehicles or variants.

**Licensing:** The Python code is MIT-licensed (`LICENSE`); the combined and modified reference catalog is documented under **CC BY-SA 4.0**, with upstream credits and applicable attribution requirements preserved. See [data/README.md](data/README.md) for full provenance, modification notes and links to upstream attribution notices.

## Project structure

```text
autobrands/
├── 01_explore.py             # Dataset profiling
├── 02_normalize.py           # Source-preserving text normalization
├── 03_match.py               # Catalog-aware entity resolution
├── data/
│   ├── vehicle_catalog.csv  # Open, attributed reference catalog
│   └── README.md            # Dataset sources and licensing
├── tests/
│   └── test_smoke.py        # Temporary-input integration smoke test
├── requirements.txt
├── LICENSE
└── README.md
```

## Known limitations

Fuzzy matches can still be wrong, particularly for near-identical vehicle variants, missing model numbers, abbreviations and multiple spelling errors. The catalog may contain omissions or inconsistent aliases. This project is not a VIN decoder or validated vehicle specification database.
