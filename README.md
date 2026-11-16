# AutoBrands

**Python-based vehicle make and model matching using fuzzy string matching, multilingual text normalization, and catalog-backed entity resolution.**

AutoBrands standardizes inconsistent car and motorcycle make/model descriptions, handling typos, Greek/Latin spellings, missing fields, swapped columns, and ambiguous model variants.

Built with Python and RapidFuzz, with an emphasis on explainable matching decisions and conservative automatic acceptance.

## Features

- **Text normalization:** case, spacing, Unicode and Greek/Latin comparison, while retaining the original text.
- **Matching:** exact, alias, fuzzy and cross-field candidate matching.
- **Safer decisions:** safeguards for model numbers, variants, conflicting evidence and close competitors.
- **Traceable output:** matched make/model, status, method, score, ranked alternatives and `extra_info`.

## Getting started

Requires Python 3.10+.

```bash
python -m pip install -r requirements.txt

# Inspect a local CSV
python explore.py path/to/input.csv

# Normalize your own input
python normalize.py path/to/input.csv output/normalized.csv

# Match against the included reference catalog
python run.py --input output/normalized.csv --catalog data/vehicle_catalog.csv --output output/matched.csv

# Run regression and integration tests (temporary inputs)
python -m unittest discover -s tests -v
```

Input CSV columns: `brand,model`. No record identifier is required or included in output. Input row order and duplicate rows are preserved. The catalog columns are `brand,model,vehicle_type,aliases,source`.

**No input records, claims dataset, or synthetic claims CSV is included in this repository.** Bring your own input file. The smoke test generates its input in a temporary directory. Generated outputs and local inputs are not intended for version control.

## Matching results

Full accepted matches use `EXACT_MATCH`, `ALIAS_MATCH` or `FUZZY_MATCH`. Other statuses, including `PARTIAL_MATCH`, `AMBIGUOUS`, `CONFLICT` and `UNRESOLVED`, should not be treated as verified full identities.

A similarity score is **not a calibrated probability**. The `extra_info` field preserves additional source text; it does not verify technical attributes such as fuel type. Catalog coverage and matching accuracy are separate questions, and this release does not claim independently verified real-world accuracy.

## Reference catalog: source of truth and attribution

The **only reference catalog** used by the default pipeline is `data/vehicle_catalog.csv` (14,742 rows). Its schema is `brand,model,vehicle_type,aliases,source`. No claims or demo input data are published.

| Upstream dataset | Version | Original license | CSV source label |
| --- | --- | --- | --- |
| [VehiclesDB](https://github.com/vehiclesdb/vehiclesdb) | 2026.11.10 | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | `VehiclesDB v2026.11.10` |
| [MotorcycleName](https://github.com/ixiehao/MotorcycleName) | 1.1.1 | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) | `MotorcycleName v1.1.1` |

The catalog is **modified and combined**, not a verbatim source dump. Make/model names were selected and merged, vehicle types harmonized, overlapping entries deduplicated, and alternative names recorded in `aliases` where available. A pipe-separated `source` label indicates multiple contributing sources. The catalog does not preserve every original upstream field.

**Attribution:** Vehicle data by [VehiclesDB](https://vehiclesdb.com). Preserve the [VehiclesDB attribution notices](https://github.com/vehiclesdb/vehiclesdb/blob/main/ATTRIBUTION.md) and [source information](https://github.com/vehiclesdb/vehiclesdb/blob/main/SOURCES.md), including applicable underlying official-register notices. Motorcycle names from [MotorcycleName](https://github.com/ixiehao/MotorcycleName).

**Licensing:** Python source code is MIT-licensed (`LICENSE`). The transformed combined catalog is separately distributed under **CC BY-SA 4.0** with attribution to both sources and notices of changes; upstream CC BY 4.0 notices remain applicable. Third-party trademarks remain with their owners.

### One source of truth per concern

- `data/vehicle_catalog.csv`: make/model reference records.
- `normalize.py`: normalization rules.
- `matching.py`: matching and fuzzy decision rules.
- `run.py`: CSV entry point; it does not implement another matcher.
- `tests/`: regression verification, not runtime data.

## Project structure

```text
autobrands/
|-- explore.py               # Optional dataset profiling
|-- normalize.py             # Source-preserving normalization
|-- matching.py              # The only matching implementation
|-- run.py                   # CSV processing and CLI
|-- data/
|   `-- vehicle_catalog.csv  # The only reference catalog
|-- tests/
|   |-- test_matching.py     # Matching regression tests
|   |-- test_refactoring.py  # Original decision snapshots and import boundary
|   `-- test_smoke.py        # Temporary-input CLI integration test
|-- requirements.txt
|-- LICENSE
`-- README.md
```

`matching.py` imports normalization directly from `normalize.py`. The CLI
coordinates CSV input/output without implementing matching rules.

## Tests and verification

Run the checked-in test suite:

```bash
python -m unittest discover -s tests -v
```

The **11 checked-in tests** cover swapped/combined fields, Greek/Latin input, fuzzy misspellings, distinct model variants, import boundaries and the command-line workflow. Inputs generated by tests are temporary; no claims dataset is published.

Refactoring was verified against **27 original decision snapshots** and a separate **15,000-row local comparison**: the original and refactored CSV outputs were byte-for-byte identical. These tests confirm **behavioral parity**, not matching accuracy on independently labeled data.

## Known limitations

The reference catalog is not exhaustive. Abbreviations, heavily corrupted inputs, near-identical model variants and catalog gaps can lead to abstentions or incorrect matches. Similarity scores are not calibrated probabilities, and reported fuel text in `extra_info` is not verified vehicle metadata.

The separate local legacy normalization/matching suite passes **45/47 tests**. Three assertions in two tests disagree with pre-existing fuel-token behavior: two expect `CATALOG_SUPPLEMENTARY_TEXT` instead of `CATALOG_MODEL_REPORTED_FUEL`, and one expects abstention for the `PIONEER DIESEL` fixture. These same assertions failed before cleanup. Matching behavior has intentionally not been changed to satisfy those assertions; this legacy suite is not included in the 11 checked-in tests.
