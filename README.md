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
|-- 01_explore.py             # Dataset profiling
|-- 02_normalize.py           # Source-preserving normalization
|-- 03_match.py               # CSV CLI and reporting
|-- autobrands/
|   |-- __init__.py
|   |-- matching.py           # Deterministic catalog matching
|   `-- llm.py                # Optional routing, validation and caching
|-- data/
|   |-- vehicle_catalog.csv   # Open, attributed reference catalog
|   `-- README.md             # Sources and licensing
|-- tests/
|   |-- test_matching.py      # Matching regression tests
|   |-- test_refactoring.py   # Original decision snapshots and module checks
|   `-- test_smoke.py         # Temporary-input CLI and dry-run smoke test
|-- requirements.txt
|-- LICENSE
`-- README.md
```

The matching module does not import the optional LLM module. The CLI coordinates
CSV input/output and opt-in LLM processing; matching thresholds and acceptance
rules remain unchanged.

## Tests and refactoring verification

Run the complete checked-in suite:

```bash
python -m unittest discover -s tests -v
```

All 12 tests passed during refactoring verification. Coverage includes swapped
and combined fields, Greek/Latin input, fuzzy typos, closely related variants,
LLM cache persistence, and all three CLI commands. The LLM dry-run smoke test
verifies zero API attempts and no output CSV.

The suite includes 27 complete decision snapshots from pre-refactor commit
`72b0751`, covering identities, statuses, scores, candidate ordering and
`extra_info`. A separate local comparison of the original and refactored CLI
produced byte-for-byte identical CSVs for all 15,000 claims, with row counts and
IDs preserved. LLM dry-run routing also matched. Those private/local inputs and
generated outputs are not included in Git; these checks establish refactoring
parity, not independently labeled matching accuracy.


## Known limitations

Fuzzy matches can still be wrong, particularly for near-identical vehicle variants, missing model numbers, abbreviations and multiple spelling errors. The catalog may contain omissions or inconsistent aliases. This project is not a VIN decoder or validated vehicle specification database.

### Existing fuel-token behavior

The separate local legacy suite passed 64 of 66 tests, with three failed
assertions across two tests. The same assertions fail against the pre-refactor
GitHub implementation:

- `test_recognized_descriptions_require_complete_catalog_identity`: the
  `PIONEER ΒΕΝΖΙΝΗ` and `PIONEER PETROL` fixtures return
  `CATALOG_MODEL_REPORTED_FUEL` instead of the older expected method
  `CATALOG_SUPPLEMENTARY_TEXT`.
- `test_unfamiliar_or_catalog_defining_tokens_are_preserved`: the
  `PIONEER DIESEL` fixture is accepted as `FUZZY_MATCH`, while the legacy
  assertion expects abstention.

These are existing fuel-token behavior differences from legacy expectations,
not regressions introduced by module extraction. They remain unchanged in this
release; reported fuel information must not be treated as verified vehicle
specification. The external legacy suite is not part of the 12 checked-in tests.
