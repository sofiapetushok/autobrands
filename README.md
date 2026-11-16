# AutoBrands

AutoBrands is a deterministic, catalog-driven vehicle make/model resolver for
noisy CSV data. It normalizes Unicode text, compares Greek and Latin forms,
handles swapped or combined fields, ranks catalog candidates, and abstains when
the evidence is insufficient. It does not use an LLM.

## Project layout

```text
data/vehicle_catalog.csv  Reference catalog
tests/                    Regression and integration tests
normalize.py              Loss-conscious CSV normalization
run.py                    Matching engine, review workflow, and CLI
```

Generated outputs and the local `data/verified_mappings.csv` are intentionally
excluded from Git.

## Installation

Python 3.11 or later is recommended. RapidFuzz is the only runtime dependency.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Input and commands

The raw UTF-8 CSV must contain `brand,model`. Values and duplicate rows are
preserved in their original order; no row identifier is required.

```powershell
python normalize.py input.csv output/normalized.csv
python run.py --input output/normalized.csv --output output/matched.csv
```

Use another catalog with `--catalog`. A catalog must contain
`brand,model,vehicle_type,aliases,source`. `aliases` is a pipe-separated list of
verified model aliases. An independently verified `brand,alias` CSV can be
supplied with `--brand-aliases`.

Matching retains the raw and normalized inputs, accepted canonical identity,
detailed status and method, similarity score, confidence, ranked candidates,
and supplementary text. Similarity scores are evidence, not probabilities.

### User-facing statuses

- `MATCHED`: accepted exact, alias, fuzzy, or human-verified identity.
- `REVIEW`: partial, ambiguous, or conflicting evidence requiring a person.
- `UNRESOLVED`: no supported identity or useful candidate.

The detailed `match_status` and `match_method` fields remain available for
diagnostics. `top_candidates` is ranked catalog evidence, never an automatic
human selection.

## Human review

Export non-matched rows for review:

```powershell
python run.py --input output/normalized.csv --output output/matched.csv `
  --review-output output/review.csv
```

For each reviewed row, set `action` explicitly:

- `MATCH`: provide a canonical brand/model that already exists in the catalog.
- `NO_MATCH`: leave canonical fields empty.
- `REPLACE`: explicitly replace an existing confirmation with another catalog identity.
- `REVOKE`: remove an existing confirmation and leave canonical fields empty.
- `SKIP` or blank: make no persistent change.

Import the completed file:

```powershell
python run.py --import-reviews output/review.csv
```

Validated decisions are stored only in `data/verified_mappings.csv` and reused
on later runs. Duplicate confirmations are idempotent; contradictory updates
are rejected unless `REPLACE` or `REVOKE` is explicit. `NO_MATCH` is persisted
as an unresolved human decision. The reference catalog is never modified.

## Tests

```powershell
python -m unittest discover -s tests -v
```

The suite covers exact and fuzzy matching, aliases, variants, numeric and
punctuation safeguards, Greek/Latin text, cross-field inputs, ambiguity,
candidate ranking, normalization, atomic CSV output, duplicates, order, and the
review/import workflow.

## Reference catalog provenance and license

`data/vehicle_catalog.csv` contains 14,742 normalized and merged reference rows
from two pinned sources retrieved on 13 November 2026:

- [VehiclesDB v2026.11.10](https://github.com/vehiclesdb/vehiclesdb/releases/tag/v2026.11.10),
  licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- [MotorcycleName v1.1.1, commit 32eaf412](https://github.com/ixiehao/MotorcycleName/tree/32eaf412eb13c57104f29b186e55e91a785d658a),
  licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).

The project normalized names, mapped categories, merged exact normalized keys,
and combined source fields. The adapted combined catalog is distributed under
CC BY-SA 4.0; the code's MIT license does not cover it. Credit VehiclesDB and
MotorcycleName, retain their linked terms and attribution, and identify these
transformations when redistributing the catalog.

Catalog columns are `brand,model,vehicle_type,aliases,source`. It contains
6,036 cars, 1,279 motorcycles, 181 scooters, and 7,246 records with unknown
vehicle type. Source coverage and classifications are incomplete and must not
be treated as an authoritative vehicle register.

## Privacy and limitations

- Keep claims, generated outputs, review files, and verified mappings out of Git.
- Catalog absence is not proof that a vehicle identity is invalid.
- Similar model families, short numeric names, trims, years, fuel descriptors,
  and multiple vehicle mentions may require review.
- Alias matches preserve the original alias text in `extra_info`; an alias does
  not create a new canonical identity.
- Tests use synthetic independent fixtures and do not establish real-world
  precision or production readiness.

Source code and tests are licensed under the [MIT License](LICENSE).
