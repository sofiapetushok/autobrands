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
`brand,model,vehicle_type,aliases`. `aliases` is a pipe-separated list of
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

Catalog columns are `brand,model,vehicle_type,aliases`. It contains
6,036 cars, 1,279 motorcycles, 181 scooters, and 7,246 records with unknown
vehicle type. Source coverage and classifications are incomplete and must not
be treated as an authoritative vehicle register.

### Greek localized alias evidence

The two catalog datasets and their licenses above are unchanged. The following
pages were consulted only to verify the manually added Greek model aliases and
their canonical identities; they are not additional bulk dataset sources.
Where a nickname applies to several catalog identities, it is repeated on the
supported family rows so nickname-only input remains a review case. Latin
spellings are evaluated through the existing Greek-to-Latin comparison channel
instead of being stored as duplicate aliases.

| Publisher | Consulted source and use | Licensing / attribution |
| --- | --- | --- |
| Top Gear Greece / Attica Media | [Greek vehicle nicknames](https://www.topgeargreece.gr/afieroma/1020633/ta-pio-dimofili-paratsouklia-se-autokinita/) documents `500αράκι` for Fiat 500, `Σκαραβαίος`, `Σκαθάρι`, and `Χελώνα` for Volkswagen Beetle, `Βάτραχος` for Citroën DS, and the brand-level `Μπέμπα` for BMW. | The publisher's [terms](https://www.topgeargreece.gr/terms-of-use/) reserve copyright and restrict reproduction. The project retains a citation only; no article text, images, or other media are redistributed. |
| Athens Voice / AutoTriti | The [Greek 2CV club feature](https://www.athensvoice.gr/life/life-in-athens/363208/episkeftikame-tin-elliniki-leshi-filon/) identifies Citroën 2CV as `Ντεσεβό`; [AutoTriti's 2CV history](https://www.autotriti.gr/auto/agora/citroen-2cv-ds-cx-gs-pos-zoun-simera-mesa-sti-gkama-tis-citroen_10442) documents `Ασχημόπαπο`. | Athens Voice's [terms](https://www.athensvoice.gr/oroi-hrisis/) protect its content. No open-content license was relied on for AutoTriti. Both pages are linked factual evidence only; no article text or media are redistributed. |
| Drive / 4Troxoi | Drive's [Renault 4 history](https://www.drive.gr/test-drives/dokimi/test-drive-renault-4-e-tech-apo-ta-sixties-sto-thymiko-mas) records `Κατρέλ` and `Κατρουλή` for the historic Renault 4/4L; [4Troxoi](https://www.4troxoi.gr/lounge/giati-irthe-i-ora-na-epistrepsei-to-2cv/) independently uses `Κατρέλ` for Renault 4. | No open-content license was relied on. The pages are cited for verification; their text and media are not redistributed. |
| Naftemporiki / Gocar | A [Naftemporiki classic-car feature](https://services.naftemporiki.gr/story/1571303/1571303) identifies the classic Fiat 500 as `Πεντακοσαράκι`; [Gocar](https://www.gocar.gr/used-models/suggestions/24681%2C5_metaxeirismena_FIAT_500_apo_5750_eyrw.html) also documents the name for Fiat 500. | No open-content license was relied on. The pages are cited for verification only; their text and media are not redistributed. |
| ERT News / Euronews | [ERT News](https://www.ertnews.gr/ert3/thessaloniki/to-emvlimatiko-katsaridaki-erxetai-stin-plateia-aristotelous-apo-ti-lesxi-skaravaiou-thessalonikis/?noamp=mobile) and [Euronews](https://gr.euronews.com/2018/09/15/volkswagen-beetle-kafer-telos-stin-paragogi-tou) identify `Κατσαριδάκι` with the classic Volkswagen Beetle. | No open-content license was relied on. The pages are cited for verification only; their text and media are not redistributed. |
| MotoMag / MotoTriti / Gocar / Car and Motor / 4Troxoi | Review-only evidence for [Honda Cub category names](https://www.motomag.gr/news/epikairotita/honda-cub-papi-thri-nti), [Honda Z50 family nicknames](https://www.mototriti.gr/data/news/preview_news/honda-z50-monkey-o-%C2%ABmpompos%C2%BB-ths-dekaetias-toy-90_164713.asp), [Greek Army 290 GD usage](https://www.gocar.gr/news/new-models/50105%2CH_Mercedes_Benz_G_Class_xanaginetai_kana.html), [Greek Navy Type 181 usage](https://www.carandmotor.gr/nea/vw-typ-181-toy-polemikoy-naytikoy-poleitai), and [Attica 200](https://www.4troxoi.gr/4t-classic/attica-200-i-aplotita-sta-ellinika/) was used to reject category-wide, fleet-specific, or catalog-absent mappings. | No open-content license was relied on. These pages are cited only to document review decisions; no text or media are redistributed. |
| Citroën Greece / Automobiles Citroën | [Manufacturer history](https://www.citroen.gr/about-citroen/history.html) confirms 2CV and DS as distinct historical Citroën identities. | Citroën's [legal notice](https://www.citroen.gr/tools/legal-information.html) reserves rights in site content, vehicle names, designs, and marks. It is used only as a linked identity check; no site content is copied. |
| Wikidata contributors / Wikimedia Foundation | Structured items for [Volkswagen Beetle](https://www.wikidata.org/wiki/Q152946), [Citroën 2CV](https://www.wikidata.org/wiki/Q326015), [Citroën DS](https://www.wikidata.org/wiki/Q465966), and [classic Fiat 500](https://www.wikidata.org/wiki/Q467767) were used as secondary model-identity and generation checks. | Wikidata [structured data is CC0](https://www.wikidata.org/wiki/Wikidata:Licensing), with no attribution requirement; links are retained for verification. Text outside the structured-data namespaces remains CC BY-SA 4.0 and was not copied. |

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
