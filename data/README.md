# Vehicle reference data: provenance and licenses

`vehicle_catalog.csv` is a derived, combined reference catalog containing **14,742 rows**. It is **not** insurance claims data and contains no claim IDs or policyholder details.

| Upstream | Version | License | Source label in CSV |
| --- | --- | --- | --- |
| [VehiclesDB](https://github.com/vehiclesdb/vehiclesdb) | 2026.11.10 | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | `VehiclesDB v2026.11.10` |
| [MotorcycleName](https://github.com/ixiehao/MotorcycleName) | 1.1.1 | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) | `MotorcycleName v1.1.1` |

## What was modified

The catalog is a **transformation/compilation**, not a verbatim source export. Source vehicle names were selected and combined into `brand,model,vehicle_type,aliases,source`. Overlapping names were deduplicated; some alternative names were merged into `aliases`. The `source` column tracks contributing datasets; a pipe-separated value indicates both sources. This file does not reproduce all original upstream fields or source-register metadata.

## Attribution and redistribution

**Vehicle data by VehiclesDB:** https://vehiclesdb.com

**VehiclesDB upstream repository and source-specific attribution:**
- https://github.com/vehiclesdb/vehiclesdb
- https://github.com/vehiclesdb/vehiclesdb/blob/main/ATTRIBUTION.md
- https://github.com/vehiclesdb/vehiclesdb/blob/main/SOURCES.md

**MotorcycleName:** https://github.com/ixiehao/MotorcycleName

The two upstream datasets use different Creative Commons licenses. This merged and transformed **catalog data** is distributed under **CC BY-SA 4.0**, with required credit to both originals and notice of changes. The original VehiclesDB CC BY 4.0 notices, including any required notices for the underlying official data sources, continue to apply. Refer to the attribution file belonging to the corresponding upstream release for complete details.

The **Python source code** is separately MIT licensed. Neither the MIT license nor the code copyright notice changes the third-party dataset licenses.

Names and models in the catalog are for identity resolution, not validated manufacturer specifications. Do not treat its contents as exhaustive.
