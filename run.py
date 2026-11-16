"""CSV command-line interface for AutoBrands vehicle entity resolution.

Run `python run.py --help` for options.
"""
import argparse
import csv
import statistics
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

from matching import Matcher, norm, comparison_text


HERE = Path(__file__).resolve().parent
INPUT = HERE / "output" / "claims_normalized.csv"
CATALOG = HERE / "data" / "vehicle_catalog.csv"
BRAND_ALIASES = None
OUTPUT = HERE / "output" / "claims_matched.csv"
OUTPUT_COLUMNS = (
    "claim_id", "brand_raw", "model_raw", "brand_normalized", "model_normalized",
    "brand_missing", "model_missing",
    "corrected_brand", "corrected_model", "vehicle_type", "match_status",
    "match_method", "match_score", "confidence_level", "candidate_count",
    "top_candidates", "quality_flags", "brand_latin", "model_latin", "extra_info",
)

def read_csv(path: Path, required: set[str]) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path.name} missing columns: {', '.join(sorted(missing))}")
        rows = list(reader)
        if any(None in row or any(row[column] is None for column in required) for row in rows):
            raise ValueError(f"Malformed CSV row in {path.name}")
        return rows


def run(input_path: Path, catalog_path: Path, aliases_path: Path | None, output_path: Path,
        min_score: float, min_margin: float, min_brand: float, min_model: float) -> dict:
    paths = (input_path.resolve(), catalog_path.resolve())
    if aliases_path is not None:
        paths += (aliases_path.resolve(),)
    if output_path.resolve() in paths:
        raise ValueError("Output must differ from input, catalog, and alias paths")
    started = time.perf_counter()
    rows = read_csv(input_path, set(OUTPUT_COLUMNS[:7]) | {"quality_flags"})
    ids = [row["claim_id"] for row in rows]
    if len(set(ids)) != len(ids) or any(not value.strip() for value in ids):
        raise ValueError("Claim IDs must be nonempty and unique")
    catalog_rows = read_csv(catalog_path, {"brand", "model", "vehicle_type", "aliases", "source"})
    alias_rows = read_csv(aliases_path, {"brand", "alias"}) if aliases_path is not None else []
    matcher = Matcher(catalog_rows, alias_rows, min_score, min_margin, min_brand, min_model)
    cache = {}
    counts = Counter()
    scores = []
    examples = {}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=output_path.parent,
                                         prefix=".claims_matched_", suffix=".tmp", delete=False) as file:
            temporary = Path(file.name)
            writer = csv.DictWriter(file, fieldnames=OUTPUT_COLUMNS, lineterminator="\n")
            writer.writeheader()
            for row in rows:
                key = row["brand_normalized"], row["model_normalized"]
                if key not in cache:
                    cache[key] = matcher.resolve(*key, brand_latin=row.get("brand_latin"),
                                                 model_latin=row.get("model_latin"))
                decision = cache[key]
                result = {name: row[name] for name in OUTPUT_COLUMNS[:7]}
                result.update({
                    "corrected_brand": decision.corrected_brand,
                    "corrected_model": decision.corrected_model,
                    "vehicle_type": decision.vehicle_type,
                    "match_status": decision.match_status,
                    "match_method": decision.match_method,
                    "match_score": decision.match_score,
                    "confidence_level": decision.confidence_level,
                    "candidate_count": decision.candidate_count,
                    "top_candidates": matcher.serialize_candidates(decision),
                    "quality_flags": row["quality_flags"],
                    "brand_latin": comparison_text(norm(key[0]), row.get("brand_latin")),
                    "model_latin": comparison_text(norm(key[1]), row.get("model_latin")),
                    "extra_info": decision.extra_info,
                })
                writer.writerow(result)
                counts[decision.match_status] += 1
                if decision.match_score:
                    scores.append(float(decision.match_score))
                examples.setdefault(decision.match_status, (
                    row["brand_normalized"], row["model_normalized"],
                    decision.corrected_brand, decision.corrected_model,
                ))
        temporary.replace(output_path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    elapsed = time.perf_counter() - started
    return {"rows": len(rows), "pairs": len(cache), "counts": counts,
            "scores": scores, "seconds": elapsed, "examples": examples,
            "brand_aliases": len(matcher.brand_aliases)}


def print_report(report: dict) -> None:
    print(f"Claims: {report['rows']:,}; unique normalized pairs: {report['pairs']:,}")
    print("Statuses:")
    for status in ("EXACT_MATCH", "ALIAS_MATCH", "FUZZY_MATCH", "PARTIAL_MATCH",
                   "AMBIGUOUS", "CONFLICT", "UNRESOLVED"):
        print(f"  {status}: {report['counts'][status]:,}")
    accepted = sum(report["counts"][status] for status in
                   ("EXACT_MATCH", "ALIAS_MATCH", "FUZZY_MATCH"))
    print(f"Auto-acceptance: {accepted:,}/{report['rows']:,} ({accepted/max(report['rows'], 1)*100:.1f}%)")
    scores = report["scores"]
    if scores:
        bins = Counter("0-59" if s < 60 else "60-79" if s < 80 else
                       "80-89" if s < 90 else "90-94" if s < 95 else "95-100" for s in scores)
        print(f"Scores (similarity, not probability): mean={statistics.mean(scores):.1f}; "
              f"range={min(scores):.1f}-{max(scores):.1f}; bins={dict(sorted(bins.items()))}")
    print(f"Processing time: {report['seconds']:.2f} s")
    print("Representative examples (brand/model -> accepted brand/model):")
    for status, example in report["examples"].items():
        print(f"  {status}: {ascii(example[0])}/{ascii(example[1])} -> "
              f"{ascii(example[2])}/{ascii(example[3])}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--catalog", type=Path, default=CATALOG)
    parser.add_argument("--brand-aliases", type=Path, default=BRAND_ALIASES,
                        help="Optional independently verified brand-alias CSV")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--min-score", type=float, default=94)
    parser.add_argument("--min-margin", type=float, default=8)
    parser.add_argument("--min-brand", type=float, default=88)
    parser.add_argument("--min-model", type=float, default=88)
    args = parser.parse_args()
    try:
        report = run(args.input, args.catalog, args.brand_aliases, args.output,
                     args.min_score, args.min_margin, args.min_brand, args.min_model)
    except (OSError, ValueError, csv.Error) as exc:
        print(f"Matching failed: {exc}", file=sys.stderr)
        return 1
    print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())