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
from contextlib import nullcontext
from pathlib import Path

from autobrands.matching import Matcher, norm, comparison_text
from autobrands.llm import (LLM_COLUMNS, LLM_ROUTES, LLMRouter, LLMResolver,
                            LLMProvider, LLMDecisionCache)

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
        min_score: float, min_margin: float, min_brand: float, min_model: float, *,
        llm_dry_run: bool = False, llm_max_requests: int = 100,
        llm_provider: LLMProvider | None = None, llm_cache: LLMDecisionCache | None = None) -> dict:
    paths = (input_path.resolve(), catalog_path.resolve())
    if aliases_path is not None:
        paths += (aliases_path.resolve(),)
    if output_path.resolve() in paths:
        raise ValueError("Output must differ from input, catalog, and alias paths")
    if llm_dry_run and llm_provider is not None:
        raise ValueError("Dry-run cannot use a provider")
    if llm_cache is not None and llm_cache.path is not None and llm_cache.path.resolve() in paths + (output_path.resolve(),):
        raise ValueError("Cache must differ from source and output paths")
    started = time.perf_counter()
    rows = read_csv(input_path, set(OUTPUT_COLUMNS[:7]) | {"quality_flags"})
    ids = [row["claim_id"] for row in rows]
    if len(set(ids)) != len(ids) or any(not value.strip() for value in ids):
        raise ValueError("Claim IDs must be nonempty and unique")
    catalog_rows = read_csv(catalog_path, {"brand", "model", "vehicle_type", "aliases", "source"})
    alias_rows = read_csv(aliases_path, {"brand", "alias"}) if aliases_path is not None else []
    matcher = Matcher(catalog_rows, alias_rows, min_score, min_margin, min_brand, min_model)
    llm = (LLMResolver(LLMRouter(matcher), llm_provider, llm_cache, llm_max_requests)
           if llm_dry_run or llm_provider is not None else None)
    cache = {}
    counts = Counter()
    scores = []
    examples = {}
    # Normalize-equivalent raw forms share a call. Choose a stable representative
    # rather than letting row order decide which raw spelling the provider sees.
    representatives = {}
    if llm is not None:
        for row in rows:
            key = row["brand_normalized"], row["model_normalized"]
            prior = representatives.get(key)
            if prior is None or (row["brand_raw"], row["model_raw"]) < (prior["brand_raw"], prior["model_raw"]):
                representatives[key] = row
        for key in sorted(representatives):
            cache[key] = matcher.resolve(*key, brand_latin=representatives[key].get("brand_latin"),
                                         model_latin=representatives[key].get("model_latin"))
            llm.process(representatives[key], cache[key], dry_run=llm_dry_run)
    if not llm_dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        output_context = (nullcontext(None) if llm_dry_run else
                          tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=output_path.parent,
                                                      prefix=".claims_matched_", suffix=".tmp", delete=False))
        with output_context as file:
            writer = None
            if file is not None:
                temporary = Path(file.name)
                columns = OUTPUT_COLUMNS + (LLM_COLUMNS if llm is not None else ())
                writer = csv.DictWriter(file, fieldnames=columns, lineterminator="\n")
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
                if llm is not None:
                    outcome = llm.outcomes[(norm(key[0]), norm(key[1]))]
                    result.update(outcome.columns())
                if writer is not None:
                    writer.writerow(result)
                counts[decision.match_status] += 1
                if decision.match_score:
                    scores.append(float(decision.match_score))
                examples.setdefault(decision.match_status, (
                    row["brand_normalized"], row["model_normalized"],
                    decision.corrected_brand, decision.corrected_model,
                ))
        if not llm_dry_run:
            if llm is not None:
                llm.cache.save()
            temporary.replace(output_path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    elapsed = time.perf_counter() - started
    llm_report = None
    if llm is not None:
        pair_routes = Counter(o.route for o in llm.outcomes.values())
        claim_routes = Counter()
        for row in rows:
            claim_routes[llm.outcomes[(norm(row["brand_normalized"]), norm(row["model_normalized"]))].route] += 1
        llm.stats["duplicate_pairs_suppressed"] = len(rows) - len(cache)
        llm.stats["eligible_duplicate_requests_suppressed"] = claim_routes["LLM_ELIGIBLE"] - pair_routes["LLM_ELIGIBLE"]
        llm_report = {"dry_run": llm_dry_run, "stats": dict(llm.stats),
                      "claim_routes": dict(claim_routes), "pair_routes": dict(pair_routes)}
    return {"rows": len(rows), "pairs": len(cache), "counts": counts,
            "scores": scores, "seconds": elapsed, "examples": examples,
            "brand_aliases": len(matcher.brand_aliases), "llm": llm_report}


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
    if report.get("llm"):
        llm = report["llm"]
        stats = Counter(llm["stats"])
        print("LLM dry-run (no output written)" if llm["dry_run"] else "LLM assistance")
        print("Routing (claims / unique normalized pairs):")
        for route in LLM_ROUTES:
            print(f"  {route}: {llm['claim_routes'].get(route, 0):,} / {llm['pair_routes'].get(route, 0):,}")
        print(f"LLM-eligible unique pairs: {stats['eligible_unique_pairs']:,}")
        print(f"Proposed requests within budget: {stats['planned_requests']:,}; "
              f"budget-deferred pairs: {stats['budget_deferred']:,}; cache hits: {stats['cache_hits']:,}")
        print(f"Duplicate pair work suppressed: {stats['duplicate_pairs_suppressed']:,}; "
              f"duplicate eligible requests suppressed: {stats['eligible_duplicate_requests_suppressed']:,}")
        print(f"Input token approximation (UTF-8 bytes / 4, rounded up; includes prompt/schema): "
              f"all eligible={stats['estimated_input_tokens']:,}; within budget={stats['planned_input_tokens']:,}")
        print(f"Actual API attempts: {stats['api_attempts']:,}; invalid payloads: {stats['invalid_payloads']:,}")
        return  # No raw claim examples in LLM reports.
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
    parser.add_argument("--llm-dry-run", action="store_true",
                        help="Construct and validate requests; make no API calls and write no CSV")
    parser.add_argument("--llm-max-requests", type=int, default=100,
                        help="Maximum provider attempts per run, including retries (default 100)")
    args = parser.parse_args()
    try:
        report = run(args.input, args.catalog, args.brand_aliases, args.output,
                     args.min_score, args.min_margin, args.min_brand, args.min_model,
                     llm_dry_run=args.llm_dry_run, llm_max_requests=args.llm_max_requests)
    except (OSError, ValueError, csv.Error) as exc:
        print(f"Matching failed: {exc}", file=sys.stderr)
        return 1
    print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())