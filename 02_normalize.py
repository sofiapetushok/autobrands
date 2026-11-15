"""Phase 2: loss-conscious text normalization of raw vehicle claims.

Run: python 02_normalize.py [input.csv] [output.csv]
This stage does not identify vehicles or consult a reference catalogue.
It adds loss-conscious Latin comparison columns without replacing normalized text.
"""

import argparse
import csv
import re
import tempfile
import unicodedata
from collections import Counter
from pathlib import Path


DIRECTORY = Path(__file__).resolve().parent
DEFAULT_INPUT = DIRECTORY / "data" / "claims_raw.csv"
DEFAULT_OUTPUT = DIRECTORY / "output" / "claims_normalized.csv"
INPUT_COLUMNS = ("claim_id", "brand", "model")
OUTPUT_COLUMNS = (
    "claim_id", "brand_raw", "model_raw", "brand_normalized",
    "model_normalized", "brand_latin", "model_latin",
    "brand_missing", "model_missing", "quality_flags",
)
FLAG_ORDER = (
    "BRAND_MISSING", "MODEL_MISSING", "BOTH_MISSING", "BRAND_EQUALS_MODEL",
    "NUMERIC_ONLY_BRAND", "NUMERIC_ONLY_MODEL", "MIXED_GREEK_LATIN",
    "UNUSUAL_SYMBOLS", "MULTIWORD_BRAND", "POSSIBLE_GARBAGE",
)
MISSING_TOKENS = frozenset({
    "", "NULL", "NONE", "N/A", "NA", "NAN", "UNKNOWN", "ΑΓΝΩΣΤΟ",
    "-", "--", "---", "???",
})
DASHES = dict.fromkeys(map(ord, "‐‑‒–—―−﹘﹣－"), "-")
QUOTES = dict.fromkeys(map(ord, "‘’‚‛`´"), "'")
QUOTES.update(dict.fromkeys(map(ord, "“”„‟"), '"'))
# These punctuation marks can occur in vehicle names or specifications.
COMMON_PUNCTUATION = frozenset("-./+&'(),:")


def strip_greek_diacritics(value: str) -> str:
    """Remove combining marks attached to Greek letters only."""
    result = []
    greek_base = False
    for char in unicodedata.normalize("NFD", value):
        if unicodedata.combining(char):
            if not greek_base:
                result.append(char)
        else:
            greek_base = unicodedata.name(char, "").startswith("GREEK ")
            result.append(char)
    return unicodedata.normalize("NFC", "".join(result))


def normalize_text(value: str | None) -> str | None:
    """Return comparable text, or None for an explicit missing placeholder."""
    if value is None:
        return None
    text = unicodedata.normalize("NFKC", value)
    text = " ".join(text.split())
    text = strip_greek_diacritics(text.casefold().upper())
    text = text.translate(DASHES).translate(QUOTES)
    return None if text in MISSING_TOKENS else text



# A generic phonetic comparison view, not a canonical name converter.
# Longer digraphs are handled before individual letters. Keep Y distinct from OU=U.
GREEK_DIGRAPHS = (
    ("ΟΥ", "U"), ("ΑΙ", "E"), ("ΕΙ", "I"), ("ΟΙ", "I"),
    ("ΥΙ", "I"), ("ΜΠ", "B"), ("ΝΤ", "D"), ("ΓΚ", "G"),
    ("ΓΓ", "NG"), ("ΤΣ", "TS"), ("ΤΖ", "TZ"),
    ("ΑΥ", "AV"), ("ΕΥ", "EV"),
)
GREEK_LETTERS = {
    "Α": "A", "Β": "V", "Γ": "G", "Δ": "D", "Ε": "E",
    "Ζ": "Z", "Η": "I", "Θ": "TH", "Ι": "I", "Κ": "K",
    "Λ": "L", "Μ": "M", "Ν": "N", "Ξ": "X", "Ο": "O",
    "Π": "P", "Ρ": "R", "Σ": "S", "Τ": "T", "Υ": "Y",
    "Φ": "F", "Χ": "CH", "Ψ": "PS", "Ω": "O",
}
# Only apply visual equivalences inside predominantly Latin words. Applying them
# to Greek words would confuse phonetic names and short model codes.
LATIN_DOMINANT_CONFUSABLES = {
    "Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z", "Η": "H",
    "Ι": "I", "Κ": "K", "Μ": "M", "Ν": "N", "Ο": "O",
    "Ρ": "P", "Τ": "T", "Υ": "Y", "Χ": "X",
}
WORD_TOKEN = re.compile(r"[^\W_]+|_+|[^\w]", flags=re.UNICODE)


def latin_comparison_text(value: str | None) -> str | None:
    """Make a deterministic, catalog-independent Latin comparison view.

    For example ΣΟΥΖΟΥΚΙ -> SUZUKI, whereas ΣΥΖΥΚΙ -> SYZYKI.
    Non-Greek text, numbers and punctuation survive unchanged. This is *not*
    a spelling corrector: model/brand identity belongs to the matcher.
    """
    if value is None:
        return None
    pieces = []
    for match in WORD_TOKEN.finditer(value):
        token = match.group()
        latin = sum(unicodedata.name(c, "").startswith("LATIN ") for c in token)
        greek = sum(unicodedata.name(c, "").startswith("GREEK ") for c in token)
        if latin >= 2 and latin > greek:
            pieces.append("".join(LATIN_DOMINANT_CONFUSABLES.get(c, GREEK_LETTERS.get(c, c)) for c in token))
            continue
        result = []
        i = 0
        while i < len(token):
            pair = token[i:i + 2]
            folded = next((replacement for source, replacement in GREEK_DIGRAPHS if pair == source), None)
            if folded is not None:
                result.append(folded)
                i += 2
            else:
                result.append(GREEK_LETTERS.get(token[i], token[i]))
                i += 1
        pieces.append("".join(result))
    return "".join(pieces)


def has_mixed_greek_latin(value: str) -> bool:
    greek = latin = False
    for char in value:
        if not char.isalpha():
            continue
        name = unicodedata.name(char, "")
        greek |= name.startswith("GREEK ")
        latin |= name.startswith("LATIN ")
    return greek and latin


def has_unusual_symbols(value: str) -> bool:
    return any(
        not (char.isalnum() or char.isspace() or char in COMMON_PUNCTUATION)
        for char in value
    )


def is_numeric_only(value: str) -> bool:
    """Digits with separators, and no letters or other symbols."""
    return any(char.isdecimal() for char in value) and all(
        char.isdecimal() or char.isspace() or char in COMMON_PUNCTUATION
        for char in value
    )


def is_possible_garbage(value: str) -> bool:
    """Flag only strong noise: no letters/digits or a repeated noise run."""
    return not any(char.isalnum() for char in value) or bool(
        re.search(r"([@#?*_\\])\1{2,}", value)
    )


def quality_flags(brand: str | None, model: str | None) -> tuple[str, ...]:
    """Describe observable text properties in a fixed, documented order."""
    fields = (value for value in (brand, model) if value is not None)
    combined = " ".join(fields)
    found = set()
    if brand is None:
        found.add("BRAND_MISSING")
    if model is None:
        found.add("MODEL_MISSING")
    if brand is None and model is None:
        found.add("BOTH_MISSING")
    if brand is not None and model is not None and brand == model:
        found.add("BRAND_EQUALS_MODEL")
    if brand is not None and is_numeric_only(brand):
        found.add("NUMERIC_ONLY_BRAND")
    if model is not None and is_numeric_only(model):
        found.add("NUMERIC_ONLY_MODEL")
    if has_mixed_greek_latin(combined):
        found.add("MIXED_GREEK_LATIN")
    if has_unusual_symbols(combined):
        found.add("UNUSUAL_SYMBOLS")
    if brand is not None and len(brand.split()) > 1:
        found.add("MULTIWORD_BRAND")
    if any(is_possible_garbage(value) for value in (brand, model) if value is not None):
        found.add("POSSIBLE_GARBAGE")
    return tuple(flag for flag in FLAG_ORDER if flag in found)


def normalize_row(row: dict[str, str]) -> dict[str, str]:
    brand = normalize_text(row["brand"])
    model = normalize_text(row["model"])
    return {
        "claim_id": row["claim_id"],
        "brand_raw": row["brand"],
        "model_raw": row["model"],
        "brand_normalized": brand or "",
        "model_normalized": model or "",
        "brand_latin": latin_comparison_text(brand) or "",
        "model_latin": latin_comparison_text(model) or "",
        "brand_missing": str(brand is None).lower(),
        "model_missing": str(model is None).lower(),
        "quality_flags": "|".join(quality_flags(brand, model)),
    }


def normalize_file(input_path: Path, output_path: Path) -> dict:
    """Stream rows to an atomic output file and return report statistics."""
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output paths must differ")
    if not input_path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    stats = {
        "rows": 0, "missing_brand": 0, "missing_model": 0,
        "changed_brand": 0, "changed_model": 0,
        "flags": Counter(), "raw_pairs": set(), "normalized_pairs": set(),
        "examples": [],
    }
    temporary_path = None
    try:
        with input_path.open("r", encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            missing_columns = set(INPUT_COLUMNS) - set(reader.fieldnames or ())
            if missing_columns:
                raise ValueError(f"Missing input columns: {', '.join(sorted(missing_columns))}")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", newline="", dir=output_path.parent,
                prefix=".claims_normalized_", suffix=".tmp", delete=False,
            ) as destination:
                temporary_path = Path(destination.name)
                writer = csv.DictWriter(
                    destination, fieldnames=OUTPUT_COLUMNS, lineterminator="\n",
                )
                writer.writeheader()
                for line_number, row in enumerate(reader, start=2):
                    if None in row or any(row[column] is None for column in INPUT_COLUMNS):
                        raise ValueError(f"Malformed CSV row at line {line_number}")
                    result = normalize_row(row)
                    writer.writerow(result)
                    stats["rows"] += 1
                    stats["missing_brand"] += result["brand_missing"] == "true"
                    stats["missing_model"] += result["model_missing"] == "true"
                    brand_changed = row["brand"] != result["brand_normalized"]
                    model_changed = row["model"] != result["model_normalized"]
                    stats["changed_brand"] += brand_changed
                    stats["changed_model"] += model_changed
                    stats["flags"].update(filter(None, result["quality_flags"].split("|")))
                    stats["raw_pairs"].add((row["brand"], row["model"]))
                    stats["normalized_pairs"].add(
                        (result["brand_normalized"], result["model_normalized"])
                    )
                    if (brand_changed or model_changed) and len(stats["examples"]) < 6:
                        stats["examples"].append((
                            row["brand"], row["model"],
                            result["brand_normalized"], result["model_normalized"],
                        ))
        temporary_path.replace(output_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return stats


def print_report(stats: dict, input_path: Path, output_path: Path) -> None:
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Processed records: {stats['rows']:,}")
    print(f"Missing brand: {stats['missing_brand']:,}")
    print(f"Missing model: {stats['missing_model']:,}")
    print(f"Changed brand values: {stats['changed_brand']:,}")
    print(f"Changed model values: {stats['changed_model']:,}")
    print(f"Unique raw brand/model pairs: {len(stats['raw_pairs']):,}")
    print(f"Unique normalized brand/model pairs: {len(stats['normalized_pairs']):,}")
    print("Most frequent quality flags:")
    for flag, count in stats["flags"].most_common(10):
        print(f"  {flag}: {count:,}")
    print("Examples (brand, model):")
    for raw_brand, raw_model, brand, model in stats["examples"]:
        print(f"  {raw_brand!r}, {raw_model!r} -> {brand!r}, {model!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("output", nargs="?", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        stats = normalize_file(args.input, args.output)
    except (OSError, UnicodeError, csv.Error, ValueError) as exc:
        parser.exit(1, f"Normalization failed: {exc}\n")
    print_report(stats, args.input, args.output)


if __name__ == "__main__":
    main()