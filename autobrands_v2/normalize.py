"""Minimal Unicode normalization for AutoBrands V2.

This module preserves Greek, Latin, mixed scripts, digits, and identity-bearing
punctuation. Latin comparison forms are separate matching aids; normalization
never transliterates the input.
"""

from __future__ import annotations

import unicodedata

MISSING_VALUES = frozenset({
    "", "NULL", "NONE", "N/A", "NA", "NAN", "UNKNOWN", "ΑΓΝΩΣΤΟ",
    "-", "--", "---", "???",
})

_DASHES = dict.fromkeys(map(ord, "‐‑‒–—―−﹘﹣－"), "-")
_SINGLE_QUOTES = dict.fromkeys(map(ord, "‘’‚‛`´"), "'")
_DOUBLE_QUOTES = dict.fromkeys(map(ord, "“”„‟"), '"')
_SINGLE_QUOTES.update(_DOUBLE_QUOTES)

# Longest digraphs first. Values are generic Greek-to-Latin phonetic renderings.
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
VISUAL_CONFUSABLES = {
    "Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z", "Η": "H",
    "Ι": "I", "Κ": "K", "Μ": "M", "Ν": "N", "Ο": "O",
    "Ρ": "P", "Τ": "T", "Υ": "Y", "Χ": "X",
}


def normalize_text(value):
    """Return NFC uppercase text, retaining Greek/Latin scripts and symbols."""
    if value is None:
        return None
    if not isinstance(value, str):
        if type(value).__name__ == "NAType":
            return None
        try:
            if value != value:
                return None
        except (TypeError, ValueError):
            pass
        raise TypeError("normalize_text expects a string or None")
    text = unicodedata.normalize("NFC", value)
    kept = []
    for char in text:
        category = unicodedata.category(char)
        if category == "Cf" or category in {"Cs", "Co", "Cn"}:
            continue
        if category == "Cc":
            if char.isspace():
                kept.append(" ")
            continue
        kept.append(char)
    text = "".join(kept).translate(_DASHES).translate(_SINGLE_QUOTES)
    return " ".join(text.upper().split())


def is_missing(value):
    text = normalize_text(value)
    return text is None or text in MISSING_VALUES


def is_garbage(value):
    """Flag placeholders and symbol-only values; numeric model codes are valid."""
    text = normalize_text(value)
    if text is None or text in MISSING_VALUES:
        return True
    return not any(char.isalpha() or char.isdecimal() for char in text)


def _phonetic(text):
    result = []
    i = 0
    while i < len(text):
        digraph = next(((source, target) for source, target in GREEK_DIGRAPHS
                        if text.startswith(source, i)), None)
        if digraph:
            result.append(digraph[1])
            i += len(digraph[0])
        else:
            result.append(GREEK_LETTERS.get(text[i], text[i]))
            i += 1
    return "".join(result)


def _visual(text):
    return "".join(VISUAL_CONFUSABLES.get(char, GREEK_LETTERS.get(char, char))
                   for char in text)


def _fold_diacritics(text):
    """Return a comparison-only form without combining accent marks."""
    decomposed = unicodedata.normalize("NFD", text)
    return unicodedata.normalize(
        "NFC", "".join(char for char in decomposed
                       if unicodedata.category(char) != "Mn")
    )


def comparison_forms(value):
    """Return bounded comparison forms; the normalized source remains untouched."""
    text = normalize_text(value)
    if not text:
        return ()
    folded = _fold_diacritics(text)
    greek = any(unicodedata.name(char, "").startswith("GREEK ")
                for char in folded)
    forms = [text]
    if folded != text:
        forms.append(folded)
    if greek:
        forms.extend((_phonetic(folded), _visual(folded)))
    return tuple(dict.fromkeys(form for form in forms if form))


def latin_fallback(value):
    """Create an uppercase Latin fallback for output only; matching keeps raw scripts."""
    text = normalize_text(value)
    if not text:
        return ""
    has_greek = any(unicodedata.name(char, "").startswith("GREEK ") for char in text)
    has_latin = any(unicodedata.name(char, "").startswith("LATIN ") for char in text)
    folded = _fold_diacritics(text)
    if has_greek and has_latin:
        return _visual(folded)
    if has_greek:
        return _phonetic(folded)
    return text


def matching_form(value):
    """Comparison-only key: ignore whitespace around punctuation, keep symbols."""
    text = normalize_text(value) or ""
    for punctuation in "+/-.&'(),:;":
        text = text.replace(f" {punctuation}", punctuation).replace(f"{punctuation} ", punctuation)
    return text
