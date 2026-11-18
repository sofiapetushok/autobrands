"""Unicode-preserving normalization and bounded comparison representations."""
from __future__ import annotations
import unicodedata

MISSING_VALUES = frozenset({"", "NULL", "NONE", "N/A", "NA", "NAN", "UNKNOWN", "ΑΓΝΩΣΤΟ", "-", "--", "---", "???"})
_DASHES = dict.fromkeys(map(ord, "‐‑‒–—―−﹘﹣－"), "-")
_QUOTES = dict.fromkeys(map(ord, "‘’‚‛`´"), "'")
_QUOTES.update(dict.fromkeys(map(ord, "“”„‟"), '"'))
GREEK_DIGRAPHS = (("ΟΥ", "U"), ("ΑΙ", "E"), ("ΕΙ", "I"), ("ΟΙ", "I"), ("ΥΙ", "I"), ("ΜΠ", "B"), ("ΝΤ", "D"), ("ΓΚ", "G"), ("ΓΓ", "NG"), ("ΤΣ", "TS"), ("ΤΖ", "TZ"), ("ΑΥ", "AV"), ("ΕΥ", "EV"))
GREEK_LETTERS = {"Α": "A", "Β": "V", "Γ": "G", "Δ": "D", "Ε": "E", "Ζ": "Z", "Η": "I", "Θ": "TH", "Ι": "I", "Κ": "K", "Λ": "L", "Μ": "M", "Ν": "N", "Ξ": "X", "Ο": "O", "Π": "P", "Ρ": "R", "Σ": "S", "Τ": "T", "Υ": "Y", "Φ": "F", "Χ": "CH", "Ψ": "PS", "Ω": "O"}
VISUAL_CONFUSABLES = {"Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z", "Η": "H", "Ι": "I", "Κ": "K", "Μ": "M", "Ν": "N", "Ο": "O", "Ρ": "P", "Τ": "T", "Υ": "Y", "Χ": "X"}


def normalize_text(value):
    """Uppercase NFC with whitespace/punctuation cleanup, retaining source scripts."""
    if value is None or type(value).__name__ == "NAType":
        return None
    if not isinstance(value, str):
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
        if category in {"Cf", "Cs", "Co", "Cn"}:
            continue
        if category == "Cc":
            if char.isspace():
                kept.append(" ")
            continue
        kept.append(char)
    return " ".join("".join(kept).translate(_DASHES).translate(_QUOTES).upper().split())


def is_missing(value):
    text = normalize_text(value)
    return text is None or text in MISSING_VALUES


def is_garbage(value):
    text = normalize_text(value)
    return text is None or text in MISSING_VALUES or not any(c.isalpha() or c.isdecimal() for c in text)


def _fold_diacritics(text):
    return unicodedata.normalize("NFC", "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"))


def _phonetic(text):
    out, i = [], 0
    while i < len(text):
        matched = next(((source, target) for source, target in GREEK_DIGRAPHS if text.startswith(source, i)), None)
        if matched:
            out.append(matched[1]); i += len(matched[0])
        else:
            out.append(GREEK_LETTERS.get(text[i], text[i])); i += 1
    return "".join(out)


def _visual(text):
    """Only visual substitutions; leave other Greek characters unchanged."""
    return "".join(VISUAL_CONFUSABLES.get(c, c) for c in text)


def comparison_forms(value):
    """Original, accent-folded, visual and phonetic views; original is never replaced."""
    text = normalize_text(value)
    if not text:
        return ()
    folded = _fold_diacritics(text)
    forms = [text, folded]
    if any(unicodedata.name(c, "").startswith("GREEK ") for c in folded):
        forms.extend((_visual(folded), _phonetic(folded)))
    return tuple(dict.fromkeys(forms))


def latin_fallback(value):
    """Legacy-compatible output fallback, not used as matching input."""
    text = normalize_text(value)
    if not text:
        return ""
    has_greek = any(unicodedata.name(c, "").startswith("GREEK ") for c in text)
    has_latin = any(unicodedata.name(c, "").startswith("LATIN ") for c in text)
    folded = _fold_diacritics(text)
    if has_greek and has_latin:
        # Visual where possible, phonetic for unsupported Greek glyphs.
        return "".join(VISUAL_CONFUSABLES.get(c, GREEK_LETTERS.get(c, c)) for c in folded)
    return _phonetic(folded) if has_greek else text


def matching_form(value):
    text = normalize_text(value) or ""
    for char in "+/-.&'(),:;":
        text = text.replace(f" {char}", char).replace(f"{char} ", char)
    return text
