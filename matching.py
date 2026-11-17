"""Catalog-driven vehicle identity matching for normalized records and DataFrames.

Use Matcher.from_csv() once, then Matcher.match_dataframe() on the output of
normalize.normalize_dataframe(). The existing normalized-CSV CLI also works.
No human-confirmation state or learning from fuzzy predictions is performed.
Similarity scores are not calibrated probabilities.
"""
import argparse
import csv
import statistics
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import json
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, replace
from functools import lru_cache

from rapidfuzz import fuzz, process
from rapidfuzz.distance import DamerauLevenshtein

import normalize

# Matching configuration: change these defaults in one place.
# Explicit Matcher(...) / from_csv(...) arguments and CLI flags override them.
DEFAULT_MIN_SCORE = 80.0
DEFAULT_MIN_MARGIN = 8.0
DEFAULT_MIN_BRAND = 80.0
DEFAULT_MIN_MODEL = 80.0
DEFAULT_TOP_K = 5
CATALOG_REQUIRED_FIELDS = {"brand", "model", "vehicle_type"}


def norm(value: str | None) -> str:
    return normalize.normalize_text(value) or ""


def compact(value: str) -> str:
    """Comparison-only key; it never changes canonical or claim text."""
    return "".join(char for char in value if char.isalnum())


def name_tokens(value: str) -> tuple[str, ...]:
    """Tokens for catalog phrase evidence; punctuation only marks boundaries."""
    return tuple(re.findall(r"[^\W_]+(?:[-.][^\W_]+)*", value))


def token_signature(value: str) -> tuple[str, ...]:
    return tuple(sorted(latin_fold(token) for token in name_tokens(value)))


# Descriptions, not vehicle aliases. Catalog name tokens always take precedence.
SUPPLEMENTARY_TOKENS = frozenset({"ΙΧ", "ΒΕΝΖΙΝΗ", "ΠΕΤΡΕΛΑΙΟ", "PETROL", "GASOLINE", "DIESEL"})


def latin_fold(value: str) -> str:
    """Fold the shared normalizer's Latin representation for comparisons."""
    value = normalize.normalize_text(value) or ""
    value = "".join(c for c in unicodedata.normalize("NFKD", value)
                    if not unicodedata.combining(c))
    return compact(value)


def comparison_text(value: str, supplied: str | None = None) -> str:
    alternatives = normalize.comparison_texts(value)
    expected = alternatives[0] if alternatives else ""
    if supplied is not None and supplied not in alternatives:
        raise ValueError("Latin comparison column disagrees with normalized text")
    return supplied if supplied is not None else expected


def phonetic_fold(value: str) -> str:
    return value.replace("W", "V").replace("Y", "U")


def brand_similarity(query: str, choice: str) -> float:
    left, right = latin_fold(query), latin_fold(choice)
    if not left or not right:
        return 0.0
    # W/V and U/Y are common cross-script phonetic alternatives. Compare both
    # forms, but retain edit distance and an independent model requirement.
    return max(fuzz.ratio(left, right),
               fuzz.ratio(phonetic_fold(left), phonetic_fold(right)))

# Cost units are normalized by the longer string length. Unicode equivalence
# changes candidate evidence only; existing acceptance thresholds remain in force.
UNICODE_MATCH_COSTS = {
    "exact": 0.0,
    "visual_mixed": 0.01,
    "visual_greek": 0.04,
    "phonetic_mixed": 0.35,
    "phonetic_greek": 0.02,
    "edit": 1.0,
}

def _has_greek(value: str | None) -> bool:
    return bool(value) and any(
        unicodedata.name(char, "").startswith("GREEK ") for char in value
    )

@lru_cache(maxsize=32768)
def unicode_weighted_similarity(query: str, choice: str) -> float:
    """Compare original Unicode text to a Latin catalog form with weighted edits.

    Greek letters and configured digraphs are traversed as DP edges, so visual
    and phonetic readings are compared without expanding strings combinatorially.
    Punctuation, spaces, and numbers remain ordinary identity-bearing characters.
    """
    source = normalize._normalize_format(query)
    target = normalize._normalize_format(choice)
    if not source or not target or not _has_greek(source):
        return 0.0
    latin_count = sum(unicodedata.name(ch, "").startswith("LATIN ") for ch in source)
    greek_count = sum(unicodedata.name(ch, "").startswith("GREEK ") for ch in source)
    mixed = bool(latin_count and greek_count)
    visual_cost = UNICODE_MATCH_COSTS["visual_mixed" if mixed else "visual_greek"]
    phonetic_cost = UNICODE_MATCH_COSTS["phonetic_mixed" if mixed else "phonetic_greek"]
    n, m = len(source), len(target)
    dp = [[float("inf")] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0
    for i in range(n + 1):
        for j in range(m + 1):
            current = dp[i][j]
            if current == float("inf"):
                continue
            if i < n:
                dp[i + 1][j] = min(dp[i + 1][j], current + UNICODE_MATCH_COSTS["edit"])
            if j < m:
                dp[i][j + 1] = min(dp[i][j + 1], current + UNICODE_MATCH_COSTS["edit"])
            if i < n and j < m:
                substitution = UNICODE_MATCH_COSTS["exact"] if source[i] == target[j] else UNICODE_MATCH_COSTS["edit"]
                dp[i + 1][j + 1] = min(dp[i + 1][j + 1], current + substitution)
            if i >= n:
                continue

            # A Greek character can emit both its visual and phonetic forms.
            char = source[i]
            if unicodedata.name(char, "").startswith("GREEK "):
                visual = normalize.LATIN_DOMINANT_CONFUSABLES.get(char)
                phonetic = normalize.GREEK_LETTERS.get(char)
                for rendered, cost in ((visual, visual_cost), (phonetic, phonetic_cost)):
                    if rendered and target.startswith(rendered, j):
                        dp[i + 1][j + len(rendered)] = min(
                            dp[i + 1][j + len(rendered)], current + cost
                        )

            # Digraphs are one-to-many or many-to-one weighted transitions.
            for greek, rendered in normalize.GREEK_DIGRAPHS:
                if source.startswith(greek, i) and target.startswith(rendered, j):
                    dp[i + len(greek)][j + len(rendered)] = min(
                        dp[i + len(greek)][j + len(rendered)], current + phonetic_cost
                    )
    distance = dp[n][m]
    return round(max(0.0, 100.0 * (1.0 - distance / max(n, m))), 1)


def numeric_signature(value: str) -> tuple[str, ...]:
    return tuple(re.findall(r"\d+", value))


def punctuation_signature(value: str) -> tuple[str, ...]:
    """Identity-bearing separators; whitespace differences do not affect them."""
    return tuple(char for char in value if char in "+/")


def model_family_risk(query: str, canonical: str) -> bool:
    """Avoid accepting a different number or added/removed model token."""
    if numeric_signature(query) != numeric_signature(canonical):
        return True
    left, right = compact(query), compact(canonical)
    return left != right and (left in right or right in left)


def informative(value: str) -> bool:
    if len(compact(value)) < 3:
        return False
    if re.fullmatch(r"-0+X+", value):
        return False
    if normalize.is_possible_garbage(value):
        return False
    return True


@dataclass(frozen=True)
class Identity:
    brand: str
    model: str
    brand_key: str
    model_key: str
    vehicle_type: str
    variants: tuple[str, ...]


@dataclass(frozen=True)
class Ranked:
    identity_id: int
    score: float
    brand_score: float
    model_score: float
    evidence: str


@dataclass(frozen=True)
class Decision:
    corrected_brand: str = ""
    corrected_model: str = ""
    vehicle_type: str = ""
    match_status: str = "UNRESOLVED"
    match_method: str = "NONE"
    match_score: str = ""
    confidence_level: str = "NONE"
    candidate_count: int = 0
    top_candidates: tuple[Ranked, ...] = ()
    extra_info: str = ""  # Reported descriptor, NOT a verified vehicle specification.


class Matcher:
    def __init__(self, catalog_rows: list[dict], brand_alias_rows: list[dict] = None,
                 min_score: float = DEFAULT_MIN_SCORE, min_margin: float = DEFAULT_MIN_MARGIN,
                 min_brand: float = DEFAULT_MIN_BRAND, min_model: float = DEFAULT_MIN_MODEL,
                 top_k: int = DEFAULT_TOP_K):
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        self.top_k = top_k
        self._unicode_candidate_cache = {}
        self.min_score = min_score
        self.min_margin = min_margin
        self.min_brand = min_brand
        self.min_model = min_model
        grouped = defaultdict(lambda: {
            "rows": [], "variants": set(), "types": set(), "comparison_variants": set()
        })
        self.brands = defaultdict(set)
        for row in catalog_rows:
            brand, model = row["brand"], row["model"]
            brand_key, model_key = norm(brand), norm(model)
            if not brand_key or not model_key:
                continue
            key = brand_key, model_key
            grouped[key]["rows"].append(row)
            grouped[key]["types"].add(row["vehicle_type"])
            grouped[key]["variants"].add(model_key)
            grouped[key]["comparison_variants"].update(
                norm(value) for value in normalize.comparison_texts(model) if norm(value)
            )
            # Current catalogs keep brand and model aliases in separate fields.
            # Accept the former `aliases` column as model aliases for old files.
            model_aliases = row.get("model_aliases") or ""
            legacy_model_aliases = row.get("aliases") or ""
            for alias in (model_aliases + "|" + legacy_model_aliases).split("|"):
                alias_key = norm(alias)
                if alias_key:
                    grouped[key]["variants"].add(alias_key)
                grouped[key]["comparison_variants"].update(
                    norm(value) for value in normalize.comparison_texts(alias) if norm(value)
                )
            self.brands[brand_key].add(brand)

        self.identities = []
        self.by_pair = {}
        self.by_brand = defaultdict(set)
        self.model_lookup = defaultdict(set)
        self.alias_lookup = defaultdict(set)
        self.full_lookup = defaultdict(set)
        self.compact_model_lookup = defaultdict(set)
        self.compact_full_lookup = defaultdict(set)
        self.token_full_lookup = defaultdict(set)
        self.folded_model_lookup = defaultdict(set)
        self.greek_alias_folded_lookup = defaultdict(set)
        self.model_tokens = set()
        self.brand_choices = {}
        self.model_choices_by_brand = defaultdict(lambda: defaultdict(set))
        self.model_choices_global = defaultdict(set)
        self.model_comparison_variants = {}
        for key in sorted(grouped):
            item = grouped[key]
            preferred = sorted(item["rows"], key=lambda r: (r["brand"], r["model"]))[0]
            types = item["types"]
            identity = Identity(
                brand=preferred["brand"], model=preferred["model"],
                brand_key=key[0], model_key=key[1],
                vehicle_type=next(iter(types)) if len(types) == 1 else "unknown",
                variants=tuple(sorted(item["variants"])),
            )
            identity_id = len(self.identities)
            self.identities.append(identity)
            comparison_variants = tuple(sorted(item["comparison_variants"]))
            self.model_comparison_variants[identity_id] = comparison_variants
            self.by_pair[key] = identity_id
            self.by_brand[key[0]].add(identity_id)
            self.model_lookup[key[1]].add(identity_id)
            self.full_lookup[f"{key[0]} {key[1]}"].add(identity_id)
            self.full_lookup[f"{key[1]} {key[0]}"].add(identity_id)
            self.compact_full_lookup[latin_fold(key[0] + key[1])].add(identity_id)
            self.compact_full_lookup[latin_fold(key[1] + key[0])].add(identity_id)
            self.token_full_lookup[token_signature(key[0] + " " + key[1])].add(identity_id)
            for variant in identity.variants:
                self.model_choices_by_brand[key[0]][variant].add(identity_id)
                self.model_choices_global[variant].add(identity_id)
                self.compact_model_lookup[compact(variant)].add(identity_id)
                self.folded_model_lookup[latin_fold(variant)].add(identity_id)
                self.folded_model_lookup[latin_fold(normalize.latin_comparison_text(variant) or "")].add(identity_id)
                if (variant != key[1] and
                        any("GREEK" in unicodedata.name(char, "") for char in variant)):
                    alias_folded = latin_fold(normalize.latin_comparison_text(variant) or variant)
                    self.greek_alias_folded_lookup[alias_folded].add(identity_id)
                self.model_tokens.update(latin_fold(token) for token in name_tokens(variant))
                if variant != key[1]:
                    self.alias_lookup[variant].add(identity_id)
            for variant in comparison_variants:
                self.model_choices_by_brand[key[0]][variant].add(identity_id)
                self.model_choices_global[variant].add(identity_id)
                self.folded_model_lookup[latin_fold(variant)].add(identity_id)

        self.brand_choices = {key: sorted(names)[0] for key, names in self.brands.items()}
        self.brand_aliases = self._load_brand_aliases(
            catalog_rows + (brand_alias_rows or [])
        )        
        self.brand_names_longest = sorted(self.brand_choices, key=lambda x: (-len(x), x))
        self.brand_folded = defaultdict(set)
        self.brand_phonetic = defaultdict(set)
        for brand in self.brand_choices:
            folded = latin_fold(brand)
            self.brand_folded[folded].add(brand)
            self.brand_phonetic[phonetic_fold(folded)].add(brand)
        self.brand_folded_strings = sorted(self.brand_folded)
        self.brand_phonetic_strings = sorted(self.brand_phonetic)
        self.global_model_strings = sorted(self.model_choices_global)
        self.brand_model_strings = {
            brand: sorted(variants) for brand, variants in self.model_choices_by_brand.items()
        }
        self.catalog_tokens = self.model_tokens | {
            latin_fold(t) for brand in self.brands for t in name_tokens(brand)
        }

    def _load_brand_aliases(self, rows: list[dict]) -> dict[str, str]:
        """Load brand aliases from the catalog, rejecting collisions."""
        targets = defaultdict(set)

        for row in rows:
            brand_key = norm(row.get("brand", ""))

            if brand_key not in self.brands:
                continue

            aliases = row.get("brand_aliases") or ""

            if row.get("alias"):
                aliases += "|" + row["alias"]

            for alias in aliases.split("|"):
                alias_key = norm(alias)

                if not alias_key or alias_key == brand_key:
                    continue

                targets[alias_key].add(brand_key)

        # An alias must not identify multiple brands or override
        # another existing canonical brand.
        return {
            alias: next(iter(brands))
            for alias, brands in targets.items()
            if len(brands) == 1 and alias not in self.brands
        }
    
    def brand_key(self, value: str) -> tuple[str, str]:
        value = norm(value)

        if value in self.brands:
            return value, ""

        target = self.brand_aliases.get(value)
        if target:
            return target, "VERIFIED_ALIAS" 

        folded = self.brand_folded.get(latin_fold(value), set())

        if len(compact(value)) >= 4 and len(folded) == 1:
            return next(iter(folded)), "FOLDED"

        return "", ""

    def exact_model_ids(self, value: str, brand_key: str = "") -> set[int]:
        ids = self.model_lookup.get(value, set()) | self.alias_lookup.get(value, set())
        return {i for i in ids if self.identities[i].brand_key == brand_key} if brand_key else set(ids)

    def exact_method(self, identity_id: int, brand_source: str, model_value: str,
                     cross_field: bool = False) -> str:
        identity = self.identities[identity_id]
        if brand_source == "VERIFIED_ALIAS" or model_value != identity.model_key:
            return "CROSS_FIELD_ALIAS" if cross_field else "EXPLICIT_ALIAS"
        if brand_source == "FOLDED":
            return "CROSS_FIELD_FOLDED" if cross_field else "FOLDED_BRAND_EXACT"
        return "CROSS_FIELD_EXACT" if cross_field else "CANONICAL_EXACT"

    def accepted(self, identity_id: int, status: str, method: str,
                 score: float = 100, candidates: tuple[Ranked, ...] = (),
                 extra_info: str = "") -> Decision:
        identity = self.identities[identity_id]
        if not candidates:
            candidates = (Ranked(identity_id, score, 100, 100, method.lower()),)
        if (identity.model_key.isdecimal() and len(identity.model_key) <= 3
                and identity.vehicle_type == "unknown"):
            return self.suggestion("AMBIGUOUS", "GENERIC_NUMERIC_MODEL", list(candidates))
        return Decision(
            corrected_brand=identity.brand, corrected_model=identity.model,
            vehicle_type=identity.vehicle_type, match_status=status,
            match_method=method, match_score=f"{score:.1f}",
            confidence_level="HIGH" if status != "FUZZY_MATCH" else "MEDIUM",
            candidate_count=len(candidates), top_candidates=tuple(candidates[:self.top_k]),
            extra_info=extra_info,
        )

    def suggestion(self, status: str, method: str, candidates: list[Ranked]) -> Decision:
        score = f"{candidates[0].score:.1f}" if candidates else ""
        return Decision(match_status=status, match_method=method, match_score=score,
                        confidence_level="LOW" if candidates else "NONE",
                        candidate_count=len(candidates), top_candidates=tuple(candidates[:self.top_k]))

    def _exact(self, brand: str, model: str) -> Decision | None:
        brand_key, brand_alias = self.brand_key(brand)
        if brand_key and model:
            mixed_ids = (self.folded_model_lookup.get(latin_fold(model), set())
                         if not model.isascii() and any(normalize.has_mixed_greek_latin(t)
                                                        for t in name_tokens(model)) else set())
            ids = self.exact_model_ids(model, brand_key)
            if len(ids) == 1:
                identity_id = next(iter(ids))
                identity = self.identities[identity_id]
                # Short numeric register buckets are not specific nameplates.
                if model.isdecimal() and len(model) <= 3 and identity.vehicle_type == "unknown":
                    return self.suggestion("AMBIGUOUS", "GENERIC_NUMERIC_MODEL",
                                           [Ranked(identity_id, 100, 100, 100, "numeric_bucket")])
                method = self.exact_method(identity_id, brand_alias, model)
                status = "ALIAS_MATCH" if "ALIAS" in method else "EXACT_MATCH"
                return self.accepted(identity_id, status, method)
            if len(ids) > 1:
                return self.suggestion("AMBIGUOUS", "EXACT_NAME_COLLISION",
                                       self._fixed_rank(ids, "exact_name_collision"))
            folded_alias_ids = (self.greek_alias_folded_lookup.get(
                latin_fold(normalize.latin_comparison_text(model) or model), set()) &
                self.by_brand[brand_key])
            if len(folded_alias_ids) == 1:
                return self.accepted(next(iter(folded_alias_ids)), "ALIAS_MATCH",
                                     "FOLDED_EXPLICIT_ALIAS")
            if len(folded_alias_ids) > 1:
                return self.suggestion(
                    "AMBIGUOUS", "FOLDED_ALIAS_COLLISION",
                    self._fixed_rank(folded_alias_ids, "folded_alias_collision"))
            if len(compact(model)) >= 4:
                compact_ids = self.compact_model_lookup.get(compact(model), set())
                local_compact = (compact_ids | mixed_ids) & self.by_brand[brand_key]
                if len(local_compact) == 1:
                    identity_id = next(iter(local_compact))
                    identity = self.identities[identity_id]
                    if ("+" in punctuation_signature(model)
                            and not any(punctuation_signature(variant) == punctuation_signature(model)
                                        for variant in identity.variants)):
                        return Decision(
                            corrected_brand=identity.brand,
                            match_status="PARTIAL_MATCH",
                            match_method="MODEL_PUNCTUATION_UNVERIFIED",
                            match_score="97.0", confidence_level="MEDIUM",
                            candidate_count=1,
                            top_candidates=(Ranked(identity_id, 97, 100, 97,
                                                   "unverified_model_punctuation"),),
                        )
                    method = "CATALOG_COMPACT_MODEL" if identity_id in compact_ids else "CATALOG_FOLDED_MODEL"
                    return self.accepted(identity_id, "FUZZY_MATCH", method, 97)
                if len(local_compact) > 1:
                    # A compact key may merge distinct identities (Golf/Golf+).
                    # Use observed identity-bearing punctuation to separate them,
                    # without requiring identical whitespace around punctuation.
                    same_punctuation = {
                        candidate for candidate in local_compact
                        if any(punctuation_signature(variant) == punctuation_signature(model)
                               for variant in self.identities[candidate].variants)
                    }
                    canonical_match = {
                        candidate for candidate in same_punctuation
                        if compact(self.identities[candidate].model_key) == compact(model)
                        and punctuation_signature(self.identities[candidate].model_key)
                        == punctuation_signature(model)
                    }
                    if len(canonical_match) == 1:
                        winner = next(iter(canonical_match))
                        return self.accepted(winner, "FUZZY_MATCH",
                                             "CANONICAL_PUNCTUATION_MATCH", 97)
                    if len(same_punctuation) == 1:
                        winner = next(iter(same_punctuation))
                        return self.accepted(winner, "FUZZY_MATCH", "CATALOG_PUNCTUATION_SPACING", 97)
                    if not same_punctuation and "+" in punctuation_signature(model):
                        return Decision(
                            corrected_brand=self.brand_choices[brand_key],
                            match_status="PARTIAL_MATCH",
                            match_method="MODEL_PUNCTUATION_UNVERIFIED",
                            match_score="97.0", confidence_level="MEDIUM",
                            candidate_count=len(local_compact),
                            top_candidates=tuple(self._fixed_rank(
                                local_compact, "unverified_model_punctuation")[:self.top_k]),
                        )
                    return self.suggestion("AMBIGUOUS", "COMPACT_NAME_COLLISION",
                                           self._fixed_rank(local_compact, "compact_name_collision"))
            foreign = self.exact_model_ids(model) - self.by_brand[brand_key]
            # Separator changes are safe to compare, including for conflicts:
            # CBR 600 RR and CBR600RR carry the same numeric identifier.
            if len(compact(model)) >= 4:
                foreign |= (self.compact_model_lookup.get(compact(model), set()) | mixed_ids) - self.by_brand[brand_key]
            # Short letter/number designations (for example one letter plus a
            # number) recur across makes; they are weak contradiction evidence.
            strong_foreign_name = (len(compact(model)) >= 4
                                   and sum(char.isalpha() for char in compact(model)) >= 3
                                   and not model.isdecimal())
            unrelated = {i for i in foreign if not (
                self.identities[i].brand_key.startswith(brand_key + " ") or
                brand_key.startswith(self.identities[i].brand_key + " "))}
            if strong_foreign_name and unrelated:
                # An exact model name in other makes contradicts a known brand.
                return self.suggestion("CONFLICT", "BRAND_MODEL_CONFLICT",
                                       self._fixed_rank(unrelated, "foreign_exact_model"))
        return None

    def _fixed_rank(self, ids: set[int], evidence: str) -> list[Ranked]:
        return [Ranked(i, 100, 100, 100, evidence) for i in sorted(ids)]

    def _shared_alias_numeric_context(self, brand: str, model: str) -> Decision | None:
        """Resolve catalog-backed alias + variant context without copying aliases."""
        brand_key, _ = self.brand_key(brand)
        tokens = name_tokens(model)

        if len(tokens) < 2:
            return None

        brand_ids = (self.by_brand[brand_key] if brand_key
                     else set(range(len(self.identities))))
        compatible = set()
        alias_candidates = set()

        for alias, ids in self.alias_lookup.items():
            local_ids = ids & brand_ids
            if not local_ids:
                continue

            alias_tokens = name_tokens(alias)
            if not alias_tokens or len(alias_tokens) >= len(tokens):
                continue

            alias_key = latin_fold(
                normalize.latin_comparison_text(alias) or alias
            )

            for start in range(len(tokens) - len(alias_tokens) + 1):
                end = start + len(alias_tokens)
                span = " ".join(tokens[start:end])

                span_key = latin_fold(
                    normalize.latin_comparison_text(span) or span
                )

                if span_key != alias_key:
                    continue

                context = tokens[:start] + tokens[end:]
                if not context:
                    continue

                alias_candidates.update(local_ids)
                context_signature = token_signature(" ".join(context))

                for base_id in local_ids:
                    base = self.identities[base_id]
                    base_tokens = name_tokens(base.model_key)
                    base_signature = tuple(latin_fold(t) for t in base_tokens)

                    candidate_ids = self.by_brand[base.brand_key]
                    for candidate_id in candidate_ids:
                        if candidate_id == base_id:
                            continue

                        candidate = self.identities[candidate_id]
                        candidate_tokens = name_tokens(candidate.model_key)

                        if len(candidate_tokens) <= len(base_tokens):
                            continue

                        prefix = tuple(
                            latin_fold(t)
                            for t in candidate_tokens[:len(base_tokens)]
                        )

                        if prefix != base_signature:
                            continue

                        suffix = candidate_tokens[len(base_tokens):]
                        if token_signature(" ".join(suffix)) == context_signature:
                            compatible.add(candidate_id)

        if len(compatible) == 1:
            identity_id = next(iter(compatible))
            return self.accepted(
                identity_id,
                "FUZZY_MATCH",
                "CATALOG_ALIAS_VARIANT_CONTEXT",
                97.0,
            )

        if compatible:
            return self.suggestion(
                "AMBIGUOUS",
                "ALIAS_VARIANT_COLLISION",
                self._fixed_rank(compatible, "alias_variant_context"),
            )

        if alias_candidates:
            return self.suggestion(
                "AMBIGUOUS",
                "ALIAS_UNKNOWN_VARIANT",
                self._fixed_rank(alias_candidates, "alias_unknown_variant"),
            )

        return None

    def _cross_field(self, brand: str, model: str) -> Decision | None:
        # Swapped fields: the second field must be an explicit brand and the
        # first an exact canonical/alias model within that brand.
        if brand and model:
            swapped_brand, alias = self.brand_key(model)
            if swapped_brand:
                ids = self.exact_model_ids(brand, swapped_brand)
                if len(ids) == 1:
                    identity_id = next(iter(ids))
                    method = self.exact_method(identity_id, alias, brand, True)
                    status = "ALIAS_MATCH" if "ALIAS" in method else "EXACT_MATCH"
                    return self.accepted(identity_id, status, method)

        # A complete make/model phrase can occur in either field. Require that
        # the other field be absent or agree with the identified make/model.
        for phrase, other in ((brand, model), (model, brand)):
            if not phrase:
                continue
            ids = self.full_lookup.get(phrase, set())
            if len(ids) != 1:
                continue
            identity_id = next(iter(ids))
            identity = self.identities[identity_id]
            other_brand, _ = self.brand_key(other)
            if other and other not in {identity.brand_key, identity.model_key} and other_brand != identity.brand_key:
                if self.compact_full_lookup.get(latin_fold(other)):
                    continue  # Let the multiple-mention check retain both identities.
                if self.token_full_lookup.get(token_signature(brand + " " + model)):
                    continue
                if other_brand and other_brand != identity.brand_key:
                    return self.suggestion("CONFLICT", "CROSS_FIELD_CONFLICT",
                                           self._fixed_rank(ids, "combined_phrase"))
                return self.suggestion("AMBIGUOUS", "COMBINED_FIELD_WITH_EXTRA_TEXT",
                                       self._fixed_rank(ids, "combined_phrase"))
            return self.accepted(identity_id, "EXACT_MATCH", "COMBINED_FIELD_EXACT")
        # A unique catalog make/model pair can be concatenated in either field.
        # The other field must not contradict it. No claim-derived split rules.
        for phrase, other in ((brand, model), (model, brand)):
            if len(compact(phrase)) < 6:
                continue
            ids = self.compact_full_lookup.get(latin_fold(phrase), set())
            if len(ids) != 1:
                continue
            identity_id = next(iter(ids))
            identity = self.identities[identity_id]
            other_brand, _ = self.brand_key(other)
            if other and other not in {identity.brand_key, identity.model_key} and other_brand != identity.brand_key:
                if self.compact_full_lookup.get(latin_fold(other)):
                    continue
                if self.token_full_lookup.get(token_signature(brand + " " + model)):
                    continue
                status = "CONFLICT" if other_brand and other_brand != identity.brand_key else "AMBIGUOUS"
                return self.suggestion(status, "COMPACT_FIELD_WITH_EXTRA_TEXT",
                                       self._fixed_rank(ids, "catalog_compact_pair"))
            return self.accepted(identity_id, "EXACT_MATCH", "CATALOG_COMPACT_PAIR")
        # Split a combined field at observed token boundaries. The resulting
        # make/model must pass the ordinary catalog-based decision policy.
        for phrase, other in ((brand, model), (model, brand)):
            tokens = phrase.split()
            if not 2 <= len(tokens) <= 6:
                continue
            found = {}
            for cut in range(1, len(tokens)):
                suffix = " ".join(tokens[cut:])
                if len(compact(suffix)) < 4 or not (
                        self.exact_model_ids(suffix) or
                        self.compact_model_lookup.get(compact(suffix))):
                    continue
                parsed = self.resolve(" ".join(tokens[:cut]), suffix)
                if parsed.match_status in {"EXACT_MATCH", "ALIAS_MATCH", "FUZZY_MATCH"}:
                    found[(parsed.corrected_brand, parsed.corrected_model)] = parsed
            if len(found) > 1:
                return self.suggestion("AMBIGUOUS", "MULTIPLE_CATALOG_SEGMENTS", [])
            if not found:
                continue
            parsed = next(iter(found.values()))
            other_brand, _ = self.brand_key(other)
            parsed_brand = norm(parsed.corrected_brand)
            parsed_model = norm(parsed.corrected_model)
            if other and other not in {parsed_brand, parsed_model} and other_brand != parsed_brand:
                if self.compact_full_lookup.get(latin_fold(other)):
                    continue
                if self.token_full_lookup.get(token_signature(brand + " " + model)):
                    continue
                status = "CONFLICT" if other_brand and other_brand != parsed_brand else "AMBIGUOUS"
                return self.suggestion(status, "SEGMENTED_FIELD_WITH_EXTRA_TEXT",
                                       list(parsed.top_candidates))
            return Decision(corrected_brand=parsed.corrected_brand,
                            corrected_model=parsed.corrected_model,
                            vehicle_type=parsed.vehicle_type,
                            match_status=parsed.match_status,
                            match_method="SEGMENTED_CATALOG_" + parsed.match_method,
                            match_score=parsed.match_score,
                            confidence_level=parsed.confidence_level,
                            candidate_count=parsed.candidate_count,
                            top_candidates=parsed.top_candidates)
        return None

    def _reported_descriptor(self, brand: str, model: str) -> Decision | None:
        """Separate a standalone reported fuel descriptor from an exact catalog model.

        The catalog validates model identity, NOT the reported fuel. No unknown
        text, numbers, trims, or family tokens are discarded or inferred.
        """
        brand_key, _ = self.brand_key(brand)
        if not brand_key or not model:
            return None
        words = model.rsplit(" ", 1)
        if len(words) != 2 or words[1] not in {
                "DIESEL", "PETROL", "GASOLINE", "ΒΕΝΖΙΝΗ", "ΠΕΤΡΕΛΑΙΟ"}:
            return None
        base, descriptor = words
        # Full model takes precedence (including explicit catalog aliases).
        if self.exact_model_ids(model, brand_key):
            return None
        ids = self.exact_model_ids(base, brand_key)
        if len(ids) != 1:
            return None
        target = next(iter(ids))
        # Do not drop a descriptor used as an actual catalog model token.
        descriptor_key = latin_fold(descriptor)
        if any(descriptor_key == latin_fold(token)
               for i in self.by_brand[brand_key]
               for variant in self.identities[i].variants
               for token in name_tokens(variant)):
            return None
        return self.accepted(target, "FUZZY_MATCH", "CATALOG_MODEL_REPORTED_FUEL",
                             97, extra_info=descriptor)

    def _contextual(self, brand: str, model: str) -> Decision | None:
        """Find complete catalog mentions before discounting surrounding text."""
        tokens = name_tokens(" ".join((brand, model)))
        if not 2 <= len(tokens) <= 12:
            return None
        # Remove only an explicit description vocabulary, and only if the entire
        # remainder is one complete catalog identity. Unknown words stay intact.
        retained = tuple(t for t in tokens if not (
            t in SUPPLEMENTARY_TOKENS and latin_fold(t) not in self.catalog_tokens))
        # A complete token permutation is stronger than a nested substring.
        signature = token_signature(" ".join(retained))
        reordered = self.token_full_lookup.get(signature, set())
        if len(reordered) == 1 and len(tokens) >= 3:
            identity_id = next(iter(reordered))
            identity = self.identities[identity_id]
            ordered = latin_fold(" ".join(retained))
            short_code = any(t.isalpha() and len(latin_fold(t)) <= 2
                             for t in name_tokens(identity.model_key))
            if short_code and numeric_signature(identity.model_key) and ordered not in {
                    latin_fold(identity.brand_key + identity.model_key),
                    latin_fold(identity.model_key + identity.brand_key)}:
                return self.suggestion("AMBIGUOUS", "NUMERIC_TOKEN_ORDER",
                                       self._fixed_rank(reordered, "unverified_numeric_token_order"))
            described = len(retained) != len(tokens)
            return self.accepted(identity_id, "FUZZY_MATCH" if described else "EXACT_MATCH",
                                 "CATALOG_SUPPLEMENTARY_TEXT" if described else "CATALOG_TOKEN_ORDER",
                                 97 if described else 100)
        if len(reordered) > 1:
            return self.suggestion("AMBIGUOUS", "TOKEN_ORDER_COLLISION",
                                   self._fixed_rank(reordered, "catalog_token_order"))
        hits = []
        for start in range(len(tokens)):
            for end in range(start + 1, min(len(tokens), start + 7) + 1):
                key = latin_fold("".join(tokens[start:end]))
                if len(key) < 6:
                    continue
                for identity_id in self.compact_full_lookup.get(key, ()):
                    if end == start + 1:
                        identity = self.identities[identity_id]
                        if (len(latin_fold(identity.brand_key)) < 4 or
                                len(latin_fold(identity.model_key)) < 4):
                            continue
                    hits.append((start, end, identity_id))
        # A nameplate may be a prefix of a longer catalog nameplate. Retain
        # maximal spans so Yaris Cross does not become an extra Yaris mention.
        hits = [hit for hit in hits if not any(
            other[0] <= hit[0] and hit[1] <= other[1] and
            (other[1] - other[0]) > (hit[1] - hit[0]) for other in hits)]
        identities = {hit[2] for hit in hits}
        if len(identities) > 1:
            brands = {self.identities[i].brand_key for i in identities}
            status = "CONFLICT" if len(brands) > 1 else "AMBIGUOUS"
            return self.suggestion(status, "MULTIPLE_CATALOG_MENTIONS",
                                   self._fixed_rank(identities, "distinct_catalog_mentions"))
        if identities:
            identity_id = next(iter(identities))
            covered = {position for start, end, _ in hits for position in range(start, end)}
            extras = [(position, token) for position, token in enumerate(tokens)
                      if position not in covered]
            if not extras:
                return None  # Existing exact/cross-field paths keep their methods.
            collapsed = tuple(token for i, token in enumerate(tokens)
                              if i == 0 or latin_fold(token) != latin_fold(tokens[i - 1]))
            if len(collapsed) < len(tokens) and self.compact_full_lookup.get(
                    latin_fold("".join(collapsed))) == {identity_id}:
                return self.accepted(identity_id, "EXACT_MATCH", "REPEATED_CATALOG_TOKENS")
            identity = self.identities[identity_id]
            base_model = latin_fold(identity.model_key)
            brand_token_count = len(name_tokens(brand))
            family_suffixes = set()
            for other_id in self.by_brand[identity.brand_key]:
                for variant in self.identities[other_id].variants:
                    folded_variant = latin_fold(variant)
                    if folded_variant.startswith(base_model) and len(folded_variant) > len(base_model):
                        family_suffixes.add(folded_variant[len(base_model):])
            for position, token in extras:
                key = latin_fold(token)
                other_brand, _ = self.brand_key(token)
                if other_brand and other_brand != identity.brand_key:
                    status = "CONFLICT" if position < brand_token_count else "AMBIGUOUS"
                    return self.suggestion(status, "CONTEXT_BRAND_CONFLICT",
                                           self._fixed_rank(identities, "catalog_mention"))
                family_risk = any(
                    fuzz.ratio(key, suffix) >= 60 for suffix in family_suffixes
                )
                if (any(char.isdecimal() for char in token) or
                        key in self.model_tokens or
                        family_risk or
                        (position >= max(end for _, end, _ in hits) and len(key) < 4)):
                    brand_key, _ = self.brand_key(brand)
                    if brand_key and model:
                        ranked = self._rank(brand, model, self._candidate_ids(brand, model))
                        relevant = [r for r in ranked if r.model_score >= self.min_model
                                    and DamerauLevenshtein.distance(
                                        compact(model), compact(self.identities[r.identity_id].model_key)) == 1]
                        if relevant:
                            return self.suggestion("AMBIGUOUS", "CATALOG_MENTION_WITH_EXTRA_MODEL_TEXT", relevant)
                    return self.suggestion("AMBIGUOUS", "CATALOG_MENTION_WITH_EXTRA_MODEL_TEXT",
                                           self._fixed_rank(identities, "catalog_mention"))
            # A complete catalog span is a strong candidate, but the catalog
            # cannot establish whether extra words are noise or specifications.
            return self.suggestion("AMBIGUOUS", "CATALOG_MENTION_IN_TEXT",
                                   self._fixed_rank(identities, "catalog_mention"))

        # With one combined field, two misspelled components can be evaluated
        # by the ordinary brand/model ranker without learning a claim mapping.
        if (not brand or not model) and len(tokens) == 2:
            parsed = self.resolve(tokens[0], tokens[1])
            if parsed.match_status in {"EXACT_MATCH", "ALIAS_MATCH", "FUZZY_MATCH"}:
                return Decision(corrected_brand=parsed.corrected_brand,
                                corrected_model=parsed.corrected_model,
                                vehicle_type=parsed.vehicle_type,
                                match_status=parsed.match_status,
                                match_method="COMBINED_FIELD_" + parsed.match_method,
                                match_score=parsed.match_score,
                                confidence_level=parsed.confidence_level,
                                candidate_count=parsed.candidate_count,
                                top_candidates=parsed.top_candidates)
        return None

    def _notation(self, brand: str, model: str) -> Decision | None:
        """Interpret only catalog-backed unit and year suffixes."""
        brand_key, _ = self.brand_key(brand)
        if brand_key and model:
            displacement = re.fullmatch(r"(.+?)\s+(\d+)\s*CC", model)
            if displacement and numeric_signature(displacement[1]):
                base, cc = displacement.groups()
                ids = self.compact_model_lookup.get(compact(base), set()) & self.by_brand[brand_key]
                if (len(ids) == 1 and numeric_signature(base)[-1] == cc and
                        numeric_signature(self.identities[next(iter(ids))].model_key) == numeric_signature(base)):
                    return self.accepted(next(iter(ids)), "FUZZY_MATCH", "DISPLACEMENT_CONFIRMATION", 97)
            without_cc = re.sub(r"(?<=\d)\s*CC$", "", model).strip()
            if without_cc != model and len(compact(without_cc)) >= 4:
                ids = self.compact_model_lookup.get(compact(without_cc), set()) & self.by_brand[brand_key]
                if len(ids) == 1:
                    return self.accepted(next(iter(ids)), "FUZZY_MATCH", "DISPLACEMENT_UNIT", 97)
                if len(ids) > 1:
                    return self.suggestion("AMBIGUOUS", "DISPLACEMENT_COLLISION",
                                           self._fixed_rank(ids, "displacement_unit"))
            without_year = re.sub(r"\s+(?:19|20)\d{2}$", "", model).strip()
            if without_year != model and len(compact(without_year)) >= 4:
                ids = self.exact_model_ids(without_year, brand_key)
                if ids:
                    return self.suggestion("AMBIGUOUS", "POTENTIAL_YEAR_WITHOUT_DATE_EVIDENCE",
                                           self._fixed_rank(ids, "unverified_year_like_number"))
        if brand or not model:
            return None
        without_year = re.sub(r"\s+(?:19|20)\d{2}$", "", model).strip()
        year_removed = without_year != model
        if len(name_tokens(without_year)) != 1:
            return None
        folded = latin_fold(without_year)
        ranked = []
        for size in range(4, min(len(folded) - 3, 40) + 1):
            suffix = folded[-size:]
            for identity_id in self.folded_model_lookup.get(suffix, ()):
                identity = self.identities[identity_id]
                prefix = folded[:-size]
                if len(prefix) < 4:
                    continue
                brand_score = brand_similarity(prefix, identity.brand_key)
                score = round(0.30 * brand_score + 70, 1)
                if brand_score >= 80:
                    ranked.append(Ranked(identity_id, score, round(brand_score, 1),
                                         100, "catalog_model_suffix"))
        ranked.sort(key=lambda item: (-item.score,
                                      self.identities[item.identity_id].brand_key,
                                      self.identities[item.identity_id].model_key))
        if not ranked:
            return None
        best = ranked[0]
        margin = best.score - (ranked[1].score if len(ranked) > 1 else 0)
        if (not year_removed and best.score >= self.min_score and
                margin >= self.min_margin):
            return self.accepted(best.identity_id, "FUZZY_MATCH", "CATALOG_SUFFIX_SPLIT",
                                 best.score, tuple(ranked))
        return self.suggestion("AMBIGUOUS", "CATALOG_SUFFIX_WITH_CONTEXT",
                               ranked)

    @staticmethod
    def _model_similarity(query: str, choice: str) -> float:
        ratio = fuzz.ratio(query, choice)
        weighted = fuzz.WRatio(query, choice)
        token = fuzz.token_sort_ratio(query, choice)
        score = max(ratio, 0.8 * weighted + 0.2 * token)
        if len(compact(query)) >= 4 and compact(query) == compact(choice):
            score = max(score, 97)
        return min(100.0, score)

    def _candidate_ids(self, brand: str, model: str) -> set[int]:
        candidates = set()
        brand_key, _ = self.brand_key(brand)
        folded_ids = self.folded_model_lookup.get(latin_fold(model), set())
        candidates.update(folded_ids & self.by_brand[brand_key] if brand_key else folded_ids)
        if brand_key and informative(model):
            candidates.update(self.compact_model_lookup.get(compact(model), set()) &
                              self.by_brand[brand_key])
            choices = self.brand_model_strings.get(brand_key, [])
            for _, _, choice in process.extract(model, choices, scorer=fuzz.WRatio,
                                                limit=12, score_cutoff=45):
                candidates.update(self.model_choices_by_brand[brand_key][choices[choice]])
            # Compact forms recover MT07/MT-07 without changing input text.
            query_compact = compact(model)
            candidates.update(i for i in self.by_brand[brand_key]
                              if len(query_compact) >= 4 and any(
                                  compact(v) == query_compact for v in self.identities[i].variants))
        elif informative(model):
            candidates.update(self.exact_model_ids(model))
            if len(compact(model)) >= 4:
                candidates.update(self.compact_model_lookup.get(compact(model), set()))
            for _, _, choice in process.extract(model, self.global_model_strings,
                                                scorer=fuzz.WRatio, limit=12, score_cutoff=60):
                candidates.update(self.model_choices_global[self.global_model_strings[choice]])

        if brand and not brand_key and informative(brand):
            possible_brands = set()
            for query, choices, lookup in (
                    (latin_fold(brand), self.brand_folded_strings, self.brand_folded),
                    (phonetic_fold(latin_fold(brand)), self.brand_phonetic_strings,
                     self.brand_phonetic)):
                for value, _, _ in process.extract(query, choices, scorer=fuzz.ratio,
                                                   limit=8, score_cutoff=60):
                    possible_brands.update(lookup[value])
            for candidate_brand in sorted(possible_brands):
                if not informative(model):
                    continue
                candidates.update(self.compact_model_lookup.get(compact(model), set()) &
                                  self.by_brand[candidate_brand])
                choices = self.brand_model_strings[candidate_brand]
                for _, _, choice in process.extract(model, choices, scorer=fuzz.WRatio,
                                                    limit=6, score_cutoff=50):
                    candidates.update(self.model_choices_by_brand[candidate_brand][choices[choice]])
        return candidates

    def _unicode_brand_candidates(self, raw_brand: str) -> tuple[str, ...]:
        """Retrieve a bounded brand set from existing indexes and interpretations."""
        candidates = set()
        for value in normalize.comparison_texts(raw_brand):
            query = latin_fold(value)
            for choice, _, _ in process.extract(
                    query, self.brand_folded_strings, scorer=fuzz.ratio,
                    limit=16, score_cutoff=45):
                candidates.update(self.brand_folded.get(choice, ()))
        return tuple(sorted(candidates))

    def _unicode_candidate_ids(self, brand: str, model: str,
                               raw_brand: str | None, raw_model: str | None) -> set[int]:
        """Retrieve from bounded existing indexes using raw Unicode alternatives."""
        if not (_has_greek(raw_brand) or _has_greek(raw_model)):
            return self._candidate_ids(brand, model)
        cache_key = (brand, model, raw_brand or "", raw_model or "")
        cached = self._unicode_candidate_cache.get(cache_key)
        if cached is not None:
            return set(cached)
        brand_forms = tuple(dict.fromkeys(
            norm(value) for value in normalize.comparison_texts(raw_brand)
            if norm(value)
        )) if _has_greek(raw_brand) else (brand,)
        model_forms = tuple(dict.fromkeys(
            norm(value) for value in normalize.comparison_texts(raw_model)
            if norm(value)
        )) if _has_greek(raw_model) else (model,)
        pairs = [(brand, model)]
        pairs.extend((form, model) for form in brand_forms if form != brand)
        pairs.extend((brand, form) for form in model_forms if form != model)
        # Include a bounded cross-product when both fields contain Greek.
        if _has_greek(raw_brand) and _has_greek(raw_model):
            pairs.extend((brand_form, model_form)
                         for brand_form in brand_forms for model_form in model_forms)
        candidates = set()
        seen = set()
        for query_brand, query_model in pairs:
            key = (query_brand, query_model)
            if key in seen:
                continue
            seen.add(key)
            candidates.update(self._candidate_ids(query_brand, query_model))
            if len(seen) >= 32:
                break
        if len(self._unicode_candidate_cache) >= 2048:
            self._unicode_candidate_cache.clear()
        self._unicode_candidate_cache[cache_key] = frozenset(candidates)
        return candidates

    def _rank(self, brand: str, model: str, ids: set[int],
              brand_latin: str | None = None, model_latin: str | None = None,
              unicode_brand: str | None = None, unicode_model: str | None = None,
              unicode_alternatives: bool = False) -> list[Ranked]:
        brand_key, brand_alias = self.brand_key(brand)
        ranked = []
        for identity_id in ids:
            identity = self.identities[identity_id]
            variants = tuple(dict.fromkeys(
                (*identity.variants, *self.model_comparison_variants.get(identity_id, ()))
            ))
            if brand_key and not (unicode_alternatives and _has_greek(unicode_brand)):
                brand_score = 100.0 if identity.brand_key == brand_key else 0.0
            elif brand:
                # WRatio treats a one-letter prefix as a near-perfect brand.
                # Plain edit ratio is safer for automatic brand correction.
                if unicode_alternatives and _has_greek(unicode_brand):
                    brand_score = unicode_weighted_similarity(unicode_brand, identity.brand_key)
                else:
                    brand_score = max(
                        brand_similarity(brand, identity.brand_key),
                        brand_similarity(brand_latin or brand, identity.brand_key),
                        unicode_weighted_similarity(unicode_brand, identity.brand_key)
                        if _has_greek(unicode_brand) else 0.0,
                    )
            else:
                brand_score = 0.0
            if unicode_alternatives and _has_greek(unicode_model):
                model_score = max((unicode_weighted_similarity(unicode_model, variant)
                                   for variant in variants), default=0.0)
            else:
                model_score = max((self._model_similarity(model, variant)
                                   for variant in variants), default=0.0)
                if model_latin is not None:
                    model_score = max(model_score, max((self._model_similarity(
                        model_latin, variant)
                        for variant in variants), default=0.0))
                if _has_greek(unicode_model):
                    model_score = max(model_score, max((unicode_weighted_similarity(
                        unicode_model, variant) for variant in variants), default=0.0))
            if brand_key and identity.brand_key != brand_key and not unicode_alternatives:
                continue
            if brand:
                score = 0.30 * brand_score + 0.70 * model_score
            else:
                score = model_score
            if model == identity.model_key:
                evidence = "canonical_model"
            elif model in identity.variants:
                evidence = "catalog_alias"
            elif compact(model) == compact(identity.model_key):
                evidence = "compact_form"
            else:
                evidence = "fuzzy_text"
            if brand_alias:
                evidence += "+" + brand_alias.lower()
            ranked.append(Ranked(identity_id, round(score, 1), round(brand_score, 1),
                                 round(model_score, 1), evidence))
        ranked.sort(key=lambda r: (-r.score, -r.model_score,
                                   self.identities[r.identity_id].brand_key,
                                   self.identities[r.identity_id].model_key))
        return ranked

    def resolve(self, brand_value: str | None, model_value: str | None, *,
                brand_latin: str | None = None, model_latin: str | None = None) -> Decision:
        brand, model = norm(brand_value), norm(model_value)
        comparison_text(brand_value, brand_latin)
        comparison_text(model_value, model_latin)
        brand_valid = bool(self.brand_key(brand)[0]) or informative(brand)
        model_valid = informative(model) or bool(
            self.exact_model_ids(model) |
            self.folded_model_lookup.get(latin_fold(model_latin), set())
        )
        usable_brand = brand if brand_valid else ""
        usable_model = model if model_valid else ""
        stripped = (brand[:-1] if isinstance(brand_value, str) and brand_value.endswith("/")
                    and brand.endswith("/") and not brand.endswith("//") else "")
        strip_brand_slash = (brand_valid and stripped in self.brands
                             and not any(c in usable_model for c in "+/")
                             and len(self.exact_model_ids(usable_model, stripped)) == 1)
        decision = self._resolve(stripped if strip_brand_slash else brand_value, model_value,
                                 brand_latin=None if strip_brand_slash else brand_latin,
                                 model_latin=model_latin)
        unicode_acceptance = decision.match_status in {"EXACT_MATCH", "ALIAS_MATCH", "FUZZY_MATCH"}
        unicode_brand_only = (decision.match_status == "PARTIAL_MATCH"
                              and bool(brand_value) and not model)
        unicode_model_only_partial = (decision.match_status == "PARTIAL_MATCH"
                                      and not brand_value and bool(model))
        if ((_has_greek(brand_value) or _has_greek(model_value))
                and (unicode_acceptance or unicode_brand_only or unicode_model_only_partial)):
            decision = self._unicode_acceptance_guard(
                decision, brand_value, model_value, brand, model, brand_latin, model_latin
            )
        if decision.match_status in {"EXACT_MATCH", "ALIAS_MATCH", "FUZZY_MATCH"}:
            identity = self.identities[self.by_pair[(norm(decision.corrected_brand), norm(decision.corrected_model))]]
            observed = punctuation_signature(
                (stripped if strip_brand_slash else usable_brand) + usable_model
            )
            supported = {punctuation_signature(identity.brand_key + variant)
                         for variant in identity.variants}
            if observed not in supported:
                if ("+" in punctuation_signature(usable_model)
                        and self.brand_key(usable_brand)[0] == identity.brand_key):
                    decision = replace(
                        decision, corrected_brand=identity.brand, corrected_model="",
                        match_status="PARTIAL_MATCH",
                        match_method="MODEL_PUNCTUATION_UNVERIFIED",
                    )
                else:
                    decision = self.suggestion("AMBIGUOUS", "MEANINGFUL_PUNCTUATION_MISMATCH",
                                               list(decision.top_candidates))
            if decision.match_status == "ALIAS_MATCH" and not decision.extra_info:
                if decision.match_method == "FOLDED_EXPLICIT_ALIAS":
                    decision = replace(decision, extra_info=model_value or "")
                else:
                    for original in (model_value, brand_value):
                        value = norm(original)
                        if value != identity.model_key and value in identity.variants:
                            decision = replace(decision, extra_info=original)
                            break
        elif decision.match_status == "PARTIAL_MATCH" and decision.corrected_model \
                and decision.top_candidates:
            # Partial catalog evidence must not erase a meaningful input
            # separator such as '+' from the corrected field.
            identity = self.identities[decision.top_candidates[0].identity_id]
            observed = punctuation_signature(usable_model)
            supported = {punctuation_signature(variant) for variant in identity.variants}
            if observed not in supported:
                decision = replace(decision, corrected_model=model)
        return replace(
            decision,
            corrected_brand=normalize.normalize_text(decision.corrected_brand) or "",
            corrected_model=normalize.normalize_text(decision.corrected_model) or "",
        )

    def _unicode_acceptance_guard(self, decision: Decision, raw_brand: str | None,
                                  raw_model: str | None, brand: str, model: str,
                                  brand_latin: str | None, model_latin: str | None) -> Decision:
        """Abstain when weighted Unicode evidence conflicts with an apparent match."""
        verified_alias_methods = {
            "EXPLICIT_ALIAS", "FOLDED_EXPLICIT_ALIAS", "MODEL_ONLY_ALIAS",
            "CATALOG_ALIAS_VARIANT_CONTEXT",
        }
        if decision.match_method in verified_alias_methods:
            return decision

        if raw_brand and not model:
            ranks = []
            brand_candidates = set(self._unicode_brand_candidates(raw_brand))
            accepted_brand = norm(decision.corrected_brand)
            if accepted_brand in self.brands:
                brand_candidates.add(accepted_brand)
            for candidate_brand in sorted(brand_candidates):
                score = unicode_weighted_similarity(raw_brand, candidate_brand)
                representative = min(
                    self.by_brand[candidate_brand],
                    key=lambda identity_id: self.identities[identity_id].model_key,
                )
                ranks.append(Ranked(representative, score, score, 0.0, "unicode_brand"))
            ranks.sort(key=lambda item: (-item.score,
                                         self.identities[item.identity_id].brand_key))
            if not ranks or ranks[0].score < self.min_brand:
                return self.suggestion("AMBIGUOUS", "UNICODE_BRAND_INSUFFICIENT", ranks)
            expected_brand = norm(decision.corrected_brand)
            winner = self.identities[ranks[0].identity_id].brand_key
            margin = ranks[0].score - (ranks[1].score if len(ranks) > 1 else 0.0)
            if expected_brand != winner or margin < self.min_margin:
                return self.suggestion("AMBIGUOUS", "UNICODE_BRAND_COMPETITION", ranks)
            return decision

        candidate_ids = self._unicode_candidate_ids(brand, model, raw_brand, raw_model)
        if decision.corrected_brand and decision.corrected_model:
            accepted_id = self.by_pair.get((norm(decision.corrected_brand),
                                            norm(decision.corrected_model)))
            if accepted_id is not None:
                candidate_ids.add(accepted_id)
        ranks = self._rank(
            brand, model, candidate_ids, brand_latin, model_latin,
            unicode_brand=raw_brand, unicode_model=raw_model,
            unicode_alternatives=True,
        )
        if not ranks or ranks[0].score < self.min_score:
            return self.suggestion("AMBIGUOUS", "UNICODE_IDENTITY_INSUFFICIENT", ranks)
        expected_brand = norm(decision.corrected_brand)
        margin = ranks[0].score - (ranks[1].score if len(ranks) > 1 else 0.0)
        if decision.match_status == "PARTIAL_MATCH" and not raw_brand and expected_brand:
            winner_brand = self.identities[ranks[0].identity_id].brand_key
            if winner_brand != expected_brand or margin < self.min_margin:
                return self.suggestion("AMBIGUOUS", "UNICODE_BRAND_CONSENSUS", ranks)
            return decision
        expected_id = self.by_pair.get((expected_brand, norm(decision.corrected_model)))
        if ranks[0].identity_id != expected_id or margin < self.min_margin:
            return self.suggestion("AMBIGUOUS", "UNICODE_IDENTITY_COMPETITION", ranks)
        return decision

    def _resolve(self, brand_value: str | None, model_value: str | None, *,
                 brand_latin: str | None = None, model_latin: str | None = None) -> Decision:
        brand, model = norm(brand_value), norm(model_value)
        brand_latin = comparison_text(brand_value, brand_latin)
        model_latin = comparison_text(model_value, model_latin)
        known_brand = self.brand_key(brand)[0] if brand else ""
        known_models = (self.exact_model_ids(model) |
                        self.folded_model_lookup.get(latin_fold(model_latin), set())) if model else set()
        if not known_brand and not informative(brand):
            brand, brand_latin = "", ""
        if not known_models and not informative(model):
            model, model_latin = "", ""
        if not brand and not model:
            return Decision()
        exact = self._exact(brand, model)
        if exact:
            return exact
        shared_alias_context = self._shared_alias_numeric_context(brand, model)
        if shared_alias_context:
            return shared_alias_context
        if brand:
            notation = self._notation(brand, model)
            if notation:
                return notation
        # Preserve catalog-backed repeated make/model tokens before the
        # segmented cross-field shortcut can treat them as unrelated extras.
        if ((len(name_tokens(brand)) > 1 or len(name_tokens(model)) > 1)
                and (len(set(name_tokens(brand))) < len(name_tokens(brand))
                     or len(set(name_tokens(model))) < len(name_tokens(model)))):
            repeated = self._contextual(brand, model)
            if repeated and repeated.match_method == "REPEATED_CATALOG_TOKENS":
                return repeated
        cross = self._cross_field(brand, model)
        if cross:
            return cross
        described = self._reported_descriptor(brand, model)
        if described:
            return described
        contextual = self._contextual(brand, model)
        if contextual:
            return contextual
        if not brand:
            notation = self._notation(brand, model)
            if notation:
                return notation

        # Additional comparison evidence cannot bypass an exact foreign-model conflict.
        if (brand_latin, model_latin) != (brand, model):
            conflict = self._exact(brand_latin, model_latin)
            if conflict and conflict.match_status == "CONFLICT":
                return conflict

        brand_key, brand_alias = self.brand_key(brand)
        if not brand and len(compact(model)) >= 3 and any(c.isalpha() for c in model):
            model_ids = (self.exact_model_ids(model) |
                         self.folded_model_lookup.get(latin_fold(model_latin), set()))
            candidate_ids = model_ids | self._unicode_candidate_ids(
                "", model, None, model_value
            )
            if model_latin and model_latin != model:
                candidate_ids |= self._candidate_ids("", model_latin)
            ranked = self._rank("", model, candidate_ids, model_latin=model_latin,
                                unicode_model=model_value)
            if not ranked:
                return Decision()

            model_punctuation = punctuation_signature(model)
            punctuation_supported = [
                item for item in ranked
                if any(punctuation_signature(variant) == model_punctuation
                       for variant in self.identities[item.identity_id].variants)
            ]
            if model_ids:
                exact_ranked = [item for item in punctuation_supported
                                if item.identity_id in model_ids]
                if len(exact_ranked) == 1:
                    best = exact_ranked[0]
                    runner_up = max((item.score for item in punctuation_supported
                                     if item.identity_id != best.identity_id), default=0.0)
                    margin = best.score - runner_up
                    if (best.model_score >= self.min_model and best.score >= self.min_score
                            and margin >= self.min_margin):
                        identity = self.identities[best.identity_id]
                        status = ("EXACT_MATCH" if model == identity.model_key
                                  else "ALIAS_MATCH")
                        method = ("MODEL_ONLY_CANONICAL" if status == "EXACT_MATCH"
                                  else "MODEL_ONLY_ALIAS")
                        return self.accepted(best.identity_id, status, method,
                                             best.score, tuple(ranked))

            best = ranked[0]
            supported_ranked = [item for item in ranked if item in punctuation_supported]
            if supported_ranked:
                best_supported = supported_ranked[0]
                supported_margin = (best_supported.score - supported_ranked[1].score
                                    if len(supported_ranked) > 1 else best_supported.score)
                if (best_supported.model_score >= self.min_model
                        and best_supported.score >= self.min_score
                        and supported_margin >= self.min_margin
                        and not model_family_risk(
                            model, self.identities[best_supported.identity_id].model_key)):
                    return self.accepted(best_supported.identity_id, "FUZZY_MATCH",
                                         "MODEL_ONLY_FUZZY", best_supported.score,
                                         tuple(ranked))

            # A shared/near model name can still establish a brand when the
            # strongest supported candidates agree and clear the configured
            # score and inter-brand margin. Never use the first model as a proxy.
            brand_scores = {}
            for item in ranked:
                if item.score >= self.min_score and item.model_score >= self.min_model:
                    key = self.identities[item.identity_id].brand_key
                    brand_scores[key] = max(brand_scores.get(key, 0.0), item.score)
            ordered_brands = sorted(brand_scores.items(), key=lambda pair: (-pair[1], pair[0]))
            if ordered_brands:
                brand_key, score = ordered_brands[0]
                brand_margin = score - (ordered_brands[1][1] if len(ordered_brands) > 1 else 0.0)
                if brand_margin >= self.min_margin:
                    return Decision(
                        corrected_brand=self.brand_choices[brand_key],
                        match_status="PARTIAL_MATCH",
                        match_method="MODEL_ONLY_BRAND_CONSENSUS",
                        match_score=f"{score:.1f}", confidence_level="MEDIUM",
                        candidate_count=len(ranked), top_candidates=tuple(ranked[:self.top_k]),
                    )
            if best.score >= self.min_score:
                return self.suggestion("AMBIGUOUS", "MODEL_ONLY_CANDIDATES", ranked)
            return Decision(match_status="UNRESOLVED", match_method="MODEL_ONLY_UNRESOLVED",
                            match_score=f"{best.score:.1f}", confidence_level="NONE",
                            candidate_count=len(ranked), top_candidates=tuple(ranked[:self.top_k]))
        if brand and not model:
            if brand_key:
                representative = min(
                    self.by_brand[brand_key],
                    key=lambda identity_id: self.identities[identity_id].model_key,
                )
                ranked = (Ranked(representative, 100.0, 100.0, 0.0,
                                 "brand_alias_only" if brand_alias else "brand_only"),)
                return Decision(
                    corrected_brand=self.brand_choices[brand_key],
                    match_status="PARTIAL_MATCH",
                    match_method=("BRAND_" + brand_alias + "_ONLY") if brand_alias else "BRAND_ONLY",
                    match_score="100.0", confidence_level="HIGH",
                    candidate_count=1, top_candidates=ranked,
                )

            comparisons = tuple(dict.fromkeys((brand, brand_latin)))
            brand_ranks = []
            brand_candidates = (self._unicode_brand_candidates(brand_value)
                                if _has_greek(brand_value) else tuple(self.brands))
            for candidate_brand in brand_candidates:
                score = max(
                    *(brand_similarity(query, candidate_brand) for query in comparisons),
                    unicode_weighted_similarity(brand_value, candidate_brand)
                    if _has_greek(brand_value) else 0.0,
                )
                representative = min(
                    self.by_brand[candidate_brand],
                    key=lambda identity_id: self.identities[identity_id].model_key,
                )
                brand_ranks.append((candidate_brand, Ranked(
                    representative, round(score, 1), round(score, 1), 0.0,
                    "brand_only_fuzzy",
                )))
            brand_ranks.sort(key=lambda item: (-item[1].score, item[0]))
            ranked = [item[1] for item in brand_ranks]
            best = ranked[0] if ranked else None
            margin = best.score - ranked[1].score if len(ranked) > 1 else (best.score if best else 0)
            if best and best.score >= self.min_brand and margin >= self.min_margin:
                identity = self.identities[best.identity_id]
                return Decision(
                    corrected_brand=identity.brand, match_status="PARTIAL_MATCH",
                    match_method="BRAND_ONLY_FUZZY", match_score=f"{best.score:.1f}",
                    confidence_level="MEDIUM", candidate_count=len(ranked),
                    top_candidates=tuple(ranked[:self.top_k]),
                )
            if best and best.score >= self.min_brand:
                return self.suggestion("AMBIGUOUS", "BRAND_ONLY_CANDIDATES", ranked)
            return Decision(
                match_status="UNRESOLVED", match_method="BRAND_ONLY_UNRESOLVED",
                match_score=f"{best.score:.1f}" if best else "",
                confidence_level="NONE", candidate_count=len(ranked),
                top_candidates=tuple(ranked[:self.top_k]),
            )
        if not informative(model):
            if brand_key:
                return Decision(corrected_brand=self.brand_choices[brand_key],
                                match_status="PARTIAL_MATCH", match_method="BRAND_ONLY_WEAK_MODEL",
                                match_score="100.0", confidence_level="LOW")
            return Decision()

        candidate_ids = self._unicode_candidate_ids(
            brand, model, brand_value, model_value
        )
        if model_latin and model_latin != model:
            candidate_ids |= self._candidate_ids(brand, model_latin)
        if (brand_latin, model_latin) != (brand, model):
            candidate_ids |= self._candidate_ids(brand_latin, model_latin)
        candidates = self._rank(brand, model, candidate_ids, brand_latin, model_latin,
                                unicode_brand=brand_value, unicode_model=model_value)
        if not candidates:
            if brand_key:
                return Decision(corrected_brand=self.brand_choices[brand_key],
                                match_status="PARTIAL_MATCH", match_method="BRAND_ONLY_NO_MODEL_CANDIDATE",
                                match_score="100.0", confidence_level="LOW")
            return Decision()
        best = candidates[0]
        runner_up = candidates[1].score if len(candidates) > 1 else 0.0
        margin = best.score - runner_up
        identity = self.identities[best.identity_id]
        if (brand and brand.isascii() and model.isascii() and not brand_key
                and best.brand_score < max(80, self.min_brand - 8) and best.model_score < 80):
            return self.suggestion("AMBIGUOUS", "INSUFFICIENT_CANDIDATE_EVIDENCE", [])
        has_brand_evidence = bool(brand and best.brand_score >= self.min_brand)
        # A distinctive exact catalog model supplies independent evidence for
        # cross-script brand spelling; this rule does not inspect claim counts.
        model_ids = (self.exact_model_ids(model) | self.compact_model_lookup.get(compact(model), set()) |
                     self.folded_model_lookup.get(latin_fold(model_latin), set()))
        unique_model_brand = (len({self.identities[i].brand_key for i in model_ids}) == 1
                              and best.identity_id in model_ids)
        if brand and unique_model_brand and best.brand_score >= max(80, self.min_brand - 8):
            has_brand_evidence = True
        if brand and not brand_key and len(compact(brand)) < 4:
            has_brand_evidence = False
        extra_brand_token = bool(brand and not brand_key and
                                 brand.startswith(identity.brand_key + " "))
        if (has_brand_evidence and best.model_score >= self.min_model
                and best.score >= self.min_score and margin >= self.min_margin
                and not extra_brand_token
                and not model_family_risk(model, identity.model_key)
                and not model_family_risk(model_latin, normalize.latin_comparison_text(identity.model_key) or "")
                and not (model.isdecimal() and len(model) <= 3
                         and identity.vehicle_type == "unknown")):
            return self.accepted(best.identity_id, "FUZZY_MATCH", "RANKED_FUZZY",
                                 best.score, tuple(candidates))
        # A narrowly bounded alternative to the existing acceptance gate.
        # Similarity >=80 alone is never proof of identity: require a known
        # brand, distinctive winner, and unchanged model-number/family tokens.
        best_tokens = name_tokens(identity.model_key)
        input_tokens = name_tokens(model)
        if (brand_key == identity.brand_key and best.brand_score == 100
                and best.model_score >= 80 and best.score >= 85 and margin >= self.min_margin
                and len(input_tokens) <= len(best_tokens)
                and bool(compact(model)) and bool(compact(identity.model_key))
                and compact(model)[0] == compact(identity.model_key)[0]
                and numeric_signature(model) == numeric_signature(identity.model_key)
                and len(compact(model)) >= 4
                and len(best_tokens) == len(input_tokens) == 1
                and DamerauLevenshtein.distance(compact(model), compact(identity.model_key)) == 1
                and not (model.isdecimal() and len(model) <= 3)
                and not (len(best_tokens) > 1 and len(input_tokens) < len(best_tokens))):
            return self.accepted(best.identity_id, "FUZZY_MATCH", "RANKED_FUZZY_REVIEW_POLICY",
                                 best.score, tuple(candidates))
        # Exact model evidence can safely reinforce a misspelled manufacturer.
        # Never use this for brandless records, short brands, or partial model names.
        # A unique model spelling alone is not sufficient: the manufacturer must
        # still be a distinct fuzzy brand candidate and model digits must agree.
        exact_local_model = (self.exact_model_ids(model, identity.brand_key) |
                             (self.compact_model_lookup.get(compact(model), set()) &
                              self.by_brand[identity.brand_key]))
        if (brand and not brand_key and len(compact(brand)) >= 4
                and len(compact(model)) >= 4
                and best.identity_id in exact_local_model
                and len(exact_local_model) == 1
                and best.model_score >= 99
                and best.brand_score >= 70
                and margin >= (3 if best.brand_score >= 90 else 6)
                and numeric_signature(model) == numeric_signature(identity.model_key)
                and not extra_brand_token
                and not model_family_risk(model, identity.model_key)
                and not model_family_risk(model_latin,
                                          normalize.latin_comparison_text(identity.model_key) or "")
                and not any(other.identity_id != best.identity_id
                            and other.brand_score >= best.brand_score - 5
                            and other.model_score >= 98
                            for other in candidates[1:])):
            return self.accepted(best.identity_id, "FUZZY_MATCH", "EXACT_MODEL_FUZZY_BRAND",
                                 best.score, tuple(candidates))
        if brand_key and best.model_score < 65:
            return Decision(corrected_brand=self.brand_choices[brand_key],
                            match_status="PARTIAL_MATCH", match_method="BRAND_ONLY_WEAK_MODEL",
                            match_score=f"{best.score:.1f}", confidence_level="LOW",
                            candidate_count=len(candidates), top_candidates=tuple(candidates[:self.top_k]))
        return self.suggestion("AMBIGUOUS", "RANKED_CANDIDATES", candidates)

    @classmethod
    def from_csv(cls, catalog_path: str | Path, *, brand_aliases_path: str | Path | None = None,
                 min_score: float = DEFAULT_MIN_SCORE, min_margin: float = DEFAULT_MIN_MARGIN,
                 min_brand: float = DEFAULT_MIN_BRAND, min_model: float = DEFAULT_MIN_MODEL,
                 top_k: int = DEFAULT_TOP_K) -> "Matcher":
        """Load and index the reference catalog once for reuse across batches."""
        rows = read_csv(Path(catalog_path), CATALOG_REQUIRED_FIELDS)
        brand_aliases = (read_csv(Path(brand_aliases_path), {"brand", "alias"})
                         if brand_aliases_path is not None else [])
        return cls(rows, brand_aliases, min_score=min_score, min_margin=min_margin,
                   min_brand=min_brand, min_model=min_model, top_k=top_k)

    def candidate_records(self, decision: Decision) -> list[dict]:
        """Return ordered canonical catalog candidates, including their evidence."""
        return [
            {"brand": self.identities[item.identity_id].brand,
             "model": self.identities[item.identity_id].model,
             "vehicle_type": self.identities[item.identity_id].vehicle_type,
             "score": item.score, "brand_score": item.brand_score,
             "model_score": item.model_score, "evidence": item.evidence}
            for item in decision.top_candidates
        ]

    def match_dataframe(self, df, *, prefix: str = "ab_", top_k: int | None = None,
                        debug: bool = False):
        """Match a normalized DataFrame and return compact result columns.

        Expected columns come from normalize.normalize_dataframe(..., prefix=prefix).
        Original brand/model columns are retained only when present. Candidates
        contain canonical brand, model, and score; debug adds normalized values,
        component scores, margin, and method.
        Set Matcher(top_k=N) or Matcher.from_csv(top_k=N) to retain N candidates.
        For matching parity, use the same missing-token policy as the default
        normalize.normalize_dataframe() configuration.
        """
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("match_dataframe requires pandas; install pandas") from exc
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        if not isinstance(prefix, str):
            raise TypeError("prefix must be a string")
        if not isinstance(debug, bool):
            raise TypeError("debug must be a bool")
        if not df.columns.is_unique:
            raise ValueError("DataFrame column names must be unique")
        if top_k is None:
            top_k = self.top_k
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= self.top_k:
            raise ValueError(f"top_k must be between 1 and the matcher's top_k ({self.top_k})")

        required = tuple(prefix + name for name in (
            "brand_normalized", "model_normalized", "brand_latin", "model_latin"))
        absent = [name for name in required if name not in df.columns]
        if absent:
            raise ValueError(f"Missing normalized DataFrame columns: {', '.join(absent)}")
        original_columns = [name for name in ("brand", "model") if name in df.columns]
        brand_only_schema = original_columns == ["brand"]
        model_only_schema = original_columns == ["model"]
        output_columns = (
            ["corrected_brand"] if brand_only_schema else
            ["corrected_model"] if model_only_schema else
            ["corrected_brand", "corrected_model"]
        ) + ["match_status", "match_score", "candidates"]
        debug_columns = ([prefix + name for name in (
            "brand_normalized", "model_normalized", "brand_score", "model_score",
            "score_margin", "match_method",
        )] if debug else [])
        generated_columns = output_columns + debug_columns
        expected_inputs = set(required) | {"brand", "model"}
        collisions = sorted(set(generated_columns) & (set(df.columns) - expected_inputs))
        if collisions:
            raise ValueError(f"Output column collision: {', '.join(collisions)}")

        cache: dict[tuple[str, ...], Decision] = {}
        records = []
        inputs = (df[name] for name in required)
        raw_brands = df["brand"] if "brand" in df.columns else [None] * len(df)
        raw_models = df["model"] if "model" in df.columns else [None] * len(df)
        for brand, model, brand_latin, model_latin, raw_brand, raw_model in zip(
                *inputs, raw_brands, raw_models):
            if any(not isinstance(value, str) for value in
                   (brand, model, brand_latin, model_latin)):
                raise TypeError("Normalized and Latin comparison columns must contain strings")
            if any(norm(value) != value for value in (brand, model)):
                raise ValueError("Normalized brand/model values are inconsistent with the matching normalization policy")
            raw_brand = raw_brand if isinstance(raw_brand, str) and brand else brand
            raw_model = raw_model if isinstance(raw_model, str) and model else model
            values = (brand, model, brand_latin, model_latin, raw_brand, raw_model)
            if values not in cache:
                brand_alternatives = normalize.comparison_texts(raw_brand)
                model_alternatives = normalize.comparison_texts(raw_model)
                if brand_latin not in brand_alternatives or model_latin not in model_alternatives:
                    raise ValueError("Latin comparison columns disagree with normalized text")
                decisions = [
                    self.resolve(raw_brand, raw_model,
                                 brand_latin=brand_alt, model_latin=model_alt)
                    for brand_alt in brand_alternatives
                    for model_alt in model_alternatives
                ]
                ranked_by_id = {}
                for item in (item for decision in decisions for item in decision.top_candidates):
                    current = ranked_by_id.get(item.identity_id)
                    if current is None or (item.score, item.model_score, item.brand_score) > (
                            current.score, current.model_score, current.brand_score):
                        ranked_by_id[item.identity_id] = item
                ranked_all = sorted(
                    ranked_by_id.values(),
                    key=lambda item: (-item.score, -item.model_score,
                                      self.identities[item.identity_id].brand_key,
                                      self.identities[item.identity_id].model_key),
                )
                accepted_decisions = [decision for decision in decisions
                                      if decision.match_status in {"EXACT_MATCH", "ALIAS_MATCH", "FUZZY_MATCH"}]
                accepted_pairs = {(decision.corrected_brand, decision.corrected_model)
                                  for decision in accepted_decisions}
                partial_decisions = [decision for decision in decisions
                                     if decision.match_status == "PARTIAL_MATCH"
                                     and (decision.corrected_brand or decision.corrected_model)]
                partial_pairs = {(decision.corrected_brand, decision.corrected_model)
                                 for decision in partial_decisions}
                if len(accepted_pairs) > 1 or (not accepted_pairs and len(partial_pairs) > 1):
                    decision = self.suggestion("AMBIGUOUS", "UNICODE_ALTERNATIVE_CONFLICT", ranked_all)
                elif accepted_pairs:
                    decision = accepted_decisions[0]
                    chosen_pair = next(iter(accepted_pairs))
                    competing = any(
                        alternative.match_status == "CONFLICT"
                        or (alternative.match_status == "AMBIGUOUS"
                            and alternative.top_candidates
                            and (self.identities[alternative.top_candidates[0].identity_id].brand,
                                 self.identities[alternative.top_candidates[0].identity_id].model) != chosen_pair
                            and alternative.top_candidates[0].score >= self.min_score)
                        for alternative in decisions
                    )
                    if competing:
                        decision = self.suggestion(
                            "AMBIGUOUS", "UNICODE_ALTERNATIVE_CONFLICT", ranked_all
                        )
                elif partial_pairs:
                    decision = partial_decisions[0]
                else:
                    decision = decisions[0]
                cache[values] = replace(
                    decision,
                    candidate_count=max(decision.candidate_count, len(ranked_all)),
                    top_candidates=tuple(ranked_all[:self.top_k]),
                )
            decision = cache[values]
            status = decision.match_status
            valid_brand = bool(brand) and (
                bool(self.brand_key(brand)[0]) or informative(brand)
            )
            valid_model = bool(model) and (
                informative(model) or bool(
                    self.exact_model_ids(model) |
                    self.folded_model_lookup.get(latin_fold(model_latin), set())
                )
            )
            if status == "PARTIAL_MATCH" and decision.corrected_brand \
                    and valid_brand and not valid_model:
                status = "MATCHED"
            elif status in {"EXACT_MATCH", "ALIAS_MATCH", "FUZZY_MATCH"} \
                    and not valid_brand and valid_model:
                status = "MATCHED"

            ranked_all = decision.top_candidates
            ranked = decision.top_candidates[:top_k]
            brand_only = brand_only_schema or not model
            candidates = []
            for item in ranked:
                identity = self.identities[item.identity_id]
                candidate = {"brand": normalize.normalize_text(identity.brand) or ""}
                if not brand_only:
                    candidate["model"] = normalize.normalize_text(identity.model) or ""
                candidate["score"] = item.score
                candidates.append(candidate)
            corrected_brand = normalize.normalize_text(decision.corrected_brand) or brand
            corrected_model = normalize.normalize_text(decision.corrected_model) or model
            row = {
                "corrected_brand": (corrected_brand or "").upper(),
                "corrected_model": (corrected_model or "").upper(),
                "match_status": status,
                "match_score": float(decision.match_score) if decision.match_score else None,
                "candidates": candidates,
            }
            if debug:
                row.update({
                    prefix + "brand_normalized": brand,
                    prefix + "model_normalized": model,
                    prefix + "brand_score": ranked_all[0].brand_score if ranked_all else None,
                    prefix + "model_score": ranked_all[0].model_score if ranked_all else None,
                    prefix + "score_margin": round(ranked_all[0].score - ranked_all[1].score, 1)
                    if len(ranked_all) > 1 else None,
                    prefix + "match_method": decision.match_method,
                })
            if brand_only_schema:
                row.pop("corrected_model", None)
            elif model_only_schema:
                row.pop("corrected_brand", None)
            records.append(row)
        result = df[original_columns].copy()
        for name in generated_columns:
            result[name] = [record[name] for record in records]
        return result

    def suggested_identity(self, decision: Decision) -> tuple[str, str]:
        """Return a unique leading candidate, never an arbitrary tied winner."""
        items = decision.top_candidates
        if not items:
            return "", ""
        # When top_k=1, a second, potentially tied candidate was not retained.
        if len(items) == 1 and decision.candidate_count > 1:
            return "", ""
        if (len(items) > 1 and items[0].score == items[1].score
                and items[0].model_score == items[1].model_score):
            # Preserve the brand when the competing identities share it.
            brands = {self.identities[item.identity_id].brand for item in items}
            return (next(iter(brands)) if len(brands) == 1 else ""), ""
        identity = self.identities[items[0].identity_id]
        return identity.brand, identity.model

    def serialize_candidates(self, decision: Decision) -> str:
        """JSON rendering for CSV output; Python callers use candidate_records()."""
        return json.dumps(self.candidate_records(decision), ensure_ascii=False,
                          separators=(",", ":"))


HERE = Path(__file__).resolve().parent
INPUT = HERE / "output" / "claims_normalized.csv"
CATALOG = HERE / "data" / "vehicle_catalog.csv"
BRAND_ALIASES = None
OUTPUT = HERE / "output" / "claims_matched.csv"
OUTPUT_COLUMNS = (
    "brand_raw", "model_raw", "brand_normalized", "model_normalized",
    "brand_missing", "model_missing", "corrected_brand", "corrected_model",
    "suggested_brand", "suggested_model", "vehicle_type", "match_status", "result_status", "match_method",
    "match_score", "brand_score", "model_score", "score_margin",
    "confidence_level", "candidate_count", "top_candidates",
    "quality_flags", "brand_latin", "model_latin", "extra_info",
)


def public_status(decision: Decision) -> str:
    if decision.match_status in {"EXACT_MATCH", "ALIAS_MATCH", "FUZZY_MATCH"}:
        return "MATCHED"
    if decision.match_status == "UNRESOLVED":
        return "UNRESOLVED"
    return "AMBIGUOUS"


def write_csv_atomic(path: Path, columns: tuple[str, ...], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent,
                                         prefix=".autobrands_", suffix=".tmp", delete=False) as f:
            temporary = Path(f.name)
            writer = csv.DictWriter(f, fieldnames=columns, lineterminator="\n", extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


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
        min_score: float, min_margin: float, min_brand: float, min_model: float,
        top_k: int = DEFAULT_TOP_K) -> dict:
    paths = (input_path.resolve(), catalog_path.resolve())
    if aliases_path is not None:
        paths += (aliases_path.resolve(),)
    if output_path.resolve() in paths:
        raise ValueError("Output must differ from input, catalog, and brand alias paths")
    started = time.perf_counter()
    rows = read_csv(input_path, set(OUTPUT_COLUMNS[:6]) | {"quality_flags"})
    catalog_rows = read_csv(catalog_path, CATALOG_REQUIRED_FIELDS)
    alias_rows = read_csv(aliases_path, {"brand", "alias"}) if aliases_path is not None else []
    matcher = Matcher(catalog_rows, alias_rows, min_score, min_margin, min_brand, min_model, top_k)
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
                key = (row["brand_normalized"], row["model_normalized"],
                       row.get("brand_latin", ""), row.get("model_latin", ""))
                brand_source = (row.get("brand_raw") if row["brand_normalized"]
                                else row["brand_normalized"])
                model_source = (row.get("model_raw") if row["model_normalized"]
                                else row["model_normalized"])
                if key not in cache:
                    cache[key] = matcher.resolve(
                        brand_source, model_source, brand_latin=row.get("brand_latin"),
                        model_latin=row.get("model_latin"),
                    )
                decision = cache[key]
                result = {name: row[name] for name in OUTPUT_COLUMNS[:6]}
                result.update({
                    "corrected_brand": (normalize.normalize_text(decision.corrected_brand)
                                        or row["brand_normalized"]).upper(),
                    "corrected_model": (normalize.normalize_text(decision.corrected_model)
                                        or row["model_normalized"]).upper(),
                    "suggested_brand": matcher.suggested_identity(decision)[0],
                    "suggested_model": matcher.suggested_identity(decision)[1],
                    "vehicle_type": decision.vehicle_type if public_status(decision) == "MATCHED" else "",
                    "match_status": decision.match_status,
                    "result_status": public_status(decision),
                    "match_method": decision.match_method,
                    "match_score": decision.match_score,
                    "brand_score": decision.top_candidates[0].brand_score if decision.top_candidates else "",
                    "model_score": decision.top_candidates[0].model_score if decision.top_candidates else "",
                    "score_margin": round(decision.top_candidates[0].score - decision.top_candidates[1].score, 1)
                    if len(decision.top_candidates) > 1 else "",
                    "confidence_level": decision.confidence_level,
                    "candidate_count": decision.candidate_count,
                    "top_candidates": matcher.serialize_candidates(decision),
                    "quality_flags": row["quality_flags"],
                    "brand_latin": comparison_text(brand_source, row.get("brand_latin")),
                    "model_latin": comparison_text(model_source, row.get("model_latin")),
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
    parser.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE)
    parser.add_argument("--min-margin", type=float, default=DEFAULT_MIN_MARGIN)
    parser.add_argument("--min-brand", type=float, default=DEFAULT_MIN_BRAND)
    parser.add_argument("--min-model", type=float, default=DEFAULT_MIN_MODEL)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    args = parser.parse_args()
    try:
        report = run(args.input, args.catalog, args.brand_aliases, args.output,
                     args.min_score, args.min_margin, args.min_brand, args.min_model,
                     args.top_k)
    except (OSError, ValueError, csv.Error) as exc:
        print(f"Matching failed: {exc}", file=sys.stderr)
        return 1
    print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
