"""Independent Unicode-aware catalog matcher for AutoBrands V2.

V2 keeps normalized Unicode unchanged, retrieves candidates with bounded
catalog indexes, and applies weighted edit distance only to those candidates.
``match_score`` is the top candidate's ranking score, not a probability or an
acceptance-confidence estimate. Exact canonical, explicit-alias, spacing and
Unicode-equivalent comparisons can score 100. ``match_status`` and candidate
margin determine whether that top candidate was safe to accept; an ambiguity
can therefore correctly have a score of 100.
"""

from __future__ import annotations

import csv
import re
import time
import unicodedata
from collections import Counter, defaultdict, OrderedDict
from dataclasses import dataclass
from pathlib import Path

from rapidfuzz import fuzz

from . import normalize

MIN_BRAND_SCORE = 80.0
MIN_MODEL_SCORE = 80.0
MIN_PAIR_SCORE = 80.0
MIN_MARGIN = 8.0
REVIEW_SCORE = 72.0
MAX_CANDIDATES = 120
MAX_BRAND_CANDIDATES = 32
MAX_SCORE_CACHE = 20000

# Weighted edit costs are internal decision parameters, not probabilities.
EDIT_COSTS = {
    "exact": 0.0,
    "visual_mixed": 0.08,
    "visual_greek": 0.10,
    "phonetic_mixed": 0.34,
    "phonetic_greek": 0.08,
    "ordinary": 1.0,
    "transposition": 0.72,
}

_NUMBER_RE = re.compile(r"\d+")
_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)
_SIGNIFICANT_PUNCTUATION = frozenset("+/")


def _script_counts(value: str) -> tuple[int, int]:
    latin = greek = 0
    for char in value:
        name = unicodedata.name(char, "")
        latin += name.startswith("LATIN ")
        greek += name.startswith("GREEK ")
    return latin, greek


def _unicode_edges(source: str, mixed: bool):
    visual_cost = EDIT_COSTS["visual_mixed" if mixed else "visual_greek"]
    phonetic_cost = EDIT_COSTS["phonetic_mixed" if mixed else "phonetic_greek"]
    edges = [[] for _ in range(len(source))]
    for i, char in enumerate(source):
        if not unicodedata.name(char, "").startswith("GREEK "):
            continue
        visual = normalize.VISUAL_CONFUSABLES.get(char)
        phonetic = normalize.GREEK_LETTERS.get(char)
        if visual:
            edges[i].append((1, visual, visual_cost))
        if phonetic:
            edges[i].append((1, phonetic, phonetic_cost))
        for greek, latin in normalize.GREEK_DIGRAPHS:
            if source.startswith(greek, i):
                edges[i].append((len(greek), latin, phonetic_cost))
    return edges


def _directed_distance(source: str, target: str) -> float:
    latin, greek = _script_counts(source)
    mixed = bool(latin and greek)
    edges = _unicode_edges(source, mixed)
    n, m = len(source), len(target)
    dp = [[float("inf")] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0
    for i in range(n + 1):
        for j in range(m + 1):
            cost = dp[i][j]
            if cost == float("inf"):
                continue
            if i < n:
                dp[i + 1][j] = min(dp[i + 1][j], cost + EDIT_COSTS["ordinary"])
            if j < m:
                dp[i][j + 1] = min(dp[i][j + 1], cost + EDIT_COSTS["ordinary"])
            if i < n and j < m:
                substitution = EDIT_COSTS["exact"] if source[i] == target[j] else EDIT_COSTS["ordinary"]
                dp[i + 1][j + 1] = min(dp[i + 1][j + 1], cost + substitution)
            if i + 1 < n and j + 1 < m and source[i] == target[j + 1] and source[i + 1] == target[j]:
                dp[i + 2][j + 2] = min(
                    dp[i + 2][j + 2], cost + EDIT_COSTS["transposition"]
                )
            if i < n:
                for consumed, rendered, edge_cost in edges[i]:
                    if target.startswith(rendered, j):
                        dp[i + consumed][j + len(rendered)] = min(
                            dp[i + consumed][j + len(rendered)], cost + edge_cost
                        )
    return dp[n][m]


def weighted_similarity(left: str, right: str) -> float:
    """Return deterministic 0–100 weighted similarity for two Unicode strings."""
    source = normalize.normalize_text(left) or ""
    target = normalize.normalize_text(right) or ""
    if not source or not target:
        return 0.0
    if _exact_comparison_equivalence(source, target):
        return 100.0
    # Evaluate the small correspondence graph in either direction; no spelling
    # combinations are materialized.
    distance = min(_directed_distance(source, target), _directed_distance(target, source))
    return round(max(0.0, 100.0 * (1.0 - distance / max(len(source), len(target)))), 1)


def _numbers(value: str) -> tuple[str, ...]:
    return tuple(_NUMBER_RE.findall(value))


def _compact(value: str) -> str:
    return "".join(char for char in value if char.isalnum())


def _punctuation(value: str) -> tuple[str, ...]:
    return tuple(char for char in value if char in _SIGNIFICANT_PUNCTUATION)


def _format_equivalent(left: str, right: str) -> bool:
    """Treat ordinary word spacing as formatting, but reject letter-by-letter splits."""
    left_key = normalize.matching_form(left)
    right_key = normalize.matching_form(right)
    if (_compact(left_key) != _compact(right_key)
            or _punctuation(left_key) != _punctuation(right_key)):
        return False
    for value in (left_key, right_key):
        letter_singletons = sum(
            len(token) == 1 and any(char.isalpha() for char in token)
            for token in value.split()
        )
        if letter_singletons > 1:
            return False
    return True


def _forms_equivalent(left: str, right: str) -> bool:
    return bool(set(normalize.comparison_forms(left))
                & set(normalize.comparison_forms(right)))


def _exact_comparison_equivalence(left: str, right: str) -> bool:
    left_key = normalize.matching_form(left)
    right_key = normalize.matching_form(right)
    return (
        left_key == right_key
        or _format_equivalent(left_key, right_key)
        or (_punctuation(left_key) == _punctuation(right_key)
            and _forms_equivalent(left, right))
    )


def _grams(value: str) -> frozenset[str]:
    text = _compact(normalize.matching_form(value))
    if not text:
        return frozenset()
    size = 3 if len(text) >= 3 else 2 if len(text) == 2 else 1
    padded = "^" + text + "$"
    return frozenset(padded[i:i + size] for i in range(len(padded) - size + 1))


def _words(value: str) -> tuple[str, ...]:
    return tuple(_WORD_RE.findall(normalize.normalize_text(value) or ""))


@dataclass(frozen=True)
class Identity:
    brand: str
    model: str
    brand_key: str
    model_key: str
    vehicle_type: str
    brand_aliases: tuple[str, ...]
    model_aliases: tuple[str, ...]

    @property
    def brand_terms(self):
        return tuple(dict.fromkeys((self.brand, *self.brand_aliases)))

    @property
    def model_terms(self):
        return tuple(dict.fromkeys((self.model, *self.model_aliases)))

    @property
    def phrases(self):
        return tuple(f"{a} {b}" for brand in self.brand_terms
                     for model in self.model_terms for a, b in ((brand, model), (model, brand)))


class Matcher:
    """Catalog-backed V2 matcher with an independent pandas DataFrame API."""

    def __init__(self, catalog_rows, *, top_k=5):
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        self.top_k = top_k
        self.min_brand = MIN_BRAND_SCORE
        self.min_model = MIN_MODEL_SCORE
        self.min_score = MIN_PAIR_SCORE
        self.min_margin = MIN_MARGIN
        self.stats = Counter()
        self._score_cache = OrderedDict()
        started = time.perf_counter()
        grouped = defaultdict(lambda: {"rows": [], "brands": set(), "models": set(), "types": set()})
        for row in catalog_rows:
            brand = row.get("brand", "")
            model = row.get("model", "")
            brand_key = normalize.normalize_text(brand) or ""
            model_key = normalize.normalize_text(model) or ""
            if not brand_key or not model_key:
                continue
            group = grouped[(brand_key, model_key)]
            group["rows"].append(row)
            group["brands"].update(self._split_aliases(row.get("brand_aliases", "")))
            group["models"].update(self._split_aliases(row.get("model_aliases", "")))
            group["types"].add(row.get("vehicle_type", "unknown") or "unknown")

        self.identities = []
        self.by_pair = {}
        self.by_brand = defaultdict(set)
        self.brand_lookup = defaultdict(set)
        self.brand_alias_lookup = defaultdict(set)
        self.model_lookup = defaultdict(set)
        self.full_lookup = defaultdict(set)
        self.brand_gram_index = defaultdict(set)
        self.model_gram_index = defaultdict(set)
        self.full_gram_index = defaultdict(set)
        self.brand_choices = set()
        for key in sorted(grouped):
            group = grouped[key]
            preferred = min(group["rows"], key=lambda item: (item["brand"], item["model"]))
            identity = Identity(
                brand=preferred["brand"], model=preferred["model"],
                brand_key=key[0], model_key=key[1],
                vehicle_type=(next(iter(group["types"])) if len(group["types"]) == 1 else "unknown"),
                brand_aliases=tuple(sorted(group["brands"])),
                model_aliases=tuple(sorted(group["models"])),
            )
            identity_id = len(self.identities)
            self.identities.append(identity)
            self.by_pair[key] = identity_id
            self.by_brand[key[0]].add(identity_id)
            self.brand_choices.add(key[0])
            self._index_identity(identity_id, identity)
        self.load_seconds = time.perf_counter() - started
        self.identity_count = len(self.identities)

    @staticmethod
    def _split_aliases(value):
        if not value:
            return ()
        return tuple(part.strip() for part in value.split("|") if part.strip())

    @classmethod
    def from_csv(cls, catalog_path, *, top_k=5):
        rows = []
        with Path(catalog_path).open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            required = {"brand", "model", "vehicle_type"}
            missing = required - set(reader.fieldnames or ())
            if missing:
                raise ValueError(f"Catalog is missing columns: {sorted(missing)}")
            rows.extend(reader)
        return cls(rows, top_k=top_k)

    def _forms(self, value):
        return normalize.comparison_forms(value)

    def _index_identity(self, identity_id, identity):
        brand_terms = identity.brand_terms
        model_terms = identity.model_terms
        for term in brand_terms:
            key = normalize.normalize_text(term) or ""
            self.brand_lookup[key].add(identity.brand_key)
            if key != identity.brand_key:
                self.brand_alias_lookup[key].add(identity.brand_key)
            for form in self._forms(term):
                for gram in _grams(form):
                    self.brand_gram_index[gram].add(identity.brand_key)
        for term in model_terms:
            key = normalize.normalize_text(term) or ""
            self.model_lookup[key].add(identity_id)
            for form in self._forms(term):
                for gram in _grams(form):
                    self.model_gram_index[gram].add(identity_id)
        for phrase in identity.phrases:
            key = normalize.matching_form(phrase)
            self.full_lookup[key].add(identity_id)
            for form in self._forms(phrase):
                for gram in _grams(form):
                    self.full_gram_index[gram].add(identity_id)

    def _retrieve(self, value, exact_index, gram_index, *, limit):
        forms = self._forms(value)
        # Comparison alternatives improve retrieval but must still pass through
        # ranking and margin checks; only raw normalized text is truly exact.
        raw_key = normalize.normalize_text(value) or ""
        exact = set(exact_index.get(raw_key, ()))
        overlap = Counter()
        for form in forms:
            for gram in _grams(form):
                for identity_id in gram_index.get(gram, ()):
                    overlap[identity_id] += 1
        ordered = sorted(overlap, key=lambda item: (-overlap[item], item))
        self.stats["candidate_pool_before_limit"] += len(set(ordered) | exact)
        pool = set(ordered[:limit]) | exact
        query_forms = forms
        choices = {}
        for item in pool:
            if exact_index is self.brand_lookup:
                terms = self.identities[min(self.by_brand[item])].brand_terms
            else:
                identity = self.identities[item]
                terms = identity.phrases if exact_index is self.full_lookup else identity.model_terms
            choices[item] = max(
                (fuzz.ratio(query_form, term_form)
                 for query_form in query_forms for term in terms
                 for term_form in self._forms(term)), default=0.0)
        # RapidFuzz orders conventional candidates after bounded gram retrieval;
        # Unicode alternatives remain in the candidate pool and are scored by
        # the Unicode-aware ranker below.
        selected = set(sorted(pool, key=lambda item: (-choices[item],
                                                       -overlap.get(item, 0), item))[:limit]) | exact
        return selected

    def _brand_candidates(self, value):
        candidates = self._retrieve(
            value, self.brand_lookup, self.brand_gram_index, limit=MAX_BRAND_CANDIDATES
        )
        return candidates

    def _model_candidates(self, value, brand_key=""):
        candidates = self._retrieve(
            value, self.model_lookup, self.model_gram_index, limit=MAX_CANDIDATES
        )
        return {identity_id for identity_id in candidates
                if not brand_key or self.identities[identity_id].brand_key == brand_key}

    def _full_candidates(self, value):
        return self._retrieve(value, self.full_lookup, self.full_gram_index,
                              limit=MAX_CANDIDATES)

    def _score(self, left, right):
        key = (normalize.normalize_text(left) or "", normalize.normalize_text(right) or "")
        cached = self._score_cache.get(key)
        if cached is not None:
            self.stats["score_cache_hits"] += 1
            self._score_cache.move_to_end(key)
            return cached
        self.stats["weighted_comparisons"] += 1
        score = weighted_similarity(left, right)
        self._score_cache[key] = score
        if len(self._score_cache) > MAX_SCORE_CACHE:
            self._score_cache.popitem(last=False)
        return score

    def _best_choice(self, query, choices):
        scored = []
        for choice in choices:
            score = (100.0 if _exact_comparison_equivalence(query, choice)
                     else self._score(query, choice))
            scored.append((score, choice))
        return max(scored, key=lambda item: (item[0], item[1])) if scored else (0.0, "")

    def _best_score(self, query, choices):
        return self._best_choice(query, choices)[0]

    def _brand_score(self, raw_brand, brand_key):
        return self._best_score(raw_brand, self.identities[min(self.by_brand[brand_key])].brand_terms)

    def _rank_brands(self, value):
        return sorted(((self._brand_score(value, key), key)
                       for key in self._brand_candidates(value)),
                      key=lambda item: (-item[0], item[1]))

    def _model_score(self, raw_model, identity_id):
        identity = self.identities[identity_id]
        return self._best_score(raw_model, identity.model_terms)

    def _numeric_context_candidates(self, raw_model, candidate_ids, brand_key=""):
        numbers = _numbers(raw_model)
        if not numbers or not candidate_ids:
            return set(candidate_ids), {}
        query_base = " ".join(part for part in _NUMBER_RE.sub(" ", normalize.normalize_text(raw_model) or "").split())
        base_scores = {}
        for identity_id in candidate_ids:
            identity = self.identities[identity_id]
            base_scores[identity_id] = self._model_score(query_base, identity_id)
        expanded = set(candidate_ids)
        context_scores = {}
        for base_id, base_score in base_scores.items():
            base = self.identities[base_id]
            if base_score < MIN_MODEL_SCORE:
                continue
            prefix = base.model_key + " "
            for possible_id in self.by_brand[base.brand_key]:
                possible = self.identities[possible_id]
                if possible_id == base_id or _numbers(possible.model_key) != numbers:
                    continue
                if possible.model_key.startswith(prefix):
                    expanded.add(possible_id)
                    context_scores[possible_id] = max(context_scores.get(possible_id, 0.0),
                                                      min(97.0, base_score))
        return expanded, context_scores

    def _candidate_output(self, ranked, *, brand_only=False):
        candidates = []
        for item in ranked[:self.top_k]:
            identity = self.identities[item["identity_id"]]
            candidate = {"brand": normalize.latin_fallback(identity.brand),
                         "score": item["score"]}
            if not brand_only:
                candidate["model"] = normalize.latin_fallback(identity.model)
                candidate = {"brand": candidate["brand"], "model": candidate["model"],
                             "score": item["score"]}
            candidates.append(candidate)
        return candidates

    @staticmethod
    def _full_phrase_kind(query, identity):
        canonical = (f"{identity.brand} {identity.model}",
                     f"{identity.model} {identity.brand}")
        query_key = normalize.matching_form(query)
        if query_key in {normalize.matching_form(p) for p in canonical}:
            return "EXACT_CANONICAL"
        if query_key in {normalize.matching_form(p) for p in identity.phrases}:
            return "EXPLICIT_ALIAS"
        if any(_format_equivalent(query_key, phrase) for phrase in identity.phrases):
            return "FORMATTING_EQUIVALENCE"
        if any(_forms_equivalent(query, phrase) for phrase in identity.phrases):
            return "UNICODE_EQUIVALENCE"
        return "ORDINARY_FUZZY"

    @staticmethod
    def _status_result(corrected_brand, corrected_model, status, score, candidates,
                       *, brand_only=False, debug=None):
        result = {
            "corrected_brand": corrected_brand,
            "corrected_model": corrected_model,
            "match_status": status,
            "match_score": float(score) if score is not None else None,
            "candidates": candidates,
        }
        if brand_only:
            result.pop("corrected_model", None)
        if debug:
            result.update(debug)
        return result

    def resolve(self, brand_value=None, model_value=None):
        """Resolve unordered input values against catalog identities."""
        values = [normalize.normalize_text(value) or ""
                  for value in (brand_value, model_value)]
        values = list(dict.fromkeys(value for value in values
                                    if not normalize.is_garbage(value)))
        fallback_brand = (normalize.latin_fallback(brand_value)
                          if not normalize.is_garbage(brand_value) else "")
        fallback_model = (normalize.latin_fallback(model_value)
                          if not normalize.is_garbage(model_value) else "")
        if not values:
            return self._status_result(fallback_brand, fallback_model,
                                       "UNRESOLVED", None, [])

        # A single shared retrieval path gathers brand, model, and complete-pair
        # evidence for every populated field, independent of its column name.
        brand_ids, model_ids, full_ids = set(), set(), set()
        for value in values:
            brand_ids.update(self._brand_candidates(value))
            model_ids.update(self._model_candidates(value))
            if len(_words(value)) > 1:
                full_ids.update(self._full_candidates(value))
        candidate_ids = set(model_ids) | set(full_ids)
        if brand_ids:
            candidate_ids.update(min(self.by_brand[key]) for key in brand_ids)
        numeric_scores = {}
        for value in values:
            if _numbers(value):
                bases = self._model_candidates(_NUMBER_RE.sub(" ", value))
                expanded, contextual = self._numeric_context_candidates(value, bases)
                candidate_ids.update(expanded)
                numeric_scores.update(contextual)
        if not candidate_ids:
            return self._status_result(fallback_brand, fallback_model,
                                       "UNRESOLVED", None, [])

        if len(values) == 1:
            value = values[0]
            brand_rank = self._rank_brands(value)
            model_hit = any(self._model_score(value, item) >= MIN_MODEL_SCORE
                            for item in self._model_candidates(value))
            if (brand_rank and brand_rank[0][0] >= MIN_BRAND_SCORE
                    and brand_rank[0][0] - (brand_rank[1][0] if len(brand_rank) > 1 else 0)
                    >= MIN_MARGIN and not model_hit):
                brand_key = brand_rank[0][1]
                representative = min(self.by_brand[brand_key])
                brand_rows = [{"identity_id": min(self.by_brand[key]), "score": score}
                              for score, key in brand_rank]
                return self._status_result(
                    normalize.latin_fallback(self.identities[representative].brand), "",
                    "MATCHED", brand_rank[0][0],
                    self._candidate_output(brand_rows, brand_only=True), brand_only=True)

        # Score each identity from the strongest valid assignment of available
        # values. One value can represent a brand, model/alias, or full phrase.
        ranked = []
        brand_score_cache = {}
        for identity_id in candidate_ids:
            identity = self.identities[identity_id]
            brand_scores = []
            for value in values:
                cache_key = (value, identity.brand_key)
                if cache_key not in brand_score_cache:
                    brand_score_cache[cache_key] = self._brand_score(value, identity.brand_key)
                brand_scores.append(brand_score_cache[cache_key])
            model_scores = [self._model_score(value, identity_id)
                            for value in values]
            model_scores = [max(score, numeric_scores.get(identity_id, 0.0))
                            for score in model_scores]
            phrase_scores = [self._best_score(value, identity.phrases)
                             for value in values if len(_words(value)) > 1]
            best_brand = max(brand_scores, default=0.0)
            best_model = max(model_scores, default=0.0)
            best_phrase = max(phrase_scores, default=0.0)
            pair_score = 0.0
            pair_brand = pair_model = 0.0
            if len(values) > 1:
                for i, first in enumerate(values):
                    for j, second in enumerate(values):
                        if i == j:
                            continue
                        b_score = self._brand_score(first, identity.brand_key)
                        m_score = max(self._model_score(second, identity_id),
                                      numeric_scores.get(identity_id, 0.0))
                        score = 0.35 * b_score + 0.65 * m_score
                        if score > pair_score:
                            pair_score, pair_brand, pair_model = score, b_score, m_score
            score = (max(best_phrase, pair_score) if len(values) > 1
                     else max(best_phrase, best_model, best_brand))
            b_score = pair_brand if len(values) > 1 else best_brand
            m_score = max(pair_model, best_phrase) if len(values) > 1 else max(best_model, best_phrase)
            _, model_term = self._best_choice(
                values[model_scores.index(max(model_scores))], identity.model_terms
            ) if identity.model_terms else (0.0, "")
            ranked.append({"identity_id": identity_id, "score": round(score, 1),
                           "brand_score": round(b_score, 1),
                           "model_score": round(m_score, 1),
                           "model_term": model_term})
        ranked.sort(key=lambda item: (-item["score"], -item["model_score"],
                                      self.identities[item["identity_id"]].brand_key,
                                      self.identities[item["identity_id"]].model_key))
        self.stats["candidate_ids_retrieved"] += len(candidate_ids)
        self.stats["rows_resolved"] += 1
        top = ranked[0]
        margin = top["score"] - (ranked[1]["score"] if len(ranked) > 1 else 0.0)
        candidates = self._candidate_output(ranked)
        identity = self.identities[top["identity_id"]]

        # Conflicting, independently recognized brand and model evidence wins
        # over a superficially strong joint score.
        recognized_brands = []
        recognized_models = []
        for value in values:
            brand_rank = self._rank_brands(value)
            if brand_rank and brand_rank[0][0] >= MIN_BRAND_SCORE:
                gap = brand_rank[0][0] - (brand_rank[1][0] if len(brand_rank) > 1 else 0)
                if gap >= MIN_MARGIN:
                    recognized_brands.append(brand_rank[0][1])
            exact_models = {item for item in self._model_candidates(value)
                            if self._model_score(value, item) >= MIN_MODEL_SCORE}
            if exact_models:
                recognized_models.extend(exact_models)
        if (len(set(recognized_brands)) == 1 and recognized_models
                and not any(self.identities[item].brand_key in recognized_brands
                            for item in recognized_models)):
            return self._status_result(fallback_brand, fallback_model, "CONFLICT",
                                       top["score"], candidates)
        if len(set(recognized_brands)) == 1:
            consistent = [row for row in ranked
                          if self.identities[row["identity_id"]].brand_key
                          == recognized_brands[0]]
            if consistent:
                ranked = consistent
                top = ranked[0]
                identity = self.identities[top["identity_id"]]
                margin = top["score"] - (ranked[1]["score"] if len(ranked) > 1 else 0.0)
                candidates = self._candidate_output(ranked)

        model_inputs = [(value, max(self._model_score(value, top["identity_id"]),
                                    numeric_scores.get(top["identity_id"], 0.0)))
                        for value in values]
        model_value, model_score = max(model_inputs, key=lambda item: item[1])
        brand_value_text, brand_score = max(
            ((value, self._brand_score(value, identity.brand_key)) for value in values),
            key=lambda item: item[1], default=("", 0.0))
        phrase_options = [(value, self._best_score(value, identity.phrases))
                          for value in values if len(_words(value)) > 1]
        phrase_value, phrase_score = max(phrase_options, key=lambda item: item[1],
                                         default=("", 0.0))
        accepted_identity = None
        method = ""
        if phrase_value and phrase_score >= MIN_PAIR_SCORE:
            _, best_phrase_term = self._best_choice(phrase_value, identity.phrases)
            phrase_safe = (_punctuation(phrase_value) == _punctuation(best_phrase_term)
                           and (not _numbers(phrase_value)
                                or _numbers(phrase_value) == _numbers(identity.model_key))
                           and len(phrase_value.split()) == len(best_phrase_term.split()))
            if phrase_safe and margin >= MIN_MARGIN:
                accepted_identity = identity
                method = self._full_phrase_kind(phrase_value, identity)
        elif len(values) > 1:
            model_is_safe = (model_score >= MIN_MODEL_SCORE and margin >= MIN_MARGIN
                             and (not _numbers(model_value)
                                  or _numbers(model_value) == _numbers(identity.model_key))
                             and (_punctuation(model_value) == _punctuation(identity.model_key)
                                  or _punctuation(model_value) == _punctuation(top["model_term"])))
            brand_safe = brand_score >= MIN_BRAND_SCORE
            if not brand_safe and model_score >= MIN_MODEL_SCORE + MIN_MARGIN:
                brand_rank = sorted(
                    ((self._brand_score(value, key), key)
                     for value in values for key in self._brand_candidates(value)),
                    key=lambda item: (-item[0], item[1]),
                )
                brand_safe = bool(
                    brand_rank and brand_rank[0][1] == identity.brand_key
                    and brand_rank[0][0] >= MIN_BRAND_SCORE - MIN_MARGIN
                    and brand_rank[0][0] - (brand_rank[1][0] if len(brand_rank) > 1 else 0)
                    >= MIN_MARGIN
                )
            if model_is_safe and brand_safe and top["score"] >= MIN_PAIR_SCORE:
                accepted_identity = identity
                methods = {
                    self._component_method(brand_value_text, identity.brand,
                                          identity.brand_aliases),
                    self._component_method(model_value, identity.model,
                                          identity.model_aliases),
                }
                method = ("ORDINARY_FUZZY" if "ORDINARY_FUZZY" in methods else
                          "EXPLICIT_ALIAS" if "EXPLICIT_ALIAS" in methods else
                          "UNICODE_EQUIVALENCE" if "UNICODE_EQUIVALENCE" in methods else
                          "FORMATTING_EQUIVALENCE" if "FORMATTING_EQUIVALENCE" in methods else
                          "EXACT_CANONICAL")
        else:
            evidence = values[0]
            if model_score >= MIN_MODEL_SCORE and margin >= MIN_MARGIN:
                if (not _numbers(evidence) or _numbers(evidence) == _numbers(identity.model_key)) and (
                        _punctuation(evidence) == _punctuation(identity.model_key)
                        or _punctuation(evidence) == _punctuation(top["model_term"])):
                    accepted_identity = identity
                    method = self._component_method(evidence, identity.model,
                                                    identity.model_aliases)
            elif brand_score >= MIN_BRAND_SCORE and margin >= MIN_MARGIN:
                method = self._component_method(evidence, identity.brand,
                                                identity.brand_aliases)
                # Brand-only evidence identifies no model; choose the stable
                # first catalog row for candidate display, but do not correct it.
                return self._status_result(normalize.latin_fallback(identity.brand), "",
                                           "MATCHED", top["score"], candidates,
                                           brand_only=True)

        if accepted_identity:
            identity = accepted_identity
            status = {"EXACT_CANONICAL": "EXACT_MATCH", "EXPLICIT_ALIAS": "ALIAS_MATCH",
                      "FORMATTING_EQUIVALENCE": "MATCHED", "UNICODE_EQUIVALENCE": "MATCHED",
                      "ORDINARY_FUZZY": "FUZZY_MATCH"}.get(method, "FUZZY_MATCH")
            return self._status_result(normalize.latin_fallback(identity.brand),
                                       normalize.latin_fallback(identity.model), status,
                                       top["score"], candidates)

        if top["score"] >= REVIEW_SCORE and margin < MIN_MARGIN:
            return self._status_result(fallback_brand, fallback_model, "AMBIGUOUS",
                                       top["score"], candidates)
        if brand_score >= MIN_BRAND_SCORE:
            return self._status_result(normalize.latin_fallback(identity.brand),
                                       fallback_model, "PARTIAL_MATCH", top["score"], candidates)
        status = "AMBIGUOUS" if top["score"] >= REVIEW_SCORE else "UNRESOLVED"
        return self._status_result(fallback_brand, fallback_model, status,
                                   top["score"], candidates)

    def _decision_match_method(self, brand_value, model_value, decision):
        status = decision["match_status"]
        accepted = {"MATCHED", "EXACT_MATCH", "ALIAS_MATCH", "FUZZY_MATCH"}
        if status not in accepted:
            return status
        corrected_brand = normalize.normalize_text(decision.get("corrected_brand")) or ""
        corrected_model = normalize.normalize_text(decision.get("corrected_model")) or ""
        identity_id = self.by_pair.get((corrected_brand, corrected_model))
        if identity_id is None and corrected_brand:
            brand_ids = self.by_brand.get(corrected_brand, set())
            identity_id = min(brand_ids) if brand_ids else None
        if identity_id is None:
            return "ORDINARY_FUZZY"
        identity = self.identities[identity_id]
        values = [normalize.normalize_text(value) or ""
                  for value in (brand_value, model_value)
                  if not normalize.is_garbage(value)]
        if len(values) == 1:
            value = values[0]
            if len(_words(value)) > 1:
                return self._full_phrase_kind(value, identity)
            if self._brand_score(value, identity.brand_key) >= self._model_score(value, identity_id):
                return self._component_method(value, identity.brand, identity.brand_aliases)
            return self._component_method(value, identity.model, identity.model_aliases)
        if len(values) == 2:
            assignment = max(
                ((self._brand_score(brand, identity.brand_key)
                  + self._model_score(model, identity_id), brand, model)
                 for brand, model in ((values[0], values[1]), (values[1], values[0]))),
                key=lambda item: item[0],
            )
            evidence = [
                self._component_method(assignment[1], identity.brand, identity.brand_aliases),
                self._component_method(assignment[2], identity.model, identity.model_aliases),
            ]
        else:
            return status
        if "ORDINARY_FUZZY" in evidence:
            return "ORDINARY_FUZZY"
        if "EXPLICIT_ALIAS" in evidence:
            return "EXPLICIT_ALIAS"
        if "UNICODE_EQUIVALENCE" in evidence:
            return "UNICODE_EQUIVALENCE"
        if "FORMATTING_EQUIVALENCE" in evidence:
            return "FORMATTING_EQUIVALENCE"
        return "EXACT_CANONICAL"

    @staticmethod
    def _component_method(value, canonical, aliases):
        value = normalize.normalize_text(value) or ""
        canonical_key = normalize.normalize_text(canonical) or ""
        if value == canonical_key:
            return "EXACT_CANONICAL"
        if value in {normalize.normalize_text(alias) or "" for alias in aliases}:
            return "EXPLICIT_ALIAS"
        if _format_equivalent(value, canonical_key):
            return "FORMATTING_EQUIVALENCE"
        if _forms_equivalent(value, canonical_key):
            return "UNICODE_EQUIVALENCE"
        return "ORDINARY_FUZZY"

    def match_dataframe(self, df, *, brand_col=None, model_col=None,
                        top_k=None, debug=False):
        """Match raw DataFrame columns while preserving source values, index, and order."""
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("V2 DataFrame matching requires pandas") from exc
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        if not df.columns.is_unique:
            raise ValueError("DataFrame column names must be unique")
        if brand_col is None and model_col is None:
            raise ValueError("Supply brand_col, model_col, or both")
        for name in (brand_col, model_col):
            if name is not None and (not isinstance(name, str) or not name or name not in df.columns):
                raise ValueError(f"Selected input column not found: {name!r}")
        if brand_col is not None and brand_col == model_col:
            raise ValueError("brand_col and model_col must differ")
        if top_k is None:
            top_k = self.top_k
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= self.top_k:
            raise ValueError(f"top_k must be between 1 and {self.top_k}")
        if not isinstance(debug, bool):
            raise TypeError("debug must be bool")

        outputs = []
        decision_cache = {}
        for position in range(len(df)):
            raw_brand = df[brand_col].iloc[position] if brand_col is not None else None
            raw_model = df[model_col].iloc[position] if model_col is not None else None
            cache_key = (normalize.normalize_text(raw_brand), normalize.normalize_text(raw_model))
            decision = decision_cache.get(cache_key)
            if decision is None:
                decision = self.resolve(raw_brand, raw_model)
                decision_cache[cache_key] = decision
            row = {}
            if brand_col is not None:
                row["corrected_brand"] = decision.get("corrected_brand", "")
            if model_col is not None:
                row["corrected_model"] = decision.get("corrected_model", "")
            row.update({
                "match_status": decision["match_status"],
                "match_score": decision["match_score"],
                "candidates": decision["candidates"][:top_k],
            })
            if debug:
                candidates = decision["candidates"]
                score = decision["match_score"]
                margin = (candidates[0]["score"] - candidates[1]["score"]
                          if len(candidates) > 1 else None)
                row.update({
                    "ab_brand_normalized": normalize.normalize_text(raw_brand) or "",
                    "ab_model_normalized": normalize.normalize_text(raw_model) or "",
                    "ab_brand_score": score if brand_col is not None else None,
                    "ab_model_score": score if model_col is not None else None,
                    "ab_score_margin": margin,
                    "ab_match_method": self._decision_match_method(
                        raw_brand, raw_model, decision
                    ),
                })
            outputs.append(row)

        result_columns = []
        if brand_col is not None:
            result_columns.append("corrected_brand")
        if model_col is not None:
            result_columns.append("corrected_model")
        result_columns.extend(["match_status", "match_score", "candidates"])
        debug_columns = (["ab_brand_normalized", "ab_model_normalized", "ab_brand_score",
                          "ab_model_score", "ab_score_margin", "ab_match_method"] if debug else [])
        collisions = set(result_columns + debug_columns) & set(df.columns)
        if collisions:
            raise ValueError(f"Output column name collision: {', '.join(sorted(collisions))}")
        result = df.copy(deep=False)
        for name in result_columns + debug_columns:
            result[name] = [row.get(name) for row in outputs]
        return result


__all__ = ["Matcher", "Identity", "weighted_similarity", "EDIT_COSTS"]
