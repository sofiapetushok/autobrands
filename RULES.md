# AutoBrands matching rules

1. **Input cleanup:** `normalize_text` uses NFC, uppercasing, whitespace cleanup, and equivalent dash/quote forms. The Greek/Latin script and meaningful punctuation remain in the normalized text.
2. **Comparison views:** the matcher may compare original, accent-folded, *visual-only* Greek/Latin substitutions (`Β→B`, `Η→H`, `Ρ→P`), and separate Greek phonetic substitutions (`Β→V`, `Ρ→R`, digraphs such as `ΟΥ→U`). These are alternatives, not destructive edits.
3. **Catalog is authoritative:** only canonical brands/models and explicit aliases in the CSV can become accepted identities. Do not add name-specific rules to code.
4. **Field independence:** both supplied text values are searched for brand, model and brand+model evidence regardless of source-column order. The notebook's `brand_col` and `model_col` specify where input comes from.
5. **Retrieval:** bounded n-gram indexes nominate candidates; RapidFuzz ranks the nomination set; a Unicode-aware weighted edit score evaluates shortlisted identities. This bounded approach may miss true candidates.
6. **Scoring:** scores are ranking similarities, **not probabilities**. Punctuation spacing is normalized before scoring, but meaningful punctuation remains significant. An exact/alias/format/Unicode-equivalent value can score 100; a 100-point candidate can still be ambiguous.
7. **Decisions:** accept only sufficiently strong candidates with adequate separation; otherwise `AMBIGUOUS`/`UNRESOLVED`/`PARTIAL_MATCH`. Independently recognized, incompatible brand/model evidence produces `CONFLICT`.
8. **Variant safety:** a trailing `+` may be ignored only when the base model is uniquely identified and no catalog or alias variant competes; otherwise keep it ambiguous.
9. **Output:** accepted identities come from the catalog; unresolved values use a deterministic Latin fallback for compatibility. The original DataFrame is not modified and its index is preserved.

**Known limitations:** The numeric-context expansion is still a separate model-family heuristic; `resolve()` still owns multiple decision branches. More aggressive simplification requires baseline regression evidence, not just fewer lines.
