# AutoBrands V2 matching rules

- **Normalization:** NFC, uppercase, trim and collapse whitespace; preserve source script, digits, and meaningful punctuation. Garbage placeholders provide no matching evidence.
- **Unicode forms:** Matching retains normalized text and compares bounded raw, accent-folded, Greek phonetic, and Greek visual forms. Visual mappings include `Β→B`, `Ρ→P`; phonetic forms retain `Β→V`, `Ρ→R`. These alternatives affect comparison only.
- **Catalog evidence:** Canonical brands, models, and explicit catalog aliases are indexed once. Candidate retrieval uses bounded character-gram indexes, reorders the retrieved pool with RapidFuzz, and ranks candidates with the Unicode-aware weighted edit score.
- **Fields:** Populated values are unordered evidence. Each can support a brand, model/alias, or catalog-supported complete brand–model phrase. A confidently recognized brand narrows the candidate set when model evidence does not contradict it.
- **Variants:** Numeric designations and `+` or `/` punctuation are identity-bearing. Number context can select an indexed numeric family member; punctuation or numbers are not silently discarded to accept a base model.
- **Decisions:** A full match requires the existing component/pair score gates and candidate margin. Independently recognized incompatible brand/model evidence returns `CONFLICT`; close competing identities return `AMBIGUOUS`; a safe brand with an unresolved model may return `PARTIAL_MATCH`.
- **Fallback and score:** Unaccepted fields retain uppercase Latin fallback text (empty for missing/garbage inputs). `match_score` is the top candidate's ranking score, not a probability; even a score of 100 can be ambiguous when identities collide.
