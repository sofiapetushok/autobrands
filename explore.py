"""Step 1: exploratory profiling of raw insurance vehicle descriptions.
Run: python explore.py [path/to/claims.csv]
Only reads the CSV; never changes any claim or guesses make/model.
"""
import csv
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

DEFAULT_FILE = Path(__file__).resolve().parent / 'data' / 'demo_claims.csv'
NULL_TOKENS = {'', 'NULL', 'NONE', 'N/A', 'NA', 'NAN', 'UNKNOWN', 'ΑΓΝΩΣΤΟ', '-', '--', '---', '???'}

def norm(value):
    return unicodedata.normalize('NFKC', value or '').strip()

def is_missing(value):
    return norm(value).upper() in NULL_TOKENS

def pct(n, total):
    return f'{n / total * 100:.1f}%' if total else '0.0%'

def top(counter, n=12):
    for value, count in counter.most_common(n):
        print(f'  {count:>5}  {value}')

def greek_and_latin(value):
    greek = bool(re.search(r'[Α-Ωα-ωΆ-Ώά-ώ]', value))
    latin = bool(re.search(r'[A-Za-z]', value))
    return greek and latin

def has_noise(value):
    return bool(re.search(r'[@#?*_]{2,}|[\t]|-{3,}|/{2,}', value))

def main(path):
    if not path.exists():
        raise SystemExit(f'CSV not found: {path}')
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        required = {'claim_id', 'brand', 'model'}
        if not required.issubset(reader.fieldnames or []):
            raise SystemExit(f'Missing columns: {required - set(reader.fieldnames or [])}')
        rows = list(reader)

    count = len(rows)
    brands = Counter()
    models = Counter()
    raw_pairs = Counter()
    normalized_pairs = Counter()
    missing_both = missing_brand = missing_model = 0
    mixed_script = noisy = digits = long_values = 0
    brand_contains_space = model_contains_space = 0
    ids = Counter()
    lengths = Counter()

    for row in rows:
        claim_id = norm(row['claim_id'])
        brand, model = norm(row['brand']), norm(row['model'])
        bm, mm = is_missing(brand), is_missing(model)
        ids[claim_id] += 1
        raw_pairs[(brand, model)] += 1
        normalized_pairs[(brand.casefold(), model.casefold())] += 1
        if bm: missing_brand += 1
        if mm: missing_model += 1
        if bm and mm: missing_both += 1
        if not bm: brands[brand] += 1
        if not mm: models[model] += 1
        if not bm and ' ' in brand: brand_contains_space += 1
        if not mm and ' ' in model: model_contains_space += 1
        full = f'{brand} {model}'
        mixed_script += greek_and_latin(full)
        noisy += has_noise(full)
        digits += bool(re.search(r'\d', full))
        long_values += len(full) > 45
        lengths[min(len(full) // 10 * 10, 100)] += 1

    print(f'\nVEHICLE CLAIMS DATA PROFILING\nFile: {path.name}')
    print('=' * 60)
    print(f'Rows: {count:,}')
    print(f'Unique claim IDs: {len(ids):,}')
    print(f'Duplicate claim IDs: {sum(n-1 for n in ids.values() if n > 1):,}')
    print(f'Unique raw brand/model pairs: {len(raw_pairs):,}')
    print(f'Unique case-folded pairs: {len(normalized_pairs):,}')
    print(f'Repeated rows by brand/model: {count-len(raw_pairs):,}')
    print('\nMISSINGNESS (counts include explicit placeholders):')
    print(f'  Missing brand: {missing_brand:,} ({pct(missing_brand,count)})')
    print(f'  Missing model: {missing_model:,} ({pct(missing_model,count)})')
    print(f'  Missing both:  {missing_both:,} ({pct(missing_both,count)})')
    print(f'  Both present: {count-missing_brand-missing_model+missing_both:,}')
    print('\nPOTENTIAL TEXT QUALITY FLAGS (not errors by themselves):')
    for label, n in [
        ('Greek + Latin mixed in same record', mixed_script),
        ('Unusual symbols', noisy),
        ('Contains digits', digits),
        ('Long combined text >45 chars', long_values),
        ('Multi-word brand field', brand_contains_space),
        ('Multi-word model field', model_contains_space),
    ]:
        print(f'  {label}: {n:,} ({pct(n,count)})')
    print(f'\nUnique nonmissing brand values: {len(brands):,}')
    print('TOP BRANDS (raw):')
    top(brands)
    print(f'\nUnique nonmissing model values: {len(models):,}')
    print('TOP MODELS (raw):')
    top(models)
    print('\nTOP BRAND/MODEL PAIRS:')
    top(Counter({f'{b!r} | {m!r}': n for (b,m),n in raw_pairs.items()}))
    print('\nExamples with missing brand but a model:')
    sample = [(b,m) for b,m in raw_pairs if is_missing(b) and not is_missing(m)]
    for b,m in sample[:15]: print(f'  brand={b!r} model={m!r}')
    print('\nNOTE: These are descriptive flags, not vehicle classifications.')
    print('No reference catalog or ground truth is used at this stage.\n')

if __name__ == '__main__':
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_FILE)