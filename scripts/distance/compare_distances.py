#!/usr/bin/env python3
"""
Compare different distance metrics: Hamming, Gower, Weighted Gower.
"""

import csv

def load_hamming():
    data = {}
    with open('data/processed/hamming_distance_summary.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            lang = row['language']
            data[lang] = {
                'hamming': int(row['hamming_distance']),
                'common': int(row['common_features']),
                'norm': int(row['hamming_distance']) / int(row['common_features']) if int(row['common_features']) > 0 else 0
            }
    return data

def load_gower():
    data = {}
    with open('data/processed/gower_distance_summary.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            lang = row['language']
            data[lang] = {
                'gower': float(row['gower_distance']),
                'common': int(row['common_features']),
                'weighted': float(row['weighted_distance']),
                'total_weight': float(row['total_weight'])
            }
    return data

def main():
    hamming_data = load_hamming()
    gower_data = load_gower()

    # Merge
    all_langs = set(hamming_data.keys()) | set(gower_data.keys())
    merged = []
    for lang in all_langs:
        h = hamming_data.get(lang, {})
        g = gower_data.get(lang, {})
        merged.append({
            'lang': lang,
            'hamming': h.get('hamming'),
            'common_h': h.get('common'),
            'norm': h.get('norm'),
            'gower': g.get('gower'),
            'common_g': g.get('common'),
            'weighted': g.get('weighted'),
            'total_weight': g.get('total_weight')
        })

    # Sort by Gower distance (descending)
    merged.sort(key=lambda x: x['gower'] if x['gower'] is not None else -1, reverse=True)

    # Language names mapping
    lang_names = {
        'th': 'Thai',
        'zu': 'Zulu',
        'gd': 'Scottish Gaelic',
        'gn': 'Guarani',
        'zh-CN': 'Chinese',
        'it': 'Italian',
        'hi': 'Hindi',
        'bn': 'Bengali',
        'hmn': 'Hmong',
        'he': 'Hebrew',
        'uk': 'Ukrainian',
        'ar': 'Arabic'
    }

    print("Comparison of Distance Metrics (from English)")
    print("=" * 90)
    print("{:<12} {:<10} {:<8} {:<8} {:<8} {:<8} {:<8}".format(
        'Language', 'Hamming', 'Common', 'Norm', 'Gower', 'Weighted', 'Common_G'))
    print("-" * 90)
    for m in merged:
        name = lang_names.get(m['lang'], m['lang'])
        hamming = m['hamming'] if m['hamming'] is not None else '-'
        common_h = m['common_h'] if m['common_h'] is not None else '-'
        norm = f"{m['norm']:.3f}" if m['norm'] is not None else '-'
        gower = f"{m['gower']:.3f}" if m['gower'] is not None else '-'
        weighted = f"{m['weighted']:.3f}" if m['weighted'] is not None else '-'
        common_g = m['common_g'] if m['common_g'] is not None else '-'
        print(f"{name:<12} {hamming:<10} {common_h:<8} {norm:<8} {gower:<8} {weighted:<8} {common_g:<8}")

    print("\nNotes:")
    print("1. Hamming: absolute count of mismatching features")
    print("2. Norm: Hamming / Common (equivalent to Gower distance)")
    print("3. Gower: 1 - (matching / common), same as Norm")
    print("4. Weighted: Gower with feature weights based on frequency across languages")
    print("5. Common: number of features present in both English and the language")

    # Check consistency between Norm and Gower
    print("\nConsistency check (Norm vs Gower):")
    diffs = []
    for m in merged:
        if m['norm'] is not None and m['gower'] is not None:
            diff = abs(m['norm'] - m['gower'])
            if diff > 0.001:
                diffs.append((m['lang'], diff))
    if diffs:
        print(f"  Found {len(diffs)} languages with difference > 0.001:")
        for lang, diff in diffs:
            print(f"    {lang}: diff = {diff:.6f}")
    else:
        print("  Norm and Gower are identical (as expected).")

    # Rank correlation between Hamming and Weighted
    from scipy.stats import spearmanr
    hamming_vals = [m['hamming'] for m in merged if m['hamming'] is not None]
    weighted_vals = [m['weighted'] for m in merged if m['weighted'] is not None]
    if len(hamming_vals) == len(weighted_vals) and len(hamming_vals) > 1:
        rho, pval = spearmanr(hamming_vals, weighted_vals)
        print(f"\nSpearman rank correlation (Hamming vs Weighted): rho = {rho:.3f}, p = {pval:.4f}")

if __name__ == '__main__':
    main()