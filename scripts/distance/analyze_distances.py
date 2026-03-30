#!/usr/bin/env python3
import csv

# Load distances
with open('data/processed/hamming_distance_summary.csv', 'r') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

# Convert to list of dicts
for r in rows:
    r['hamming_distance'] = int(r['hamming_distance'])
    r['common_features'] = int(r['common_features'])
    # Calculate normalized distance (distance / common_features)
    if r['common_features'] > 0:
        r['normalized'] = r['hamming_distance'] / r['common_features']
    else:
        r['normalized'] = 0.0

# Sort by distance (descending)
sorted_by_dist = sorted(rows, key=lambda x: x['hamming_distance'], reverse=True)
sorted_by_norm = sorted(rows, key=lambda x: x['normalized'], reverse=True)

print("Hamming distances from English (largest to smallest):")
print("{:<6} {:<15} {:<15} {:<15}".format('Lang', 'Distance', 'Common', 'Normalized'))
for r in sorted_by_dist:
    print("{:<6} {:<15} {:<15} {:<15.3f}".format(
        r['language'], r['hamming_distance'], r['common_features'], r['normalized']))

print("\n\nLanguage codes mapping:")
lang_names = {
    'th': 'Thai',
    'zu': 'Zulu',
    'gd': 'Scottish Gaelic',
    'gn': 'Guarani',
    'zh-CN': 'Chinese (Simplified)',
    'it': 'Italian',
    'hi': 'Hindi',
    'bn': 'Bengali',
    'hmn': 'Hmong',
    'he': 'Hebrew',
    'uk': 'Ukrainian',
    'ar': 'Arabic'
}
for code, name in lang_names.items():
    print(f"{code}: {name}")