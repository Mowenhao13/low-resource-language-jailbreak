#!/usr/bin/env python3
"""
Build a feature matrix from WALS data.
Rows: languages (including English), columns: WALS parameters, cells: feature values.
Missing values left empty.
"""

import csv
import os
import glob
from collections import defaultdict

def load_english_features(english_csv_path='data/external_wals/values.csv'):
    """Load English features from values.csv, return dict {parameter_id: value}."""
    eng_features = {}
    with open(english_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            if len(row) < 4:
                continue
            lang_id = row[1]
            if lang_id == 'eng':
                param_id = row[2]
                value = row[3]
                eng_features[param_id] = value
    return eng_features

def load_language_features(lang_csv_path):
    """Load features from a language-specific CSV, return dict {parameter_id: value}."""
    features = {}
    with open(lang_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)  # comma-separated
        next(reader)  # skip header
        for row in reader:
            if len(row) < 3:
                continue
            fid = row[0]
            value = row[1]
            features[fid] = value
    return features

def main():
    # Load English features
    print("Loading English features...")
    eng_features = load_english_features()
    print(f"English features: {len(eng_features)}")

    # Get list of language CSV files
    lang_files = glob.glob('data/wals/*.csv')
    print(f"Found {len(lang_files)} language files.")

    # Collect all language codes and their feature dicts
    languages = ['eng']
    feature_dicts = [eng_features]

    for lang_file in lang_files:
        lang_code = os.path.basename(lang_file).replace('_wals.csv', '')
        print(f"Loading {lang_code}...")
        lang_features = load_language_features(lang_file)
        languages.append(lang_code)
        feature_dicts.append(lang_features)

    # Gather all unique parameter IDs across all languages
    all_params = set()
    for fd in feature_dicts:
        all_params.update(fd.keys())
    print(f"Total unique parameters: {len(all_params)}")

    # Sort parameters for consistent ordering (e.g., numeric-alpha)
    # Parameters like '1A', '2A', ... '144K'
    # We'll sort by the numeric part then the suffix
    def param_key(p):
        # split into digits and letters
        import re
        match = re.match(r'(\d+)([A-Za-z]*)', p)
        if match:
            num = int(match.group(1))
            suffix = match.group(2)
            return (num, suffix)
        return (0, p)

    sorted_params = sorted(all_params, key=param_key)

    # Write matrix CSV
    output_path = 'data/processed/wals_feature_matrix.csv'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Header row: language, then param IDs
        header = ['language'] + sorted_params
        writer.writerow(header)
        # Data rows
        for lang, fd in zip(languages, feature_dicts):
            row = [lang]
            for param in sorted_params:
                row.append(fd.get(param, ''))
            writer.writerow(row)

    print(f"Feature matrix written to {output_path}")
    print(f"Shape: {len(languages)} languages x {len(sorted_params)} parameters")

    # Also write a simplified version with Hamming distances
    # Compute Hamming distances between English and each language
    eng_fd = eng_features
    distances = []
    for lang, fd in zip(languages[1:], feature_dicts[1:]):  # skip English
        common = set(eng_fd.keys()) & set(fd.keys())
        dist = sum(1 for p in common if eng_fd.get(p) != fd.get(p))
        distances.append((lang, dist, len(common)))

    dist_path = 'data/processed/hamming_distance_summary.csv'
    with open(dist_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['language', 'hamming_distance', 'common_features'])
        writer.writerows(distances)

    print(f"Hamming distances written to {dist_path}")

if __name__ == '__main__':
    main()