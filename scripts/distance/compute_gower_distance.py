#!/usr/bin/env python3
"""
Compute Gower distance (simple matching coefficient) between English and other languages.
Handles missing values appropriately: only features present in both languages are considered.
Distance = 1 - (matching features / common features).
Also compute weighted distance where each feature's weight is based on its frequency across all languages.
"""

import csv
import numpy as np

def load_feature_matrix(csv_path):
    """Load feature matrix, return dicts: lang->{feature->value}, list of features."""
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        features = header[1:]  # skip 'language' column
        data = {}
        for row in reader:
            lang = row[0]
            values = row[1:]
            # Convert empty strings to None
            feat_dict = {f: (v if v != '' else None) for f, v in zip(features, values)}
            data[lang] = feat_dict
        return data, features

def compute_gower_distance(eng_features, lang_features, features):
    """Compute Gower distance (simple matching coefficient)."""
    matching = 0
    common = 0
    for f in features:
        eng_val = eng_features.get(f)
        lang_val = lang_features.get(f)
        if eng_val is None or lang_val is None:
            continue
        common += 1
        if eng_val == lang_val:
            matching += 1
    if common == 0:
        return None, 0
    similarity = matching / common
    distance = 1 - similarity
    return distance, common

def compute_weighted_gower_distance(eng_features, lang_features, features, weights):
    """Compute weighted Gower distance with feature weights."""
    weighted_matching = 0.0
    total_weight = 0.0
    for f in features:
        eng_val = eng_features.get(f)
        lang_val = lang_features.get(f)
        if eng_val is None or lang_val is None:
            continue
        w = weights.get(f, 1.0)
        total_weight += w
        if eng_val == lang_val:
            weighted_matching += w
    if total_weight == 0:
        return None, 0
    similarity = weighted_matching / total_weight
    distance = 1 - similarity
    return distance, total_weight

def compute_feature_weights(data, features):
    """Compute weight for each feature based on non-missing frequency across all languages."""
    weights = {}
    for f in features:
        present = sum(1 for lang in data.values() if lang.get(f) is not None)
        # Weight = proportion of languages that have this feature
        # Features present in more languages get higher weight (more reliable)
        weight = present / len(data) if len(data) > 0 else 0
        weights[f] = weight
    return weights

def main():
    matrix_path = 'data/processed/wals_feature_matrix.csv'
    print(f"Loading feature matrix from {matrix_path}")
    data, features = load_feature_matrix(matrix_path)

    if 'eng' not in data:
        print("Error: English data not found.")
        return

    eng_features = data['eng']

    # Compute weights based on feature frequency
    weights = compute_feature_weights(data, features)

    results = []
    for lang, lang_features in data.items():
        if lang == 'eng':
            continue
        # Gower distance (unweighted)
        gower_dist, common = compute_gower_distance(eng_features, lang_features, features)
        # Weighted Gower distance
        weighted_dist, total_weight = compute_weighted_gower_distance(
            eng_features, lang_features, features, weights)

        results.append({
            'language': lang,
            'gower_distance': gower_dist,
            'common_features': common,
            'weighted_distance': weighted_dist,
            'total_weight': total_weight
        })

    # Sort by Gower distance (descending)
    results.sort(key=lambda x: x['gower_distance'] if x['gower_distance'] is not None else -1, reverse=True)

    # Output
    output_path = 'data/processed/gower_distance_summary.csv'
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['language', 'gower_distance', 'common_features', 'weighted_distance', 'total_weight']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print("\nGower distances from English (largest to smallest):")
    print("{:<6} {:<15} {:<15} {:<15} {:<15}".format(
        'Lang', 'Gower Dist', 'Common', 'Weighted Dist', 'Total Weight'))
    for r in results:
        gd = r['gower_distance'] if r['gower_distance'] is not None else 'N/A'
        wd = r['weighted_distance'] if r['weighted_distance'] is not None else 'N/A'
        print("{:<6} {:<15.3f} {:<15} {:<15.3f} {:<15.1f}".format(
            r['language'], gd, r['common_features'], wd, r['total_weight']))

    print(f"\nResults written to {output_path}")

    # Also compute and print correlation between Gower and weighted distances
    gower_vals = [r['gower_distance'] for r in results if r['gower_distance'] is not None]
    weighted_vals = [r['weighted_distance'] for r in results if r['weighted_distance'] is not None]
    if gower_vals and weighted_vals:
        corr = np.corrcoef(gower_vals, weighted_vals)[0,1]
        print(f"Correlation between Gower and weighted distances: {corr:.3f}")

if __name__ == '__main__':
    main()