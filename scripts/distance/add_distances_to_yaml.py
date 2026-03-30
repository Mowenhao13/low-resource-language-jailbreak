#!/usr/bin/env python3
"""
Add syntax distance metrics from WALS to languages.yaml.
"""

import csv
import re

def load_distance_data(csv_path='data/processed/final_distance_metrics.csv'):
    """Load distance data into dict keyed by language code."""
    distances = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            code = row['language']
            distances[code] = {
                'weighted_distance': float(row['weighted_distance']),
                'gower_distance': float(row['gower_distance']),
                'common_features': int(row['common_features_g']),
                'total_weight': float(row['total_weight'])
            }
    return distances

def update_yaml_file(yaml_path, distances):
    """Update languages.yaml with distance metrics."""
    with open(yaml_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    new_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        new_lines.append(line)

        # Check if this line starts a language entry
        if line.strip().startswith('- code:'):
            # Extract language code
            match = re.search(r'code:\s*(\S+)', line)
            if match:
                code = match.group(1).strip()
                if code in distances:
                    dist = distances[code]
                    # Find the indentation level (2 spaces before '-')
                    indent = ''
                    for j in range(len(line)):
                        if line[j] == '-':
                            indent = line[:j]
                            break

                    # Add distance fields after the code line
                    # We'll insert them before the next language entry or end of languages list
                    # Actually, we need to insert them at the end of this language block
                    # We'll look ahead to find where this block ends
                    j = i + 1
                    while j < len(lines) and not lines[j].strip().startswith('- code:'):
                        j += 1
                    # j is now at next language entry or end of file
                    # Insert distance lines at position j
                    # But we need to insert them before we process the next language
                    # Instead, let's insert right after the code line, but we need to maintain order
                    # We'll insert after the current line, but before the next line
                    # Actually, easier: we'll add the fields immediately
                    new_lines.append(f'{indent}  syntax_distance_weighted: {dist["weighted_distance"]:.4f}  # 加权Gower距离 (0-1)\n')
                    new_lines.append(f'{indent}  syntax_distance_gower: {dist["gower_distance"]:.4f}      # 标准Gower距离\n')
                    new_lines.append(f'{indent}  syntax_common_features: {dist["common_features"]}        # 与英语共同的特征数\n')
                    new_lines.append(f'{indent}  syntax_distance_source: "WALS"          # 特征来源\n')

        i += 1

    # Write back
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)

def main():
    print("Loading distance data...")
    distances = load_distance_data()
    print(f"Loaded distances for {len(distances)} languages")

    yaml_path = 'config/languages.yaml'
    print(f"Updating {yaml_path}...")
    update_yaml_file(yaml_path, distances)
    print("Done!")

if __name__ == '__main__':
    main()