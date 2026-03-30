"""
模式挖掘模块
识别高漏洞语言特征组合
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import KBinsDiscretizer

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def discretize_features(df, feature_names):
    """
    离散化数值特征用于模式挖掘

    Args:
        df: 完整数据框
        feature_names: 特征名称列表

    Returns:
        DataFrame: 离散化后的特征
    """
    df_disc = df.copy()

    for feat in feature_names:
        if feat not in df.columns:
            continue

        # 根据特征类型选择离散化方式
        if 'resource_level_ord' in feat:
            # 已经是序数
            continue
        elif 'distance' in feat or 'syntax' in feat:
            # 距离特征：二分为高/低
            if df[feat].nunique() > 2:
                median = df[feat].median()
                df_disc[f'{feat}_bin'] = (df[feat] > median).astype(int)
        elif 'training' in feat:
            # 训练数据：二分为高/低
            if df[feat].nunique() > 2:
                median = df[feat].median()
                df_disc[f'{feat}_bin'] = (df[feat] > median).astype(int)

    return df_disc


def find_high_risk_combinations_tree(df, feature_names, target_col='asr'):
    """
    使用决策树寻找高风险特征组合

    Args:
        df: 数据框
        feature_names: 特征名称列表
        target_col: 目标变量列

    Returns:
        dict: 决策树规则
    """
    if target_col not in df.columns:
        return None

    # 定义高漏洞：ASR > 中位数
    median_asr = df[target_col].median()
    y = (df[target_col] > median_asr).astype(int)

    # 准备特征（使用原始分类特征）
    X_dict = {}
    features_used = []

    # 形态类型
    if 'morph_type' in df.columns:
        morph_map = {'Agglutinative': 0, 'Fusional': 1, 'Isolating': 2}
        X_dict['morph_type'] = df['morph_type'].map(morph_map).fillna(0)
        features_used.append('morph_type')

    # 词序
    if 'basic_word_order' in df.columns:
        word_map = {'SVO': 0, 'VSO': 1, 'SOV': 2}
        X_dict['word_order'] = df['basic_word_order'].str.strip().map(word_map).fillna(0)
        features_used.append('word_order')

    # 资源级别
    if 'resource_level' in df.columns:
        resource_map = {'LRL': 0, 'MRL': 1, 'HRL': 2}
        X_dict['resource_level'] = df['resource_level'].map(resource_map)
        features_used.append('resource_level')

    # 语系
    if 'family' in df.columns:
        X_dict['is_indo_european'] = (df['family'] == 'Indo-European').astype(int)
        features_used.append('is_indo_european')

    # 数值特征二值化
    if 'syntax_distance_weighted' in df.columns:
        median_dist = df['syntax_distance_weighted'].median()
        X_dict['high_distance'] = (df['syntax_distance_weighted'] > median_dist).astype(int)
        features_used.append('high_distance')

    if 'training_data_estimate' in df.columns:
        median_train = df['training_data_estimate'].median()
        X_dict['low_training'] = (df['training_data_estimate'] < median_train).astype(int)
        features_used.append('low_training')

    if not X_dict:
        return None

    X = pd.DataFrame(X_dict).values

    # 训练浅层决策树
    tree = DecisionTreeClassifier(
        max_depth=3,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42
    )
    tree.fit(X, y)

    # 提取规则
    rules = export_text(tree, feature_names=features_used)

    # 找出高风险叶子节点
    leaf_nodes = []
    for i, (value, impurity, n_node) in enumerate(zip(tree.tree_.value, tree.tree_.impurity, tree.tree_.n_node_samples)):
        if tree.tree_.children_left[i] == -1:  # 叶子节点
            class_prob = value[0] / value[0].sum() if value[0].sum() > 0 else 0
            leaf_nodes.append({
                'node_id': i,
                'n_samples': n_node,
                'class_0_prob': class_prob[0],
                'class_1_prob': class_prob[1] if len(class_prob) > 1 else 0
            })

    high_risk_leaves = [l for l in leaf_nodes if l['class_1_prob'] > 0.5]

    return {
        'tree': tree,
        'rules': rules,
        'feature_names': features_used,
        'leaf_nodes': leaf_nodes,
        'high_risk_leaves': high_risk_leaves,
        'median_asr': median_asr
    }


def mine_association_rules(df, target_col='asr', min_support=0.2, min_confidence=0.6):
    """
    简单的关联规则挖掘（基于频率统计）

    Args:
        df: 数据框
        target_col: 目标变量列
        min_support: 最小支持度
        min_confidence: 最小置信度

    Returns:
        list: 关联规则列表
    """
    if target_col not in df.columns:
        return []

    median_asr = df[target_col].median()
    df = df.copy()
    df['high_risk'] = (df[target_col] > median_asr).astype(int)

    n_total = len(df)

    # 创建特征项
    itemsets = []
    for idx, row in df.iterrows():
        items = set()
        items.add(f'high_risk={row["high_risk"]}')

        if 'resource_level' in row:
            items.add(f'resource={row["resource_level"]}')

        if 'morph_type' in row:
            items.add(f'morph={row["morph_type"]}')

        if 'basic_word_order' in row:
            items.add(f'word_order={row["basic_word_order"].strip()}')

        if 'family' in row:
            items.add(f'family={row["family"]}')

        if 'syntax_distance_weighted' in row:
            dist = row['syntax_distance_weighted']
            if dist > df['syntax_distance_weighted'].median():
                items.add('distance=high')
            else:
                items.add('distance=low')

        if 'training_data_estimate' in row:
            train = row['training_data_estimate']
            if train < df['training_data_estimate'].median():
                items.add('training=low')
            else:
                items.add('training=high')

        itemsets.append(items)

    # 计算简单规则
    rules = []

    # 单特征规则
    feature_candidates = [
        'resource=LRL', 'resource=MRL', 'resource=HRL',
        'morph=Isolating', 'morph=Agglutinative', 'morph=Fusional',
        'word_order=SVO', 'word_order=VSO', 'word_order=SOV',
        'distance=high', 'distance=low',
        'training=low', 'training=high'
    ]

    for feat in feature_candidates:
        # 计算支持度和置信度
        n_antecedent = sum(1 for items in itemsets if feat in items)
        n_both = sum(1 for items in itemsets if feat in items and 'high_risk=1' in items)

        support = n_antecedent / n_total if n_total > 0 else 0
        confidence = n_both / n_antecedent if n_antecedent > 0 else 0

        if support >= min_support and confidence > 0:
            # 计算提升度
            n_high_risk = sum(1 for items in itemsets if 'high_risk=1' in items)
            expected_conf = n_high_risk / n_total if n_total > 0 else 0
            lift = confidence / expected_conf if expected_conf > 0 else 1

            rules.append({
                'antecedent': feat,
                'consequent': 'high_risk=1',
                'support': support,
                'confidence': confidence,
                'lift': lift,
                'coverage': n_antecedent,
                'hit_count': n_both
            })

    # 双特征组合
    for i, feat1 in enumerate(feature_candidates):
        for feat2 in feature_candidates[i+1:]:
            n_antecedent = sum(1 for items in itemsets if feat1 in items and feat2 in items)
            n_both = sum(1 for items in itemsets if feat1 in items and feat2 in items and 'high_risk=1' in items)

            support = n_antecedent / n_total if n_total > 0 else 0
            confidence = n_both / n_antecedent if n_antecedent > 0 else 0

            if support >= min_support * 0.5 and confidence > 0:
                n_high_risk = sum(1 for items in itemsets if 'high_risk=1' in items)
                expected_conf = n_high_risk / n_total if n_total > 0 else 0
                lift = confidence / expected_conf if expected_conf > 0 else 1

                rules.append({
                    'antecedent': f'{feat1} & {feat2}',
                    'consequent': 'high_risk=1',
                    'support': support,
                    'confidence': confidence,
                    'lift': lift,
                    'coverage': n_antecedent,
                    'hit_count': n_both
                })

    rules_df = pd.DataFrame(rules)
    if len(rules_df) > 0:
        rules_df = rules_df.sort_values(['lift', 'confidence'], ascending=[False, False])

    return rules_df


def score_combination(row, combination):
    """
    为单个语言评分是否符合特征组合

    Args:
        row: 语言数据行
        combination: 特征组合定义

    Returns:
        bool: 是否符合
    """
    for key, value in combination.items():
        if key == 'name':
            continue
        if key not in row:
            return False

        if isinstance(value, tuple):
            # 范围条件
            if value[0] == '>':
                if not (row[key] > value[1]):
                    return False
            elif value[0] == '<':
                if not (row[key] < value[1]):
                    return False
            elif value[0] == '>=':
                if not (row[key] >= value[1]):
                    return False
            elif value[0] == '<=':
                if not (row[key] <= value[1]):
                    return False
        else:
            # 相等条件
            if str(row[key]).strip() != str(value).strip():
                return False

    return True


def identify_high_risk_combinations(df, target_col='asr'):
    """
    主函数：识别高风险特征组合

    Args:
        df: 完整数据框
        target_col: 目标变量列

    Returns:
        dict: 高风险组合分析结果
    """
    print("\n" + "=" * 60)
    print("高风险语言特征组合识别")
    print("=" * 60)

    results = {}

    # 1. 决策树分析
    print("\n[1/2] 决策树分析...")
    tree_results = find_high_risk_combinations_tree(df, [], target_col)
    if tree_results:
        results['decision_tree'] = tree_results
        print("  完成")
        print(f"\n决策树规则:\n{tree_results['rules']}")

    # 2. 关联规则挖掘
    print("\n[2/2] 关联规则挖掘...")
    rules_df = mine_association_rules(df, target_col)
    if len(rules_df) > 0:
        results['association_rules'] = rules_df
        print("  完成")
        print(f"\n发现 {len(rules_df)} 条规则，Top 5:")
        print(rules_df.head()[['antecedent', 'confidence', 'lift', 'coverage']].to_string(index=False))

    # 3. 基于领域知识定义候选组合
    print("\n领域知识候选组合:")
    candidate_combinations = [
        {
            'name': '孤立语 + 低资源',
            'morph_type': 'Isolating',
            'resource_level': 'LRL'
        },
        {
            'name': '低资源 + 高语法距离',
            'resource_level': 'LRL',
            'syntax_distance_weighted': ('>', 0.4)
        },
        {
            'name': '非印欧语系 + 低训练数据',
            'family': ('!=', 'Indo-European'),
            'training_data_estimate': ('<', 0.001)
        },
        {
            'name': '黏着语 + VSO词序',
            'morph_type': 'Agglutinative',
            'basic_word_order': 'VSO'
        }
    ]

    if target_col in df.columns:
        median_asr = df[target_col].median()
        domain_results = []

        for combo in candidate_combinations:
            # 简化版评分
            matches = 0
            high_risk_in_match = 0
            for _, row in df.iterrows():
                match = True
                for key, value in combo.items():
                    if key == 'name':
                        continue
                    if key not in row:
                        match = False
                        break

                    if isinstance(value, tuple):
                        op, val = value
                        if op == '>':
                            if not (row[key] > val):
                                match = False
                        elif op == '<':
                            if not (row[key] < val):
                                match = False
                        elif op == '!=':
                            if not (row[key] != val):
                                match = False
                    else:
                        if str(row[key]).strip() != str(value).strip():
                            match = False

                if match:
                    matches += 1
                    if row[target_col] > median_asr:
                        high_risk_in_match += 1

            if matches > 0:
                purity = high_risk_in_match / matches
                domain_results.append({
                    'combination': combo['name'],
                    'coverage': matches,
                    'purity': purity,
                    'high_risk_count': high_risk_in_match
                })

        if domain_results:
            results['domain_combinations'] = pd.DataFrame(domain_results)
            print("\n领域知识组合评估:")
            print(results['domain_combinations'].to_string(index=False))

    return results


if __name__ == '__main__':
    print("此模块需要从 main_analysis.py 调用")
