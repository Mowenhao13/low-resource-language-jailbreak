"""
特征重要性模块
整合多方法特征重要性评分
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def integrate_feature_importance(model_results, feature_names):
    """
    整合多方法特征重要性

    Args:
        model_results: run_all_models 返回的结果字典
        feature_names: 特征名称列表

    Returns:
        DataFrame: 整合后的特征重要性
    """
    importance_dict = {name: {} for name in feature_names}

    # 1. Spearman 相关
    if 'spearman' in model_results:
        spearman_df = model_results['spearman']
        for _, row in spearman_df.iterrows():
            if row['feature'] in importance_dict:
                importance_dict[row['feature']]['spearman'] = row['abs_corr']

    # 2. LASSO 系数（标准化）
    if 'lasso' in model_results:
        coef_df = model_results['lasso']['coef_df']
        max_abs_coef = coef_df['abs_coef'].max() if coef_df['abs_coef'].max() > 0 else 1
        for _, row in coef_df.iterrows():
            if row['feature'] in importance_dict:
                importance_dict[row['feature']]['lasso'] = row['abs_coef'] / max_abs_coef

    # 3. 随机森林重要性
    if 'random_forest' in model_results:
        imp_df = model_results['random_forest']['importance_df']
        max_imp = imp_df['importance'].max() if imp_df['importance'].max() > 0 else 1
        for _, row in imp_df.iterrows():
            if row['feature'] in importance_dict:
                importance_dict[row['feature']]['random_forest'] = row['importance'] / max_imp

        # 排列重要性
        perm_imp_df = model_results['random_forest']['permutation_importance_df']
        max_perm = perm_imp_df['permutation_importance'].max() if perm_imp_df['permutation_importance'].max() > 0 else 1
        for _, row in perm_imp_df.iterrows():
            if row['feature'] in importance_dict:
                importance_dict[row['feature']]['permutation'] = max(0, row['permutation_importance'] / max_perm)

    # 4. 贝叶斯后验概率
    if 'bayesian' in model_results:
        post_df = model_results['bayesian']['posterior_df']
        for _, row in post_df.iterrows():
            feat = row['feature']
            if feat in importance_dict:
                importance_dict[feat]['bayesian'] = row['prob_non_zero']
            elif feat == 'intercept':
                continue

    # 计算整合评分
    rows = []
    for feat in feature_names:
        scores = importance_dict[feat]

        # 各方法评分（带权重）
        # 0.3 × 贝叶斯 + 0.3 × LASSO + 0.2 × RF + 0.2 × Spearman
        final_score = 0.0

        if 'bayesian' in scores:
            final_score += 0.3 * scores['bayesian']

        if 'lasso' in scores:
            final_score += 0.3 * scores['lasso']

        rf_score = 0.0
        if 'random_forest' in scores:
            rf_score += 0.5 * scores['random_forest']
        if 'permutation' in scores:
            rf_score += 0.5 * scores['permutation']
        final_score += 0.2 * rf_score

        if 'spearman' in scores:
            final_score += 0.2 * scores['spearman']

        rows.append({
            'feature': feat,
            'final_score': final_score,
            'spearman': scores.get('spearman', 0),
            'lasso': scores.get('lasso', 0),
            'random_forest': scores.get('random_forest', 0),
            'permutation': scores.get('permutation', 0),
            'bayesian': scores.get('bayesian', 0)
        })

    df = pd.DataFrame(rows)
    df = df.sort_values('final_score', ascending=False)

    return df


def print_feature_importance(importance_df):
    """
    打印特征重要性

    Args:
        importance_df: 整合后的特征重要性数据框
    """
    print("\n" + "=" * 60)
    print("特征重要性整合结果")
    print("=" * 60)

    print("\n{:<30} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
        "特征", "最终评分", "Spearman", "LASSO", "RF", "贝叶斯"
    ))
    print("-" * 80)

    for _, row in importance_df.iterrows():
        print("{:<30} {:<10.3f} {:<10.3f} {:<10.3f} {:<10.3f} {:<10.3f}".format(
            row['feature'][:28],
            row['final_score'],
            row['spearman'],
            row['lasso'],
            row['random_forest'],
            row['bayesian']
        ))


def run_feature_importance(model_results, feature_names):
    """
    运行特征重要性分析

    Args:
        model_results: 模型结果字典
        feature_names: 特征名称列表

    Returns:
        DataFrame: 整合后的特征重要性
    """
    print("\n" + "=" * 60)
    print("特征重要性分析阶段")
    print("=" * 60)

    importance_df = integrate_feature_importance(model_results, feature_names)
    print_feature_importance(importance_df)

    return importance_df


if __name__ == '__main__':
    print("此模块需要从 main_analysis.py 调用")
