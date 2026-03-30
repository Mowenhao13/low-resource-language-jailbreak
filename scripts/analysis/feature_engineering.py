"""
特征工程模块
编码分类变量、转换数值特征、生成交互特征
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def encode_categorical(df):
    """
    编码分类变量

    Args:
        df: 原始数据框

    Returns:
        tuple: (encoded_df, feature_info)
            encoded_df: 编码后的数据框
            feature_info: 特征信息字典
    """
    df = df.copy()
    feature_info = {
        'numerical': [],
        'categorical_encoded': [],
        'original': []
    }

    # 保留原始分类列用于可视化
    categorical_cols = ['morph_type', 'basic_word_order', 'resource_level', 'family']
    for col in categorical_cols:
        if col in df.columns:
            feature_info['original'].append(col)

    # 1. morph_type - 独热编码 (3类别 -> 2维避免多重共线性)
    if 'morph_type' in df.columns:
        morph_dummies = pd.get_dummies(df['morph_type'], prefix='morph', drop_first=True)
        df = pd.concat([df, morph_dummies], axis=1)
        feature_info['categorical_encoded'].extend(morph_dummies.columns.tolist())

    # 2. basic_word_order - 独热编码
    if 'basic_word_order' in df.columns:
        # 标准化词序（去除空格）
        df['basic_word_order'] = df['basic_word_order'].str.strip()
        word_dummies = pd.get_dummies(df['basic_word_order'], prefix='word_order', drop_first=True)
        df = pd.concat([df, word_dummies], axis=1)
        feature_info['categorical_encoded'].extend(word_dummies.columns.tolist())

    # 3. resource_level - 序数编码 (LRL=0, MRL=1, HRL=2)
    if 'resource_level' in df.columns:
        resource_map = {'LRL': 0, 'MRL': 1, 'HRL': 2}
        df['resource_level_ord'] = df['resource_level'].map(resource_map)
        feature_info['numerical'].append('resource_level_ord')

    # 4. family - 合并为 Indo-European vs Other
    if 'family' in df.columns:
        df['is_indo_european'] = (df['family'] == 'Indo-European').astype(int)
        feature_info['categorical_encoded'].append('is_indo_european')

    return df, feature_info


def transform_numerical(df):
    """
    转换数值特征

    Args:
        df: 数据框

    Returns:
        DataFrame: 转换后的数据框
    """
    df = df.copy()

    # training_data_estimate: 对数转换
    if 'training_data_estimate' in df.columns:
        eps = 1e-6
        df['training_data_log'] = np.log1p(df['training_data_estimate'] / eps)

    return df


def create_interaction_features(df):
    """
    创建交互特征

    Args:
        df: 数据框

    Returns:
        DataFrame: 含交互特征的数据框
    """
    df = df.copy()

    # 交互特征 1: resource_level × syntax_distance
    if 'resource_level_ord' in df.columns and 'syntax_distance_weighted' in df.columns:
        df['resource_x_distance'] = df['resource_level_ord'] * df['syntax_distance_weighted']

    # 交互特征 2: Isolating 形态 × resource_level
    if 'morph_Isolating' in df.columns and 'resource_level_ord' in df.columns:
        df['isolating_x_resource'] = df['morph_Isolating'] * df['resource_level_ord']

    return df


def prepare_features(df, target_col='asr'):
    """
    准备用于建模的特征矩阵

    Args:
        df: 合并后的数据框
        target_col: 目标变量列名

    Returns:
        tuple: (X, y, feature_names, df_full)
            X: 特征矩阵
            y: 目标变量
            feature_names: 特征名称列表
            df_full: 包含所有特征的完整数据框
    """
    df = df.copy()

    # 编码分类变量
    df, feature_info = encode_categorical(df)

    # 转换数值特征
    df = transform_numerical(df)

    # 创建交互特征
    df = create_interaction_features(df)

    # 定义用于建模的特征
    feature_names = []

    # 数值特征
    numerical_candidates = [
        'syntax_distance_weighted',
        'syntax_distance_gower',
        'training_data_log',
        'gower_distance',
        'hamming_distance',
        'resource_level_ord'
    ]
    for feat in numerical_candidates:
        if feat in df.columns:
            feature_names.append(feat)

    # 编码后的分类特征
    for feat in feature_info['categorical_encoded']:
        if feat in df.columns:
            feature_names.append(feat)

    # 交互特征（可选）
    interaction_candidates = ['resource_x_distance', 'isolating_x_resource']
    for feat in interaction_candidates:
        if feat in df.columns:
            feature_names.append(feat)

    # 移除有缺失值的特征
    X_df = df[feature_names].copy()
    X_df = X_df.dropna(axis=1, how='any')
    feature_names = X_df.columns.tolist()

    # 提取 X 和 y
    X = X_df.values.astype(float)

    y = None
    if target_col in df.columns:
        y = df[target_col].values.astype(float)

    return X, y, feature_names, df


def run_feature_engineering(merged_df):
    """
    运行完整的特征工程流程

    Args:
        merged_df: 合并后的原始数据框

    Returns:
        tuple: (X, y, feature_names, df_full)
    """
    print("\n" + "=" * 60)
    print("特征工程阶段")
    print("=" * 60)

    X, y, feature_names, df_full = prepare_features(merged_df)

    print(f"\n特征矩阵形状: {X.shape}")
    print(f"\n特征列表 ({len(feature_names)}):")
    for i, name in enumerate(feature_names, 1):
        print(f"  {i:2d}. {name}")

    if y is not None:
        print(f"\n目标变量统计:")
        print(f"  均值: {np.mean(y):.4f}")
        print(f"  标准差: {np.std(y):.4f}")
        print(f"  最小值: {np.min(y):.4f}")
        print(f"  最大值: {np.max(y):.4f}")

    return X, y, feature_names, df_full


if __name__ == '__main__':
    from data_preparation import prepare_full_dataset
    merged_df = prepare_full_dataset()
    X, y, feature_names, df_full = run_feature_engineering(merged_df)
