"""
数据准备模块
加载并合并语言特征、ASR数据和距离度量
"""

import pandas as pd
import yaml
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_language_features(yaml_path=None):
    """
    从 YAML 文件加载语言特征

    Args:
        yaml_path: config/languages.yaml 路径

    Returns:
        DataFrame: 语言特征数据框
    """
    if yaml_path is None:
        yaml_path = PROJECT_ROOT / 'config' / 'languages.yaml'

    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    languages = config.get('languages', [])

    # 提取字段
    data = []
    for lang in languages:
        data.append({
            'lang_code': lang['code'],
            'lang_name': lang['name'],
            'lang_name_chinese': lang.get('name_chinese', ''),
            'syntax_distance_weighted': lang.get('syntax_distance_weighted'),
            'syntax_distance_gower': lang.get('syntax_distance_gower'),
            'syntax_common_features': lang.get('syntax_common_features'),
            'training_data_estimate': lang.get('training_data_estimate'),
            'morph_type': lang.get('morph_type'),
            'basic_word_order': lang.get('basic_word_order', '').strip(),
            'resource_level': lang.get('resource_level'),
            'family': lang.get('family')
        })

    return pd.DataFrame(data)


def calculate_asr(csv_path=None):
    """
    从 LLM 输出 CSV 计算每种语言的 ASR

    Args:
        csv_path: advbench_output CSV 文件路径

    Returns:
        DataFrame: 包含 lang_code, asr, success, total 的数据框
    """
    if csv_path is None:
        csv_path = PROJECT_ROOT / 'outputs' / 'llm' / 'deepseek-chat' / 'advbench_output_deepseek-chat.csv'

    if not csv_path.exists():
        print(f"Warning: {csv_path} not found, returning empty ASR data")
        return pd.DataFrame(columns=['lang_code', 'asr', 'success', 'total'])

    df = pd.read_csv(csv_path)

    # 过滤掉错误响应
    df = df[~df['model_resp'].astype(str).str.startswith('[Error', na=False)]

    lang_results = []

    for lang in df['source_lang_code'].unique():
        lang_df = df[df['source_lang_code'] == lang]
        success_count = lang_df['evaluation'].sum()
        total_count = len(lang_df)
        asr = success_count / total_count if total_count > 0 else 0.0

        lang_results.append({
            'lang_code': lang,
            'asr': float(asr),
            'success': int(success_count),
            'total': int(total_count)
        })

    return pd.DataFrame(lang_results)


def load_distance_metrics(csv_path=None):
    """
    加载额外的距离度量

    Args:
        csv_path: final_distance_metrics.csv 路径

    Returns:
        DataFrame: 距离度量数据框
    """
    if csv_path is None:
        csv_path = PROJECT_ROOT / 'data' / 'processed' / 'final_distance_metrics.csv'

    if not csv_path.exists():
        print(f"Warning: {csv_path} not found, returning empty distance data")
        return pd.DataFrame(columns=['lang_code'])

    df = pd.read_csv(csv_path)

    # 标准化列名以匹配
    df = df.rename(columns={'language': 'lang_code', 'language_name': 'lang_name'})

    return df


def merge_all(lang_feat, asr_df, dist_df=None):
    """
    合并所有数据源

    Args:
        lang_feat: 语言特征 DataFrame
        asr_df: ASR 数据 DataFrame
        dist_df: 距离度量 DataFrame (可选)

    Returns:
        DataFrame: 合并后的数据
    """
    # 首先合并语言特征和 ASR
    merged = lang_feat.copy()

    if asr_df is not None and len(asr_df) > 0:
        merged = merged.merge(asr_df, on='lang_code', how='left')
        # 如果没有 ASR 数据，填充为 NaN
        merged['asr'] = merged['asr'].fillna(0.0)
        merged['success'] = merged['success'].fillna(0).astype(int)
        merged['total'] = merged['total'].fillna(0).astype(int)

    # 补充距离度量
    if dist_df is not None and len(dist_df) > 0:
        # 选择需要的列
        dist_cols = ['lang_code', 'hamming_distance', 'gower_distance']
        available_cols = [c for c in dist_cols if c in dist_df.columns]
        if len(available_cols) > 1:
            merged = merged.merge(dist_df[available_cols], on='lang_code', how='left')

    return merged


def prepare_full_dataset(save_path=None):
    """
    准备完整数据集的主函数

    Args:
        save_path: 保存合并数据的路径

    Returns:
        DataFrame: 合并后的完整数据集
    """
    print("=" * 60)
    print("数据准备阶段")
    print("=" * 60)

    # 加载各数据源
    print("\n[1/4] 加载语言特征...")
    lang_feat = load_language_features()
    print(f"  加载了 {len(lang_feat)} 种语言的特征")

    print("\n[2/4] 计算 ASR...")
    asr_df = calculate_asr()
    print(f"  有 ASR 数据的语言: {len(asr_df)} 种")

    print("\n[3/4] 加载距离度量...")
    dist_df = load_distance_metrics()
    print(f"  距离度量数据: {len(dist_df)} 种语言")

    print("\n[4/4] 合并数据...")
    merged = merge_all(lang_feat, asr_df, dist_df)
    print(f"  合并后数据形状: {merged.shape}")

    # 保存
    if save_path is None:
        save_path = PROJECT_ROOT / 'data' / 'analysis' / 'merged_data.csv'

    save_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(save_path, index=False, encoding='utf-8')
    print(f"\n数据已保存至: {save_path}")

    # 打印预览
    print("\n数据预览:")
    print("-" * 60)
    cols_to_show = ['lang_code', 'lang_name', 'asr', 'resource_level', 'morph_type',
                    'syntax_distance_weighted', 'training_data_estimate']
    available_cols = [c for c in cols_to_show if c in merged.columns]
    print(merged[available_cols].to_string(index=False))

    return merged


if __name__ == '__main__':
    prepare_full_dataset()
