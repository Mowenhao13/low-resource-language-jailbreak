#!/usr/bin/env python3
"""
语言特征标注主协调脚本
基于论文 "Low-Resource Languages Jailbreak GPT-4" 的12种语言
提取6个语言特征：形态复杂度、词序灵活度、语法距离、训练数据占比、音系复杂度、书写系统
"""

import os
import sys
import argparse
import yaml
import pandas as pd
from datetime import datetime

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def load_config():
    """加载配置文件"""
    config_path = os.path.join(project_root, 'config', 'languages.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def check_dependencies():
    """检查必要的依赖和目录"""
    required_dirs = [
        'data/raw',
        'data/processed',
        'config',
        'outputs',
        'scripts'
    ]

    for dir_path in required_dirs:
        full_path = os.path.join(project_root, dir_path)
        if not os.path.exists(full_path):
            os.makedirs(full_path, exist_ok=True)
            print(f"Created directory: {full_path}")

    print("Directory structure checked.")

def run_feature_extraction(features_to_extract=None):
    """
    运行特征提取流程

    Args:
        features_to_extract: 要提取的特征列表，如果为None则提取所有特征
    """
    # 默认提取所有特征
    all_features = [
        'morphology',      # 形态复杂度
        'word_order',      # 词序灵活度
        'grammar_distance', # 语法距离
        'training_proportion', # 训练数据占比
        'phonology',       # 音系复杂度
        'writing_system'   # 书写系统
    ]

    if features_to_extract is None:
        features_to_extract = all_features

    print(f"Starting feature extraction for: {', '.join(features_to_extract)}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 60)

    results = []
    config = load_config()
    languages = config['languages']

    for lang in languages:
        lang_code = lang['code']
        lang_name = lang['name']

        print(f"\nProcessing {lang_name} ({lang_code})...")

        # 这里将调用各个特征提取模块
        # 实际实现时，每个特征有独立的提取函数

        # 示例：收集语言基本信息
        lang_features = {
            'language_code': lang_code,
            'language_name': lang_name,
            'language_family': lang['family'],
            'resource_level': lang['resource_level']
        }

        # 添加各个特征（这里只是占位符）
        for feature in features_to_extract:
            if feature == 'morphology':
                lang_features.update(_extract_morphology(lang_code))
            elif feature == 'word_order':
                lang_features.update(_extract_word_order(lang_code))
            elif feature == 'grammar_distance':
                lang_features.update(_extract_grammar_distance(lang_code))
            elif feature == 'training_proportion':
                lang_features.update(_extract_training_proportion(lang_code))
            elif feature == 'phonology':
                lang_features.update(_extract_phonology(lang_code))
            elif feature == 'writing_system':
                lang_features.update(_extract_writing_system(lang_code))

        results.append(lang_features)
        print(f"  Completed {lang_name}")

    return results

def _extract_morphology(lang_code):
    """提取形态复杂度特征（占位符）"""
    # 实际实现应调用 extract_morphology.py 中的函数
    return {
        'morph_type': 'unknown',
        'morph_confidence': 0.0,
        'morph_source': 'placeholder'
    }

def _extract_word_order(lang_code):
    """提取词序灵活度特征（占位符）"""
    return {
        'basic_word_order': 'unknown',
        'order_flexibility': 0.0,
        'order_source': 'placeholder'
    }

def _extract_grammar_distance(lang_code):
    """提取语法距离特征（占位符）"""
    return {
        'grammar_distance_composite': 0.5,
        'grammar_source': 'placeholder'
    }

def _extract_training_proportion(lang_code):
    """提取训练数据占比特征（占位符）"""
    return {
        'training_data_proportion': 0.0,
        'proportion_source': 'placeholder'
    }

def _extract_phonology(lang_code):
    """提取音系复杂度特征（占位符）"""
    return {
        'phonological_complexity': 0.0,
        'consonant_count': 0,
        'vowel_count': 0,
        'phonology_source': 'placeholder'
    }

def _extract_writing_system(lang_code):
    """提取书写系统特征（占位符）"""
    return {
        'script_type': 'unknown',
        'writing_direction': 'unknown',
        'script_source': 'placeholder'
    }

def save_results(results, output_file=None):
    """保存特征提取结果"""
    if output_file is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f'data/processed/language_features_{timestamp}.csv'

    output_path = os.path.join(project_root, output_file)

    # 确保目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 转换为DataFrame并保存
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False, encoding='utf-8')

    print(f"\nResults saved to: {output_path}")
    print(f"Total languages processed: {len(df)}")
    print(f"Total features extracted: {len(df.columns) - 4}")  # 减去语言基本信息列

    # 显示简要统计
    print("\nSummary statistics:")
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    if len(numeric_cols) > 0:
        print(df[numeric_cols].describe().round(3))

    return output_path

def generate_report(results_file):
    """生成分析报告（占位符）"""
    report_template = f"""
# 语言特征标注分析报告

**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**数据文件**: {results_file}
**语言数量**: 12
**特征数量**: 6

## 特征概述
1. 形态复杂度 (morphology)
2. 词序灵活度 (word_order)
3. 语法距离 (grammar_distance)
4. 训练数据占比 (training_proportion)
5. 音系复杂度 (phonology)
6. 书写系统 (writing_system)

## 后续步骤
1. 实现各个特征提取模块的具体逻辑
2. 添加数据验证和质量控制
3. 进行特征相关性分析
4. 与越狱成功率数据进行关联分析

## 备注
当前为框架版本，各特征提取函数需要具体实现。
参考实现位于 scripts/ 目录下的各个独立脚本。
"""

    report_path = os.path.join(project_root, 'outputs', 'feature_extraction_report.md')
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_template)

    print(f"Report generated: {report_path}")
    return report_path

def main():
    parser = argparse.ArgumentParser(description='语言特征标注主协调脚本')
    parser.add_argument('--features', nargs='+',
                       choices=['morphology', 'word_order', 'grammar_distance',
                                'training_proportion', 'phonology', 'writing_system'],
                       help='指定要提取的特征（默认提取所有）')
    parser.add_argument('--output', type=str,
                       help='输出文件路径（默认: data/processed/language_features_YYYYMMDD_HHMMSS.csv）')
    parser.add_argument('--check-only', action='store_true',
                       help='仅检查环境和配置，不执行特征提取')

    args = parser.parse_args()

    print("=" * 60)
    print("Language Feature Extraction System")
    print("Based on: Low-Resource Languages Jailbreak GPT-4")
    print("=" * 60)

    # 检查依赖和目录
    check_dependencies()

    if args.check_only:
        print("\nEnvironment check completed. Exiting.")
        return

    # 运行特征提取
    results = run_feature_extraction(args.features)

    # 保存结果
    output_file = save_results(results, args.output)

    # 生成报告
    report_file = generate_report(output_file)

    print("\n" + "=" * 60)
    print("Feature extraction completed successfully!")
    print(f"Results: {output_file}")
    print(f"Report: {report_file}")
    print("=" * 60)

if __name__ == "__main__":
    main()