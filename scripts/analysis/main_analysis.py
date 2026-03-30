"""
主流程脚本
整合所有模块，运行完整分析流程
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analysis.data_preparation import prepare_full_dataset
from scripts.analysis.feature_engineering import run_feature_engineering
from scripts.analysis.regression_models import run_all_models
from scripts.analysis.feature_importance import run_feature_importance
from scripts.analysis.pattern_mining import identify_high_risk_combinations
from scripts.analysis.visualization import generate_all_visualizations


def generate_report(df_full, feature_names, model_results, importance_df, pattern_results, output_dir):
    """
    生成分析报告

    Args:
        df_full: 完整数据框
        feature_names: 特征名称列表
        model_results: 模型结果字典
        importance_df: 特征重要性数据框
        pattern_results: 模式挖掘结果
        output_dir: 输出目录
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / 'report.md'

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 语言特征与攻击成功率 (ASR) 回归分析报告\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # 1. 数据概览
        f.write("## 1. 数据概览\n\n")
        f.write(f"分析语言数量: {len(df_full)}\n\n")

        if 'asr' in df_full.columns:
            f.write("### 攻击成功率 (ASR) 统计\n\n")
            f.write("| 语言 | 中文名称 | ASR | 成功/总数 | 资源级别 | 形态类型 |\n")
            f.write("|------|---------|-----|----------|---------|---------|\n")
            for _, row in df_full.iterrows():
                success = row.get('success', 0)
                total = row.get('total', 0)
                asr_val = row.get('asr', 0)
                f.write(f"| {row['lang_code']} | {row.get('lang_name_chinese', row['lang_name'])} | {asr_val:.2%} | {success}/{total} | {row['resource_level']} | {row['morph_type']} |\n")
            f.write("\n")

            f.write(f"ASR 均值: {df_full['asr'].mean():.2%}\n\n")
            f.write(f"ASR 中位数: {df_full['asr'].median():.2%}\n\n")

        # 2. 特征重要性
        f.write("## 2. 特征重要性\n\n")
        f.write("### 整合评分 (Top 特征)\n\n")
        f.write("| 排名 | 特征 | 最终评分 | Spearman | LASSO | 随机森林 | 贝叶斯 |\n")
        f.write("|------|------|---------|----------|-------|---------|--------|\n")
        for i, (_, row) in enumerate(importance_df.head(5).iterrows(), 1):
            f.write(f"| {i} | {row['feature']} | {row['final_score']:.3f} | {row['spearman']:.3f} | {row['lasso']:.3f} | {row['random_forest']:.3f} | {row['bayesian']:.3f} |\n")
        f.write("\n")

        # 3. 模型结果
        f.write("## 3. 回归模型结果\n\n")

        if 'lasso' in model_results:
            f.write("### LASSO 回归\n\n")
            f.write(f"- LOOCV R²: {model_results['lasso']['loo_r2']:.4f}\n\n")
            f.write("特征系数:\n\n")
            for _, row in model_results['lasso']['coef_df'].iterrows():
                if abs(row['coef']) > 1e-6:
                    f.write(f"- {row['feature']}: {row['coef']:.4f}\n")
            f.write("\n")

        if 'bayesian' in model_results:
            f.write("### 贝叶斯线性回归\n\n")
            f.write(f"- LOOCV R²: {model_results['bayesian']['loo_r2']:.4f}\n\n")
            post_df = model_results['bayesian']['posterior_df']
            post_df = post_df[post_df['feature'] != 'intercept']
            if len(post_df) > 0:
                f.write("后验概率 > 50% 的特征:\n\n")
                for _, row in post_df.iterrows():
                    if row['prob_non_zero'] > 0.5:
                        f.write(f"- {row['feature']}: 后验均值={row['posterior_mean']:.4f} (95% CI: [{row['lower_95']:.4f}, {row['upper_95']:.4f}]), P(≠0)={row['prob_non_zero']:.1%}\n")
                f.write("\n")

        if 'spearman' in model_results:
            f.write("### Spearman 秩相关 (Top 5)\n\n")
            for _, row in model_results['spearman'].head(5).iterrows():
                f.write(f"- {row['feature']}: ρ={row['spearman_corr']:.4f}, p={row['p_value']:.4f}\n")
            f.write("\n")

        # 4. 高风险特征组合
        f.write("## 4. 高风险语言特征组合\n\n")

        if 'association_rules' in pattern_results:
            rules_df = pattern_results['association_rules']
            if len(rules_df) > 0:
                f.write("### 关联规则 (Top 5)\n\n")
                f.write("| 规则 | 置信度 | 提升度 | 覆盖度 |\n")
                f.write("|------|--------|--------|--------|\n")
                for _, row in rules_df.head(5).iterrows():
                    f.write(f"| {row['antecedent']} → high_risk | {row['confidence']:.1%} | {row['lift']:.2f}x | {row['coverage']} |\n")
                f.write("\n")

        if 'decision_tree' in pattern_results:
            f.write("### 决策树规则\n\n")
            f.write("```\n")
            f.write(pattern_results['decision_tree']['rules'])
            f.write("```\n\n")
            f.write(f"高风险定义: ASR > {pattern_results['decision_tree']['median_asr']:.2%}\n\n")

        if 'domain_combinations' in pattern_results:
            f.write("### 领域知识候选组合\n\n")
            dom_df = pattern_results['domain_combinations']
            if len(dom_df) > 0:
                f.write("| 组合 | 覆盖语言数 | 纯度 |\n")
                f.write("|------|-----------|------|\n")
                for _, row in dom_df.iterrows():
                    f.write(f"| {row['combination']} | {row['coverage']} | {row['purity']:.1%} |\n")
                f.write("\n")

        # 5. 结论
        f.write("## 5. 结论与建议\n\n")
        f.write("### 主要发现\n\n")

        top_features = importance_df.head(3)['feature'].tolist()
        f.write(f"1. 最重要的特征: {', '.join(top_features)}\n\n")

        if 'asr' in df_full.columns:
            lrl_asr = df_full[df_full['resource_level'] == 'LRL']['asr'].mean()
            hrl_asr = df_full[df_full['resource_level'] == 'HRL']['asr'].mean()
            f.write(f"2. 低资源语言平均 ASR: {lrl_asr:.2%}, 高资源语言平均 ASR: {hrl_asr:.2%}\n\n")

        f.write("### 建议\n\n")
        f.write("1. **注意小样本局限性**: 本分析仅基于 12 种语言，结果为探索性，需更多语言验证\n\n")
        f.write("2. **多模型验证**: 建议在更多 LLM 上重复测试\n\n")
        f.write("3. **高风险特征组合**: 建议重点关注识别出的高风险特征组合\n\n")

        # 6. 可视化
        f.write("## 6. 可视化图表\n\n")
        f.write("所有可视化图表保存在 `figures/` 目录中:\n\n")
        f.write("- `correlation_heatmap.png` - 相关性热图\n")
        f.write("- `feature_importance.png` - 特征重要性对比\n")
        f.write("- `bayesian_coefficients.png` - 贝叶斯系数后验分布\n")
        f.write("- `asr_by_category.png` - 分类特征 vs ASR\n")
        f.write("- `asr_scatters.png` - 数值特征 vs ASR\n")
        f.write("- `pca_projection.png` - PCA 2D 投影\n")
        f.write("- `model_predictions.png` - 模型预测 vs 实际值\n")

    print(f"\n报告已保存至: {report_path}")
    return report_path


def main():
    """主分析流程"""
    print("\n" + "=" * 60)
    print("语言特征与 ASR 回归分析")
    print("=" * 60)

    # 输出目录
    output_dir = PROJECT_ROOT / 'outputs' / 'analysis'
    figures_dir = output_dir / 'figures'

    # 1. 数据准备
    merged_df = prepare_full_dataset()

    # 2. 特征工程
    X, y, feature_names, df_full = run_feature_engineering(merged_df)

    # 3. 回归建模
    model_results = None
    if y is not None and len(y) >= 3 and np.var(y) > 0:
        model_results = run_all_models(X, y, feature_names)
    else:
        print("\n警告: ASR 数据不足或无方差，跳过建模")
        # 创建模拟 ASR 用于演示
        print("\n提示: 请先运行完整的 AdvBench 测试:")
        print("  python scripts/extract_advbench_output.py --limit 100")
        return

    # 4. 特征重要性
    importance_df = run_feature_importance(model_results, feature_names)

    # 5. 模式挖掘
    pattern_results = identify_high_risk_combinations(df_full)

    # 6. 可视化
    generate_all_visualizations(df_full, X, y, feature_names, model_results, importance_df, figures_dir)

    # 7. 生成报告
    generate_report(df_full, feature_names, model_results, importance_df, pattern_results, output_dir)

    print("\n" + "=" * 60)
    print("分析完成!")
    print("=" * 60)


if __name__ == '__main__':
    main()
