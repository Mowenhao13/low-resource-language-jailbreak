"""
可视化模块
所有分析结果的可视化图表
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sns.set_style("whitegrid")
sns.set_palette("husl")

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def set_chinese_font():
    """设置中文字体支持"""
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False


def plot_correlation_heatmap(df, feature_names, output_dir):
    """
    相关性热图

    Args:
        df: 完整数据框
        feature_names: 数值特征列表
        output_dir: 输出目录
    """
    numerical_cols = []
    for col in ['syntax_distance_weighted', 'training_data_estimate', 'gower_distance', 'asr']:
        if col in df.columns:
            numerical_cols.append(col)

    if len(numerical_cols) < 2:
        return

    corr_mat = df[numerical_cols].corr(method='spearman')

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_mat, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title('Spearman 相关性热图', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_feature_importance(importance_df, output_dir):
    """
    特征重要性对比条形图

    Args:
        importance_df: 特征重要性数据框
        output_dir: 输出目录
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 最终评分
    df_sorted = importance_df.sort_values('final_score', ascending=True)
    colors = plt.cm.viridis(np.linspace(0, 1, len(df_sorted)))
    ax1.barh(df_sorted['feature'], df_sorted['final_score'], color=colors)
    ax1.set_xlabel('最终重要性评分', fontsize=12)
    ax1.set_title('特征重要性整合评分', fontsize=14, fontweight='bold')

    # 多方法对比
    df_melt = importance_df.melt(id_vars=['feature'],
                                  value_vars=['spearman', 'lasso', 'random_forest', 'bayesian'],
                                  var_name='method', value_name='score')
    pivot_df = df_melt.pivot(index='feature', columns='method', values='score')
    pivot_df = pivot_df.loc[importance_df['feature']]
    pivot_df.plot(kind='barh', ax=ax2, width=0.8)
    ax2.set_xlabel('评分', fontsize=12)
    ax2.set_title('多方法特征重要性对比', fontsize=14, fontweight='bold')
    ax2.legend(['Spearman', 'LASSO', 'Random Forest', 'Bayesian'])

    plt.tight_layout()
    plt.savefig(output_dir / 'feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_bayesian_coefficients(bayesian_results, output_dir):
    """
    贝叶斯系数后验分布森林图

    Args:
        bayesian_results: 贝叶斯回归结果
        output_dir: 输出目录
    """
    post_df = bayesian_results['posterior_df']

    # 排除截距
    post_df = post_df[post_df['feature'] != 'intercept'].copy()
    if len(post_df) == 0:
        return

    post_df = post_df.sort_values('posterior_mean', ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(4, len(post_df) * 0.5)))

    y_pos = np.arange(len(post_df))
    ax.errorbar(post_df['posterior_mean'], y_pos,
                xerr=[post_df['posterior_mean'] - post_df['lower_95'],
                      post_df['upper_95'] - post_df['posterior_mean']],
                fmt='o', capsize=5, color='steelblue', linewidth=2, markersize=8)

    ax.axvline(0, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(post_df['feature'], fontsize=11)
    ax.set_xlabel('后验均值 (95% CI)', fontsize=12)
    ax.set_title('贝叶斯回归系数后验分布', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(output_dir / 'bayesian_coefficients.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_asr_by_category(df, output_dir):
    """
    分类特征 vs ASR 箱线图

    Args:
        df: 数据框
        output_dir: 输出目录
    """
    if 'asr' not in df.columns:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    # 1. Resource Level
    if 'resource_level' in df.columns:
        sns.boxplot(x='resource_level', y='asr', data=df, ax=axes[0],
                    order=['LRL', 'MRL', 'HRL'], palette='Set2')
        axes[0].set_title('ASR vs 资源级别', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('资源级别', fontsize=11)

    # 2. Morphology Type
    if 'morph_type' in df.columns:
        sns.boxplot(x='morph_type', y='asr', data=df, ax=axes[1], palette='Set3')
        axes[1].set_title('ASR vs 形态类型', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('形态类型', fontsize=11)

    # 3. Word Order
    if 'basic_word_order' in df.columns:
        df['word_order_clean'] = df['basic_word_order'].str.strip()
        sns.boxplot(x='word_order_clean', y='asr', data=df, ax=axes[2], palette='Set1')
        axes[2].set_title('ASR vs 基本词序', fontsize=12, fontweight='bold')
        axes[2].set_xlabel('基本词序', fontsize=11)

    # 4. Family
    if 'family' in df.columns:
        df['is_indo_european'] = df['family'] == 'Indo-European'
        sns.boxplot(x='is_indo_european', y='asr', data=df, ax=axes[3], palette='Pastel1')
        axes[3].set_title('ASR vs 印欧语系', fontsize=12, fontweight='bold')
        axes[3].set_xlabel('印欧语系', fontsize=11)
        axes[3].set_xticklabels(['否', '是'])

    for i in range(4):
        axes[i].set_ylabel('攻击成功率 (ASR)', fontsize=11)

    plt.tight_layout()
    plt.savefig(output_dir / 'asr_by_category.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_asr_scatters(df, output_dir):
    """
    数值特征 vs ASR 散点图

    Args:
        df: 数据框
        output_dir: 输出目录
    """
    if 'asr' not in df.columns:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Syntax Distance
    if 'syntax_distance_weighted' in df.columns:
        ax = axes[0]
        scatter = ax.scatter(df['syntax_distance_weighted'], df['asr'],
                            s=100, alpha=0.7, c=plt.cm.viridis(df['asr']))
        ax.set_xlabel('语法距离 (加权)', fontsize=12)
        ax.set_ylabel('攻击成功率 (ASR)', fontsize=12)
        ax.set_title('ASR vs 语法距离', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # 添加标签
        for _, row in df.iterrows():
            ax.annotate(row['lang_code'],
                       (row['syntax_distance_weighted'], row['asr']),
                       xytext=(5, 5), textcoords='offset points', fontsize=9)

        # 添加趋势线
        if len(df) > 2:
            z = np.polyfit(df['syntax_distance_weighted'], df['asr'], 1)
            p = np.poly1d(z)
            x_range = np.linspace(df['syntax_distance_weighted'].min(), df['syntax_distance_weighted'].max(), 100)
            ax.plot(x_range, p(x_range), "r--", alpha=0.7)

    # 2. Training Data
    if 'training_data_estimate' in df.columns:
        ax = axes[1]
        scatter = ax.scatter(np.log1p(df['training_data_estimate'] / 1e-6), df['asr'],
                            s=100, alpha=0.7, c=plt.cm.plasma(df['asr']))
        ax.set_xlabel('训练数据占比 (log)', fontsize=12)
        ax.set_ylabel('攻击成功率 (ASR)', fontsize=12)
        ax.set_title('ASR vs 训练数据占比', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)

        for _, row in df.iterrows():
            ax.annotate(row['lang_code'],
                       (np.log1p(row['training_data_estimate'] / 1e-6), row['asr']),
                       xytext=(5, 5), textcoords='offset points', fontsize=9)

        if len(df) > 2:
            z = np.polyfit(np.log1p(df['training_data_estimate'] / 1e-6), df['asr'], 1)
            p = np.poly1d(z)
            x_log = np.log1p(df['training_data_estimate'] / 1e-6)
            x_range = np.linspace(x_log.min(), x_log.max(), 100)
            ax.plot(x_range, p(x_range), "r--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_dir / 'asr_scatters.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_pca_projection(X, y, feature_names, df_full, output_dir):
    """
    PCA 2D 投影

    Args:
        X: 特征矩阵
        y: 目标变量
        feature_names: 特征名称
        df_full: 完整数据框
        output_dir: 输出目录
    """
    if X.shape[0] < 3 or X.shape[1] < 2:
        return

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=min(2, X.shape[1]))
    X_pca = pca.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(10, 8))

    # 颜色映射 ASR
    if y is not None:
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, s=200,
                            cmap='RdYlGn_r', alpha=0.8, edgecolor='black', linewidth=2)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('攻击成功率 (ASR)', fontsize=12)
    else:
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], s=200,
                            alpha=0.8, edgecolor='black', linewidth=2)

    # 添加标签
    for i, (_, row) in enumerate(df_full.iterrows()):
        if i < len(X_pca):
            ax.annotate(row['lang_code'],
                       (X_pca[i, 0], X_pca[i, 1]),
                       xytext=(8, 8), textcoords='offset points',
                       fontsize=11, fontweight='bold')

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} 方差)', fontsize=12)
    if X_pca.shape[1] > 1:
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} 方差)', fontsize=12)
    ax.set_title('语言特征 PCA 投影（颜色=ASR）', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'pca_projection.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_model_predictions(model_results, df_full, output_dir):
    """
    模型预测 vs 实际值

    Args:
        model_results: 模型结果字典
        df_full: 完整数据框
        output_dir: 输出目录
    """
    if 'asr' not in df_full.columns:
        return

    y_true = df_full['asr'].values

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # LASSO
    if 'lasso' in model_results:
        y_pred = model_results['lasso']['y_pred_loo']
        ax = axes[0]
        ax.scatter(y_true, y_pred, s=100, alpha=0.7)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        ax.set_xlabel('实际 ASR', fontsize=11)
        ax.set_ylabel('预测 ASR', fontsize=11)
        ax.set_title(f'LASSO (LOOCV R²={model_results["lasso"]["loo_r2"]:.3f})', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

    # Random Forest
    if 'random_forest' in model_results:
        y_pred = model_results['random_forest']['y_pred_loo']
        ax = axes[1]
        ax.scatter(y_true, y_pred, s=100, alpha=0.7)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        ax.set_xlabel('实际 ASR', fontsize=11)
        ax.set_ylabel('预测 ASR', fontsize=11)
        ax.set_title(f'Random Forest (LOOCV R²={model_results["random_forest"]["loo_r2"]:.3f})', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

    # Bayesian
    if 'bayesian' in model_results:
        y_pred = model_results['bayesian']['y_pred_loo']
        ax = axes[2]
        ax.scatter(y_true, y_pred, s=100, alpha=0.7)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        ax.set_xlabel('实际 ASR', fontsize=11)
        ax.set_ylabel('预测 ASR', fontsize=11)
        ax.set_title(f'Bayesian (LOOCV R²={model_results["bayesian"]["loo_r2"]:.3f})', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # 添加语言标签
        for i, (_, row) in enumerate(df_full.iterrows()):
            if i < len(y_true):
                axes[2].annotate(row['lang_code'], (y_true[i], y_pred[i]),
                                 xytext=(3, 3), textcoords='offset points', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / 'model_predictions.png', dpi=150, bbox_inches='tight')
    plt.close()


def generate_all_visualizations(df_full, X, y, feature_names, model_results, importance_df, output_dir):
    """
    生成所有可视化

    Args:
        df_full: 完整数据框
        X: 特征矩阵
        y: 目标变量
        feature_names: 特征名称列表
        model_results: 模型结果字典
        importance_df: 特征重要性数据框
        output_dir: 输出目录
    """
    print("\n" + "=" * 60)
    print("生成可视化图表")
    print("=" * 60)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_chinese_font()

    print("\n生成图表...")

    # 1. ASR 分类箱线图
    if 'asr' in df_full.columns:
        print("  - ASR by category...")
        plot_asr_by_category(df_full, output_dir)

    # 2. ASR 散点图
    if 'asr' in df_full.columns:
        print("  - ASR scatter plots...")
        plot_asr_scatters(df_full, output_dir)

    # 3. 相关性热图
    print("  - Correlation heatmap...")
    plot_correlation_heatmap(df_full, feature_names, output_dir)

    # 4. 特征重要性
    print("  - Feature importance...")
    plot_feature_importance(importance_df, output_dir)

    # 5. 贝叶斯系数
    if 'bayesian' in model_results:
        print("  - Bayesian coefficients...")
        plot_bayesian_coefficients(model_results['bayesian'], output_dir)

    # 6. PCA 投影
    print("  - PCA projection...")
    plot_pca_projection(X, y, feature_names, df_full, output_dir)

    # 7. 模型预测
    if 'asr' in df_full.columns:
        print("  - Model predictions...")
        plot_model_predictions(model_results, df_full, output_dir)

    print(f"\n所有图表已保存至: {output_dir}")


if __name__ == '__main__':
    print("此模块需要从 main_analysis.py 调用")
