"""
回归模型模块
贝叶斯线性回归、LASSO、随机森林等模型
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
from scipy import stats
from sklearn.linear_model import Lasso, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def spearman_correlation(X, y, feature_names):
    """
    Spearman 秩相关分析（单变量）

    Args:
        X: 特征矩阵
        y: 目标变量
        feature_names: 特征名称列表

    Returns:
        DataFrame: 相关系数和 p 值
    """
    results = []

    for i, name in enumerate(feature_names):
        corr, p_val = stats.spearmanr(X[:, i], y)
        results.append({
            'feature': name,
            'spearman_corr': corr,
            'p_value': p_val,
            'abs_corr': abs(corr)
        })

    df = pd.DataFrame(results)
    df = df.sort_values('abs_corr', ascending=False)
    return df


def lasso_regression(X, y, feature_names):
    """
    LASSO 回归（带 LOOCV 调优）

    Args:
        X: 特征矩阵
        y: 目标变量
        feature_names: 特征名称列表

    Returns:
        dict: 包含模型、系数、最佳 alpha 等结果
    """
    # 标准化特征
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # LOOCV
    loo = LeaveOneOut()

    # LASSO CV 寻找最佳 alpha
    alphas = np.logspace(-4, 1, 50)
    lasso_cv = LassoCV(alphas=alphas, cv=loo, random_state=42)
    lasso_cv.fit(X_scaled, y)

    # 拟合最佳模型
    best_alpha = lasso_cv.alpha_
    lasso = Lasso(alpha=best_alpha, random_state=42)
    lasso.fit(X_scaled, y)

    # 计算 LOOCV 预测
    y_pred = np.zeros_like(y)
    for train_idx, test_idx in loo.split(X_scaled):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train = y[train_idx]
        model = Lasso(alpha=best_alpha, random_state=42)
        model.fit(X_train, y_train)
        y_pred[test_idx] = model.predict(X_test)

    # 特征系数
    coef_df = pd.DataFrame({
        'feature': feature_names,
        'coef': lasso.coef_,
        'abs_coef': np.abs(lasso.coef_)
    })
    coef_df = coef_df.sort_values('abs_coef', ascending=False)

    # 计算 R²
    r2 = lasso.score(X_scaled, y)
    loo_r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)

    return {
        'model': lasso,
        'scaler': scaler,
        'best_alpha': best_alpha,
        'coef_df': coef_df,
        'r2': r2,
        'loo_r2': loo_r2,
        'y_pred_loo': y_pred
    }


def random_forest_regression(X, y, feature_names, n_estimators=1000):
    """
    随机森林回归

    Args:
        X: 特征矩阵
        y: 目标变量
        feature_names: 特征名称列表
        n_estimators: 树的数量

    Returns:
        dict: 包含模型、特征重要性等结果
    """
    # LOOCV
    loo = LeaveOneOut()
    y_pred = np.zeros_like(y)

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]
        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=3,
            random_state=42,
            bootstrap=True
        )
        rf.fit(X_train, y_train)
        y_pred[test_idx] = rf.predict(X_test)

    # 拟合完整模型用于特征重要性
    rf_full = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=3,
        random_state=42,
        bootstrap=True
    )
    rf_full.fit(X, y)

    # 特征重要性
    importances = rf_full.feature_importances_
    imp_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    })
    imp_df = imp_df.sort_values('importance', ascending=False)

    # 排列重要性
    perm_importances = []
    for i in range(X.shape[1]):
        X_perm = X.copy()
        X_perm[:, i] = np.random.permutation(X_perm[:, i])
        y_pred_perm = rf_full.predict(X_perm)
        r2_orig = rf_full.score(X, y)
        r2_perm = 1 - np.sum((y - y_pred_perm) ** 2) / np.sum((y - np.mean(y)) ** 2)
        perm_importances.append(r2_orig - r2_perm)

    perm_imp_df = pd.DataFrame({
        'feature': feature_names,
        'permutation_importance': perm_importances
    })
    perm_imp_df = perm_imp_df.sort_values('permutation_importance', ascending=False)

    loo_r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)

    return {
        'model': rf_full,
        'importance_df': imp_df,
        'permutation_importance_df': perm_imp_df,
        'loo_r2': loo_r2,
        'y_pred_loo': y_pred
    }


def bayesian_regression(X, y, feature_names, num_samples=2000, num_chains=2):
    """
    贝叶斯线性回归（简化版：使用正态近似）

    Args:
        X: 特征矩阵
        y: 目标变量
        feature_names: 特征名称列表
        num_samples: 采样数
        num_chains: 链数

    Returns:
        dict: 包含后验统计结果
    """
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = np.hstack([np.ones((X_scaled.shape[0], 1)), X_scaled])  # add intercept

    # 贝叶斯回归：使用共轭先验 N(0, 1)
    n, p = X_scaled.shape

    # 先验
    prior_mean = np.zeros(p)
    prior_cov = np.eye(p) * 1.0  # 弱信息先验

    # 似然
    sigma2 = np.var(y) * 0.5 + 0.5  # 初始化
    XTX = X_scaled.T @ X_scaled
    XTy = X_scaled.T @ y

    # 后验
    posterior_cov = np.linalg.inv(XTX / sigma2 + np.linalg.inv(prior_cov))
    posterior_mean = posterior_cov @ (XTy / sigma2 + np.linalg.inv(prior_cov) @ prior_mean)

    # 采样（多元正态）
    samples = np.random.multivariate_normal(posterior_mean, posterior_cov, num_samples * num_chains)

    # 分析结果
    all_feature_names = ['intercept'] + feature_names
    results = []

    for i, name in enumerate(all_feature_names):
        samp = samples[:, i]
        prob_non_zero = np.mean(np.abs(samp) > 0.1 * np.std(samp))  # 效应量阈值
        results.append({
            'feature': name,
            'posterior_mean': np.mean(samp),
            'posterior_std': np.std(samp),
            'lower_95': np.percentile(samp, 2.5),
            'upper_95': np.percentile(samp, 97.5),
            'prob_non_zero': prob_non_zero
        })

    post_df = pd.DataFrame(results)

    # 预测（LOOCV）
    loo = LeaveOneOut()
    y_pred = np.zeros_like(y)

    for train_idx, test_idx in loo.split(X_scaled):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train = y[train_idx]

        XTX_train = X_train.T @ X_train
        XTy_train = X_train.T @ y_train
        post_cov_train = np.linalg.inv(XTX_train / sigma2 + np.linalg.inv(prior_cov))
        post_mean_train = post_cov_train @ (XTy_train / sigma2)

        y_pred[test_idx] = X_test @ post_mean_train

    loo_r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)

    return {
        'posterior_df': post_df,
        'samples': samples,
        'posterior_mean': posterior_mean,
        'posterior_cov': posterior_cov,
        'loo_r2': loo_r2,
        'y_pred_loo': y_pred,
        'scaler': scaler
    }


def run_all_models(X, y, feature_names):
    """
    运行所有模型

    Args:
        X: 特征矩阵
        y: 目标变量
        feature_names: 特征名称列表

    Returns:
        dict: 所有模型的结果
    """
    print("\n" + "=" * 60)
    print("回归建模阶段")
    print("=" * 60)

    results = {}

    # 1. Spearman 相关
    print("\n[1/4] Spearman 秩相关分析...")
    results['spearman'] = spearman_correlation(X, y, feature_names)
    print("  完成")

    # 2. LASSO
    print("\n[2/4] LASSO 回归...")
    results['lasso'] = lasso_regression(X, y, feature_names)
    print(f"  完成 (最佳 alpha: {results['lasso']['best_alpha']:.4f}, LOOCV R²: {results['lasso']['loo_r2']:.4f})")

    # 3. 随机森林
    print("\n[3/4] 随机森林回归...")
    results['random_forest'] = random_forest_regression(X, y, feature_names)
    print(f"  完成 (LOOCV R²: {results['random_forest']['loo_r2']:.4f})")

    # 4. 贝叶斯回归
    print("\n[4/4] 贝叶斯线性回归...")
    results['bayesian'] = bayesian_regression(X, y, feature_names)
    print(f"  完成 (LOOCV R²: {results['bayesian']['loo_r2']:.4f})")

    return results


if __name__ == '__main__':
    from data_preparation import prepare_full_dataset
    from feature_engineering import run_feature_engineering

    merged_df = prepare_full_dataset()
    X, y, feature_names, df_full = run_feature_engineering(merged_df)

    if y is not None and len(y) >= 3:
        results = run_all_models(X, y, feature_names)
    else:
        print("\n警告: ASR 数据不足，跳过建模")
