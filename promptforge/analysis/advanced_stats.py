"""
高级统计分析

- Bootstrap 置信区间
- 交叉验证
- 功效分析
- 多重比较校正
"""

import numpy as np
from scipy import stats as sp_stats
from typing import Dict, List, Tuple


def bootstrap_ci(scores: np.ndarray, n_bootstrap: int = 1000, ci: float = 0.95) -> Dict:
    """
    Bootstrap 置信区间

    Args:
        scores: 得分数组
        n_bootstrap: 重采样次数
        ci: 置信水平

    Returns:
        置信区间字典
    """
    n = len(scores)
    means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(scores, size=n, replace=True)
        means.append(np.mean(sample))

    means = np.array(means)
    alpha = (1 - ci) / 2
    lower = np.percentile(means, alpha * 100)
    upper = np.percentile(means, (1 - alpha) * 100)

    return {
        "mean": round(float(np.mean(scores)), 2),
        "ci_lower": round(float(lower), 2),
        "ci_upper": round(float(upper), 2),
        "ci_level": ci,
    }


def cross_validate(questions: List[Dict], model_fn, scorer, builder,
                   config: Dict, n_folds: int = 5) -> Dict:
    """
    K 折交叉验证

    Args:
        questions: 评测题目
        model_fn: 模型调用函数
        scorer: 评分器
        builder: Prompt 构建器
        config: Prompt 配置
        n_folds: 折数

    Returns:
        交叉验证结果
    """
    np.random.seed(42)
    indices = np.random.permutation(len(questions))
    fold_size = len(questions) // n_folds

    fold_scores = []
    for fold in range(n_folds):
        start = fold * fold_size
        end = start + fold_size if fold < n_folds - 1 else len(questions)
        test_indices = indices[start:end]

        scores = []
        for idx in test_indices:
            q = questions[idx]
            prompt = builder.build(q["question"], config)
            answer = model_fn(prompt)
            score = scorer.score(q["question"], answer, q.get("expected_hint", ""))
            scores.append(score)

        fold_scores.append(float(np.mean(scores)))

    return {
        "fold_scores": fold_scores,
        "mean": round(float(np.mean(fold_scores)), 2),
        "std": round(float(np.std(fold_scores)), 2),
        "n_folds": n_folds,
    }


def power_analysis(effect_size: float, alpha: float = 0.05, power: float = 0.8) -> Dict:
    """
    功效分析：计算所需样本量

    Args:
        effect_size: 预期效应量 (Cohen's d)
        alpha: 显著性水平
        power: 统计功效

    Returns:
        所需样本量
    """
    from scipy.stats import norm

    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(power)

    # 配对 t 检验的样本量公式
    n = ((z_alpha + z_beta) / effect_size) ** 2

    return {
        "effect_size": effect_size,
        "alpha": alpha,
        "power": power,
        "required_n": int(np.ceil(n)),
    }


def bonferroni_correction(p_values: List[float], alpha: float = 0.05) -> Dict:
    """
    Bonferroni 多重比较校正

    Args:
        p_values: 原始 p 值列表
        alpha: 显著性水平

    Returns:
        校正后的结果
    """
    n_comparisons = len(p_values)
    adjusted_alpha = alpha / n_comparisons

    results = []
    for i, p in enumerate(p_values):
        results.append({
            "comparison": i + 1,
            "p_value": round(p, 4),
            "adjusted_alpha": round(adjusted_alpha, 4),
            "significant": p < adjusted_alpha,
        })

    return {
        "n_comparisons": n_comparisons,
        "original_alpha": alpha,
        "adjusted_alpha": round(adjusted_alpha, 4),
        "results": results,
    }


def detectable_effect_size(n: int, alpha: float = 0.05, power: float = 0.8) -> Dict:
    """
    给定样本量，计算可检测的最小效应量

    Args:
        n: 样本量
        alpha: 显著性水平
        power: 统计功效

    Returns:
        可检测效应量
    """
    from scipy.stats import norm

    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(power)

    d = (z_alpha + z_beta) / np.sqrt(n)

    magnitude = "大" if d <= 0.8 else ("中" if d <= 0.5 else ("小" if d <= 0.2 else "可忽略"))

    return {
        "n": n,
        "detectable_d": round(d, 3),
        "magnitude": magnitude,
        "interpretation": f"样本量 {n} 可检测 Cohen's d >= {d:.3f} 的效应",
    }
