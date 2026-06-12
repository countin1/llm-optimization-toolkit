"""
统计检验模块

提供配对 t 检验、Cohen's d、Wilcoxon 符号秩检验等。
"""

import numpy as np
from scipy import stats as sp_stats
from typing import Dict, List, Tuple


def paired_t_test(scores_a: np.ndarray, scores_b: np.ndarray) -> Dict:
    """
    配对 t 检验

    Args:
        scores_a: 策略 A 的得分
        scores_b: 策略 B 的得分

    Returns:
        检验结果字典
    """
    t_stat, p_val = sp_stats.ttest_rel(scores_a, scores_b)
    return {
        "t_stat": round(t_stat, 3),
        "p_value": round(p_val, 4),
        "significant": "***" if p_val < 0.01 else ("**" if p_val < 0.05 else ("*" if p_val < 0.1 else "")),
    }


def cohens_d(scores_a: np.ndarray, scores_b: np.ndarray) -> Dict:
    """
    Cohen's d 效应量（配对版本）

    Args:
        scores_a: 策略 A 的得分
        scores_b: 策略 B 的得分

    Returns:
        效应量字典
    """
    diffs = scores_a - scores_b
    d = diffs.mean() / diffs.std() if diffs.std() > 0 else 0
    magnitude = "大" if abs(d) >= 0.8 else ("中" if abs(d) >= 0.5 else ("小" if abs(d) >= 0.2 else "可忽略"))
    return {
        "d": round(d, 3),
        "magnitude": magnitude,
    }


def wilcoxon_test(scores_a: np.ndarray, scores_b: np.ndarray) -> Dict:
    """
    Wilcoxon 符号秩检验

    Args:
        scores_a: 策略 A 的得分
        scores_b: 策略 B 的得分

    Returns:
        检验结果字典
    """
    if len(scores_a) < 6:
        return {"w_stat": None, "p_value": None, "significant": "", "note": "样本太少"}

    try:
        w_stat, p_val = sp_stats.wilcoxon(scores_a, scores_b)
        return {
            "w_stat": round(w_stat, 1),
            "p_value": round(p_val, 4),
            "significant": "***" if p_val < 0.01 else ("**" if p_val < 0.05 else ("*" if p_val < 0.1 else "")),
        }
    except ValueError:
        return {"w_stat": None, "p_value": None, "significant": "", "note": "差值全为0"}


def dimension_analysis(results: List[Dict], score_key_a: str, score_key_b: str) -> Dict[str, Dict]:
    """
    按维度分析

    Args:
        results: 评测结果列表（每项需含 dimension 字段）
        score_key_a: 策略 A 的分数字段名
        score_key_b: 策略 B 的分数字段名

    Returns:
        {维度名: 分析结果}
    """
    dims = sorted(set(r["dimension"] for r in results))
    analysis = {}

    for dim in dims:
        dim_results = [r for r in results if r["dimension"] == dim]
        a_scores = np.array([r[score_key_a] for r in dim_results])
        b_scores = np.array([r[score_key_b] for r in dim_results])
        diffs = a_scores - b_scores

        info = {
            "n": len(dim_results),
            "a_mean": round(float(a_scores.mean()), 2),
            "b_mean": round(float(b_scores.mean()), 2),
            "diff_mean": round(float(diffs.mean()), 2),
            "a_wins": int((diffs > 0).sum()),
            "b_wins": int((diffs < 0).sum()),
            "ties": int((diffs == 0).sum()),
        }

        # 统计检验
        if len(dim_results) >= 6:
            wt = wilcoxon_test(a_scores, b_scores)
            info.update(wt)
        else:
            info["significant"] = ""

        analysis[dim] = info

    return analysis


def overall_analysis(results: List[Dict], score_key_a: str, score_key_b: str) -> Dict:
    """
    整体分析

    Args:
        results: 评测结果列表
        score_key_a: 策略 A 的分数字段名
        score_key_b: 策略 B 的分数字段名

    Returns:
        整体分析结果
    """
    a_scores = np.array([r[score_key_a] for r in results])
    b_scores = np.array([r[score_key_b] for r in results])

    tt = paired_t_test(a_scores, b_scores)
    cd = cohens_d(a_scores, b_scores)

    return {
        "n": len(results),
        "a_mean": round(float(a_scores.mean()), 2),
        "a_std": round(float(a_scores.std()), 2),
        "b_mean": round(float(b_scores.mean()), 2),
        "b_std": round(float(b_scores.std()), 2),
        "diff_mean": round(float((a_scores - b_scores).mean()), 2),
        "paired_ttest": tt,
        "cohens_d": cd,
    }
