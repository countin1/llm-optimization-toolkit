"""
统计检验模块

配对 t 检验 + Cohen's d 效应量 + 规则评分。

注意：rule_score/paired_t_test/cohens_d 在 promptforge/core/scorer.py 和
promptforge/analysis/stats.py 中有相同实现。如需修改，请同步更新。
"""

import numpy as np
from scipy import stats as sp_stats
from typing import Dict


def paired_t_test(scores_a: np.ndarray, scores_b: np.ndarray) -> Dict:
    """配对 t 检验"""
    t_stat, p_val = sp_stats.ttest_rel(scores_a, scores_b)
    return {
        "t_stat": round(t_stat, 3),
        "p_value": round(p_val, 4),
        "significant": "***" if p_val < 0.01 else ("**" if p_val < 0.05 else ("*" if p_val < 0.1 else "")),
    }


def cohens_d(scores_a: np.ndarray, scores_b: np.ndarray) -> Dict:
    """Cohen's d 效应量"""
    diffs = scores_a - scores_b
    d = diffs.mean() / diffs.std() if diffs.std() > 0 else 0
    magnitude = "大" if abs(d) >= 0.8 else ("中" if abs(d) >= 0.5 else ("小" if abs(d) >= 0.2 else "可忽略"))
    return {"d": round(d, 3), "magnitude": magnitude}


def rule_score(answer: str, expected_hint: str = "") -> int:
    """规则评分"""
    score = 3
    if expected_hint:
        hints = [h.strip() for h in expected_hint.replace("；", ";").split(";") if h.strip()]
        if hints:
            hits = sum(1 for h in hints if h.lower() in answer.lower())
            score += round((hits / len(hints)) * 4)
    if len(answer) > 200: score += 1
    if len(answer) > 500: score += 1
    if any(m in answer for m in ["##", "| ", "1.", "- ", "```"]): score += 1
    if "[ERROR]" in answer: score = 1
    if len(answer) < 20: score = min(score, 2)
    return max(0, min(10, score))
