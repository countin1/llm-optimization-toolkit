"""评测模块测试"""

import pytest
import numpy as np
from loraforge.eval.stats import paired_t_test, cohens_d, rule_score


def test_rule_score():
    """测试规则评分"""
    score = rule_score("均值是算术平均值", "均值；算术平均")
    assert score >= 5


def test_paired_t_test():
    """测试配对 t 检验"""
    a = np.array([7, 8, 6, 7, 8])
    b = np.array([6, 7, 5, 6, 7])
    result = paired_t_test(a, b)
    assert "t_stat" in result
    assert "p_value" in result
    assert "significant" in result


def test_cohens_d():
    """测试 Cohen's d"""
    a = np.array([7, 8, 6, 7, 8])
    b = np.array([6, 7, 5, 6, 7])
    result = cohens_d(a, b)
    assert "d" in result
    assert "magnitude" in result
    assert result["d"] > 0  # a 比 b 高


def test_cohens_d_magnitude():
    """测试效应量等级"""
    # 大效应
    a = np.array([9, 9, 9, 9, 9])
    b = np.array([5, 5, 5, 5, 5])
    result = cohens_d(a, b)
    assert result["magnitude"] == "大"
