"""评分函数测试"""

import pytest
from promptforge.core.scorer import rule_score


def test_rule_score_basic():
    """测试基础评分"""
    score = rule_score("这是一个测试回答，包含一些内容。")
    assert 0 <= score <= 10


def test_rule_score_with_hints():
    """测试关键词命中评分"""
    answer = "均值是算术平均值，标准差衡量离散程度。"
    hint = "均值；标准差"
    score = rule_score(answer, hint)
    assert score >= 5  # 命中两个关键词


def test_rule_score_short_answer():
    """测试过短回答"""
    score = rule_score("短")
    assert score <= 2


def test_rule_score_error():
    """测试错误标记"""
    score = rule_score("[ERROR] 调用失败")
    assert score == 1


def test_rule_score_structured():
    """测试结构化回答加分"""
    answer = "## 标题\n\n1. 第一点\n2. 第二点\n\n详细内容。" * 20
    score = rule_score(answer)
    assert score >= 5
