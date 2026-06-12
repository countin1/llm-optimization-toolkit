"""Prompt 模板测试"""

import pytest
from promptforge.core.templates import PromptTemplates, ROLE, FORMAT, REASONING, FEWSHOT


def test_component_options():
    """测试组件选项"""
    assert "none" in ROLE.options
    assert "expert" in ROLE.options
    assert "professor" in ROLE.options
    assert "none" in FORMAT.options
    assert "structured" in FORMAT.options


def test_get_combinations():
    """测试组合生成"""
    templates = PromptTemplates()
    combos = templates.get_all_combinations(
        roles=["none", "expert"],
        formats=["none", "structured"],
        reasonings=["none", "cot"],
        fewshots=["none"],
    )
    assert len(combos) == 2 * 2 * 2 * 1  # 8 种组合


def test_config_to_name():
    """测试配置转名称"""
    templates = PromptTemplates()
    name = templates.config_to_name({
        "role": "expert",
        "format": "structured",
        "reasoning": "cot",
        "fewshot": "none",
    })
    assert name == "expert+structured+cot"


def test_baseline_name():
    """测试 baseline 名称"""
    templates = PromptTemplates()
    name = templates.config_to_name({
        "role": "none",
        "format": "none",
        "reasoning": "none",
        "fewshot": "none",
    })
    assert name == "baseline"
