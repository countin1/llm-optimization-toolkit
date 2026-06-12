"""搜索算法测试"""

import pytest
from promptforge.core.templates import PromptTemplates
from promptforge.core.builder import PromptBuilder


def test_builder_baseline():
    """测试 baseline 构建"""
    builder = PromptBuilder()
    prompt = builder.build_baseline("什么是均值？")
    assert "什么是均值？" in prompt
    assert "逐步推理" not in prompt


def test_builder_cot():
    """测试 CoT 构建"""
    builder = PromptBuilder()
    prompt = builder.build_cot("什么是均值？")
    assert "什么是均值？" in prompt
    assert "逐步推理" in prompt


def test_builder_with_role():
    """测试带角色的构建"""
    builder = PromptBuilder()
    prompt = builder.build("什么是均值？", {
        "role": "professor",
        "format": "structured",
        "reasoning": "cot",
        "fewshot": "none",
    })
    assert "统计学教授" in prompt
    assert "核心概念" in prompt
    assert "逐步推理" in prompt
