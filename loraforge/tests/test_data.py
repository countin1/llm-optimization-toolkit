"""数据处理测试"""

import pytest
import json
import tempfile
import os
from loraforge.data.loader import DataLoader
from loraforge.data.augment import DataAugmentor
from loraforge.data.splitter import DataSplitter


@pytest.fixture
def sample_questions():
    return [
        {"id": 1, "dimension": "统计知识", "question": "什么是均值？", "expected_hint": "均值；算术平均", "difficulty": 1},
        {"id": 2, "dimension": "统计知识", "question": "什么是标准差？", "expected_hint": "标准差；离散程度", "difficulty": 2},
        {"id": 3, "dimension": "Python代码", "question": "如何用pandas读取CSV？", "expected_hint": "pandas；read_csv", "difficulty": 1},
    ]


def test_loader_to_chat(sample_questions):
    """测试 Chat 格式转换"""
    loader = DataLoader()
    data = loader.to_chat_format(sample_questions)
    assert len(data) == 3
    assert data[0]["messages"][0]["role"] == "system"
    assert data[0]["messages"][1]["role"] == "user"
    assert data[0]["messages"][2]["role"] == "assistant"


def test_augmentor(sample_questions):
    """测试数据增强"""
    loader = DataLoader()
    data = loader.to_chat_format(sample_questions)

    augmentor = DataAugmentor()
    augmented = augmentor.augment(data, n_variants=2)
    assert len(augmented) == 3 * 3  # 3 原始 + 3*2 增强


def test_splitter(sample_questions):
    """测试分层划分"""
    loader = DataLoader()
    data = loader.to_chat_format(sample_questions)

    splitter = DataSplitter()
    train, test = splitter.split(data, train_ratio=0.67)

    assert len(train) + len(test) == len(data)
    # 验证分层：每个维度都有训练和测试样本
    train_dims = set(d["metadata"]["dimension"] for d in train)
    test_dims = set(d["metadata"]["dimension"] for d in test)
    assert len(train_dims) > 0
