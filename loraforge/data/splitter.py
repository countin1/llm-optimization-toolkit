"""
数据划分

按维度分层划分训练集和测试集。
"""

import random
import warnings
from typing import Dict, List, Tuple


class DataSplitter:
    """数据划分器"""

    def split(self, data: List[Dict], train_ratio: float = 0.8, seed: int = 42) -> Tuple[List[Dict], List[Dict]]:
        """
        分层划分

        Args:
            data: 数据列表
            train_ratio: 训练集比例
            seed: 随机种子

        Returns:
            (train_data, test_data)
        """
        random.seed(seed)

        # 按维度分层
        by_dim = {}
        for item in data:
            dim = item["metadata"]["dimension"]
            by_dim.setdefault(dim, []).append(item)

        train_data = []
        test_data = []

        for dim, items in by_dim.items():
            if len(items) < 2:
                warnings.warn(
                    f"维度 '{dim}' 只有 {len(items)} 条数据，无法划分训练/测试集，全部放入训练集",
                    UserWarning, stacklevel=2,
                )
                train_data.extend(items)
                continue

            random.shuffle(items)
            split_idx = int(len(items) * train_ratio)
            # 确保训练集和测试集都至少有 1 条
            split_idx = max(1, min(split_idx, len(items) - 1))
            train_data.extend(items[:split_idx])
            test_data.extend(items[split_idx:])

        random.shuffle(train_data)
        random.shuffle(test_data)

        return train_data, test_data
