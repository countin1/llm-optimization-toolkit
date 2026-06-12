"""
数据增强

对训练数据进行增强，生成不同表述方式的变体。
"""

from typing import Dict, List


class DataAugmentor:
    """数据增强器"""

    def __init__(self):
        self.variants = [
            lambda q: f"请详细解释：{q}",
            lambda q: f"{q}请举例说明。",
            lambda q: f"{q}请分点作答。",
            lambda q: f"从学术角度分析：{q}",
        ]

    def augment(self, data: List[Dict], n_variants: int = 3) -> List[Dict]:
        """
        增强数据

        Args:
            data: 原始数据（Chat 格式）
            n_variants: 每题生成的变体数

        Returns:
            增强后的数据
        """
        augmented = []
        for item in data:
            augmented.append(item)  # 保留原始

            question = item["messages"][1]["content"]
            answer = item["messages"][2]["content"]
            system = item["messages"][0]["content"]

            for variant_fn in self.variants[:n_variants]:
                new_question = variant_fn(question)
                augmented.append({
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": new_question},
                        {"role": "assistant", "content": answer},
                    ],
                    "metadata": {
                        **item["metadata"],
                        "augmented": True,
                    }
                })

        return augmented
