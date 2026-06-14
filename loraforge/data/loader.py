"""
数据加载器

加载 questions.json 并转换为训练格式。
"""

import json
import os
from typing import Dict, List, Optional


class DataLoader:
    """数据加载器"""

    def __init__(self, questions_path: str = "data/questions.json"):
        self.questions_path = questions_path

    def load(self) -> List[Dict]:
        """加载题目"""
        with open(self.questions_path, "r", encoding="utf-8") as f:
            questions = json.load(f)
        return questions

    def to_chat_format(self, questions: List[Dict], system_prompt: str = "你是一位专业的数据分析和统计学专家。") -> List[Dict]:
        """转换为 Chat 格式"""
        data = []
        for q in questions:
            answer = self._generate_answer(q.get("expected_hint", ""))
            if answer:
                data.append({
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": q["question"]},
                        {"role": "assistant", "content": answer},
                    ],
                    "metadata": {
                        "id": q["id"],
                        "dimension": q["dimension"],
                        "difficulty": q.get("difficulty", 1),
                    }
                })
        return data

    def to_alpaca_format(self, questions: List[Dict]) -> List[Dict]:
        """转换为 Alpaca 格式"""
        data = []
        for q in questions:
            answer = self._generate_answer(q.get("expected_hint", ""))
            if answer:
                data.append({
                    "instruction": q["question"],
                    "input": "",
                    "output": answer,
                    "metadata": {
                        "id": q["id"],
                        "dimension": q["dimension"],
                        "difficulty": q.get("difficulty", 1),
                    }
                })
        return data

    def _generate_answer(self, expected_hint: str) -> Optional[str]:
        """
        根据 expected_hint 构造参考答案

        注意：生成的是 hint 的编号拼接（简化版），不是完整回答。
        如需更高质量的训练数据，建议用 LLM 生成完整参考答案。
        """
        if not expected_hint:
            return None
        hints = [h.strip() for h in expected_hint.replace("；", ";").split(";") if h.strip()]
        if not hints:
            return None
        return "\n".join(f"{i}. {hint}" for i, hint in enumerate(hints, 1))
