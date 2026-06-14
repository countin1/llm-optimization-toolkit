"""
数据加载器

加载 questions.json 并转换为训练格式。
支持两种答案生成模式：
- 快速模式（默认）：用 expected_hint 拼接编号列表
- LLM 模式：调用 API 生成完整参考答案（推荐用于微调）
"""

import json
import os
import time
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

    def to_chat_format(self, questions: List[Dict], system_prompt: str = "你是一位专业的数据分析和统计学专家。",
                       llm_client=None) -> List[Dict]:
        """
        转换为 Chat 格式

        Args:
            questions: 题目列表
            system_prompt: 系统提示词
            llm_client: 可选的 LLM 客户端，传入后用 LLM 生成完整参考答案
        """
        data = []
        for q in questions:
            if llm_client:
                answer = self._generate_answer_llm(q["question"], q.get("expected_hint", ""), llm_client)
            else:
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

    def to_alpaca_format(self, questions: List[Dict], llm_client=None) -> List[Dict]:
        """转换为 Alpaca 格式"""
        data = []
        for q in questions:
            if llm_client:
                answer = self._generate_answer_llm(q["question"], q.get("expected_hint", ""), llm_client)
            else:
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
        """快速模式：用 expected_hint 拼接编号列表"""
        if not expected_hint:
            return None
        hints = [h.strip() for h in expected_hint.replace("；", ";").split(";") if h.strip()]
        if not hints:
            return None
        return "\n".join(f"{i}. {hint}" for i, hint in enumerate(hints, 1))

    def _generate_answer_llm(self, question: str, expected_hint: str, client) -> Optional[str]:
        """
        LLM 模式：调用 API 生成完整参考答案

        Args:
            question: 题目
            expected_hint: 参考要点
            client: OpenAI 兼容客户端
        """
        hint_text = f"\n参考要点（必须涵盖）：{expected_hint}" if expected_hint else ""
        prompt = f"""请回答以下问题，要求准确、完整、结构清晰。{hint_text}

问题：{question}

请直接给出回答，不要重复问题本身。"""
        try:
            resp = client(prompt)
            if resp and not resp.startswith("[ERROR]"):
                return resp
            return self._generate_answer(expected_hint)  # fallback
        except Exception as e:
            print(f"  LLM 生成答案失败 (Q{question[:20]}...): {e}")
            return self._generate_answer(expected_hint)  # fallback
