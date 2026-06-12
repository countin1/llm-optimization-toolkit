"""
评分模块

支持两种评分模式：
- rule: 规则评分（基于关键词匹配，不花 token）
- ai: AI 评分（LLM-as-judge，需要 API）
"""

import re
from typing import Optional, Callable


def rule_score(answer: str, expected_hint: str = "") -> int:
    """
    规则评分（0-10 分）

    评分逻辑：
    - 基础分 3 分
    - 关键词命中：最高 +4 分
    - 回答长度：>200 字 +1 分，>500 字 +1 分
    - 结构化：有标题/列表/代码块 +1 分
    - 错误检测：包含 [ERROR] 扣分
    - 过短：<20 字最高 2 分
    """
    score = 3

    # 关键词命中
    if expected_hint:
        hints = [h.strip() for h in expected_hint.replace("；", ";").split(";") if h.strip()]
        if hints:
            hits = sum(1 for h in hints if h.lower() in answer.lower())
            score += round((hits / len(hints)) * 4)

    # 回答长度
    if len(answer) > 200:
        score += 1
    if len(answer) > 500:
        score += 1

    # 结构化标记
    if any(marker in answer for marker in ["##", "| ", "1.", "- ", "```"]):
        score += 1

    # 错误检测
    if "[ERROR]" in answer:
        score = 1

    # 过短
    if len(answer) < 20:
        score = min(score, 2)

    return max(0, min(10, score))


def create_ai_scorer(client, model_name: str) -> Callable[[str, str, str], int]:
    """
    创建 AI 评分函数

    Args:
        client: OpenAI 客户端
        model_name: 评分模型名称

    Returns:
        AI 评分函数
    """
    def ai_score(question: str, answer: str, expected_hint: str = "") -> int:
        hint_text = f"\n参考要点：{expected_hint}" if expected_hint else ""
        prompt = f"""你是一个严格的模型回答评估专家。请根据以下问题和回答，给出一个0-10的整数分数。

评估标准：
- 准确性（答案是否正确，关键概念有无错误）
- 完整性（是否回答了问题的核心要点）
- 清晰度（表达是否清晰，逻辑是否连贯）
{hint_text}

问题：{question}

回答：{answer}

请只输出一个整数分数（0-10），不要有任何其他文字。"""
        try:
            resp = client.chat.completions.create(
                model=model_name, max_tokens=10, temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            match = re.search(r'\d+', resp.choices[0].message.content.strip())
            if match:
                return min(max(int(match.group()), 0), 10)
            return 5
        except Exception as e:
            print(f"    AI评分失败: {e}")
            return 5

    return ai_score


class Scorer:
    """评分管理器"""

    def __init__(self, mode: str = "rule", client=None, model_name: str = ""):
        """
        Args:
            mode: "rule" 或 "ai"
            client: OpenAI 客户端（ai 模式需要）
            model_name: 评分模型名称（ai 模式需要）
        """
        self.mode = mode
        if mode == "ai" and client:
            self._score_fn = create_ai_scorer(client, model_name)
        else:
            self._score_fn = None

    def score(self, question: str, answer: str, expected_hint: str = "") -> int:
        """评分"""
        if self.mode == "ai" and self._score_fn:
            return self._score_fn(question, answer, expected_hint)
        return rule_score(answer, expected_hint)
