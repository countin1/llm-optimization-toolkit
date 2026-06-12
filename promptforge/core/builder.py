"""
Prompt 构建器

将模板组件组合成完整的 prompt。
"""

from typing import Dict, Optional
from .templates import PromptTemplates


class PromptBuilder:
    """Prompt 构建器"""

    def __init__(self, templates: Optional[PromptTemplates] = None):
        self.templates = templates or PromptTemplates()

    def build(self, question: str, config: Dict[str, str]) -> str:
        """
        根据配置构建 prompt

        Args:
            question: 用户问题
            config: {"role": "expert", "format": "structured", ...}

        Returns:
            组合后的完整 prompt
        """
        parts = []

        # 1. 角色设定
        role = self.templates.get_component("role").get(config.get("role", "none"))
        if role:
            parts.append(role)

        # 2. 问题
        parts.append(question)

        # 3. 输出格式
        fmt = self.templates.get_component("format").get(config.get("format", "none"))
        if fmt:
            parts.append(fmt)

        # 4. 推理指令
        reasoning = self.templates.get_component("reasoning").get(config.get("reasoning", "none"))
        if reasoning:
            parts.append(reasoning)

        # 5. Few-shot 示例
        fewshot = self.templates.get_component("fewshot").get(config.get("fewshot", "none"))
        if fewshot:
            parts.append(fewshot)

        return "\n".join(parts)

    def build_baseline(self, question: str) -> str:
        """构建 baseline prompt（无任何增强）"""
        return self.build(question, {
            "role": "none",
            "format": "none",
            "reasoning": "none",
            "fewshot": "none",
        })

    def build_cot(self, question: str) -> str:
        """构建 Chain-of-Thought prompt"""
        return self.build(question, {
            "role": "none",
            "format": "none",
            "reasoning": "cot",
            "fewshot": "none",
        })
