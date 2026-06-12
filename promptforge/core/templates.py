"""
Prompt 模板定义

定义 4 个正交维度的 prompt 组件：
- role: 角色设定
- format: 输出格式
- reasoning: 推理指令
- fewshot: 少样本示例
"""

from dataclasses import dataclass, field
from itertools import product
from typing import Dict, List, Optional


@dataclass
class PromptComponent:
    """单个 prompt 组件"""
    name: str
    options: Dict[str, str]  # key -> prompt text

    def get(self, key: str) -> str:
        return self.options.get(key, "")


# ========== 组件定义 ==========

ROLE = PromptComponent(
    name="role",
    options={
        "none": "",
        "expert": "你是一位资深的数据分析专家。",
        "professor": "你是一位统计学教授，擅长用通俗易懂的方式解释复杂概念。",
        "analyst": "你是一位严谨的量化分析师，所有结论必须有数据支撑。",
    }
)

FORMAT = PromptComponent(
    name="format",
    options={
        "none": "",
        "structured": "\n请用以下格式回答：\n1. 核心概念（一句话定义）\n2. 详细解释（2-3段）\n3. 实际应用举例",
        "bullet": "\n请分点作答，每点一个关键信息。",
        "academic": "\n请以学术论文的风格回答，包含定义、推导、结论。",
    }
)

REASONING = PromptComponent(
    name="reasoning",
    options={
        "none": "",
        "cot": "\n请逐步推理，最后给出答案。",
        "think": "\n请先分析问题的关键点，然后组织答案。",
        "verify": "\n回答后请自我检查，确保没有错误。",
    }
)

FEWSHOT = PromptComponent(
    name="fewshot",
    options={
        "none": "",
        "example_1": "\n\n示例：\n问：什么是均值？\n答：均值是一组数据的算术平均值，计算方法是将所有数值相加后除以数据个数。它是描述数据集中趋势的常用指标。",
    }
)


class PromptTemplates:
    """Prompt 模板管理器"""

    def __init__(self):
        self.components = {
            "role": ROLE,
            "format": FORMAT,
            "reasoning": REASONING,
            "fewshot": FEWSHOT,
        }

    def get_component(self, name: str) -> PromptComponent:
        return self.components[name]

    def get_options(self, component_name: str) -> List[str]:
        return list(self.components[component_name].options.keys())

    def get_all_combinations(self,
                             roles: Optional[List[str]] = None,
                             formats: Optional[List[str]] = None,
                             reasonings: Optional[List[str]] = None,
                             fewshots: Optional[List[str]] = None) -> List[Dict[str, str]]:
        """生成所有模板组合"""
        roles = roles or list(ROLE.options.keys())
        formats = formats or list(FORMAT.options.keys())
        reasonings = reasonings or list(REASONING.options.keys())
        fewshots = fewshots or list(FEWSHOT.options.keys())

        combos = []
        for r, f, re, fs in product(roles, formats, reasonings, fewshots):
            combos.append({
                "role": r,
                "format": f,
                "reasoning": re,
                "fewshot": fs,
            })
        return combos

    def config_to_name(self, config: Dict[str, str]) -> str:
        """配置转名称"""
        parts = [v for k, v in config.items() if v != "none"]
        return "+".join(parts) if parts else "baseline"
