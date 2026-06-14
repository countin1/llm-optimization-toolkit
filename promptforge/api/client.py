"""
OpenAI 兼容客户端

支持任何 OpenAI 兼容的 API（DeepSeek、MiMo、GPT 等）。
"""

import os
import time
from typing import Optional
from openai import OpenAI


class ModelClient:
    """模型客户端"""

    def __init__(self, api_key: str, base_url: str, model: str,
                 max_tokens: int = 2000, temperature: float = 0.7):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def __call__(self, prompt: str) -> str:
        """调用模型，异常由调用方处理"""
        resp = self.client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
            timeout=60,
        )
        return resp.choices[0].message.content


def create_client_from_env(model: str = "mimo-v2.5-pro") -> ModelClient:
    """
    从环境变量创建客户端

    环境变量:
        MIMO_API_KEY: API 密钥（必需）

    Args:
        model: 模型名称

    Returns:
        ModelClient 实例
    """
    api_key = os.environ.get("MIMO_API_KEY", "")
    if not api_key:
        raise ValueError("环境变量 MIMO_API_KEY 未设置")

    return ModelClient(
        api_key=api_key,
        base_url="https://token-plan-cn.xiaomimimo.com/v1",
        model=model,
    )
