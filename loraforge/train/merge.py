"""
模型合并

将 LoRA adapter 合并到基座模型，导出完整模型。
"""

import os
from typing import Optional


class ModelMerger:
    """模型合并器"""

    def __init__(self, base_model: str):
        self.base_model = base_model

    def merge(self, adapter_path: str, output_path: str,
              push_to_hub: bool = False, hub_name: Optional[str] = None) -> str:
        """
        合并 LoRA adapter 到基座模型

        Args:
            adapter_path: LoRA adapter 路径
            output_path: 输出路径
            push_to_hub: 是否推送到 HuggingFace Hub
            hub_name: Hub 仓库名

        Returns:
            输出路径
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        print(f"加载基座模型: {self.base_model}")
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model, trust_remote_code=True,
            torch_dtype=torch.float16, device_map="auto"
        )

        print(f"加载 LoRA adapter: {adapter_path}")
        model = PeftModel.from_pretrained(base_model, adapter_path)

        print("合并权重...")
        model = model.merge_and_unload()

        print(f"保存到: {output_path}")
        os.makedirs(output_path, exist_ok=True)
        model.save_pretrained(output_path)

        # 保存 tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.base_model, trust_remote_code=True)
        tokenizer.save_pretrained(output_path)

        if push_to_hub and hub_name:
            print(f"推送到 Hub: {hub_name}")
            model.push_to_hub(hub_name)
            tokenizer.push_to_hub(hub_name)

        print(f"合并完成! 模型保存在: {output_path}")
        return output_path


def export_to_gguf(model_path: str, output_path: str, quantize: str = "q4_0") -> str:
    """
    导出为 GGUF 格式（用于 llama.cpp / Ollama）

    Args:
        model_path: 模型路径
        output_path: 输出路径
        quantize: 量化方式（q4_0, q8_0, f16）

    Returns:
        输出路径

    Raises:
        NotImplementedError: 需要安装 llama-cpp-python
    """
    raise NotImplementedError(
        f"GGUF 导出需要安装 llama-cpp-python 或使用 convert 脚本。"
        f"参考: https://github.com/ggerganov/llama.cpp#prepare-and-quantize"
    )


def export_to_onnx(model_path: str, output_path: str) -> str:
    """
    导出为 ONNX 格式

    Args:
        model_path: 模型路径
        output_path: 输出路径

    Returns:
        输出路径

    Raises:
        NotImplementedError: 需要安装 optimum 库
    """
    raise NotImplementedError(
        "ONNX 导出需要安装 optimum 库。"
        "参考: https://huggingface.co/docs/optimum/onnxruntime/usage_guides/export_a_model"
    )
