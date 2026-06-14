"""
FastAPI 服务

提供 REST API 接口，支持远程调用微调和评测。
需要 GPU 环境运行 /train 和 /eval 端点。
"""

import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional

app = FastAPI(title="LoRAForge API", version="1.0.0")


class TrainRequest(BaseModel):
    """训练请求"""
    model_name: str = "Qwen/Qwen2.5-7B"
    epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 2e-4
    lora_r: int = 8
    lora_alpha: int = 16
    qlora: bool = False
    augment: int = 3
    output_dir: str = "outputs/adapters"


class EvalRequest(BaseModel):
    """评测请求"""
    base_model: str
    adapter_path: str
    max_samples: Optional[int] = None


class DataRequest(BaseModel):
    """数据准备请求"""
    augment: int = 3
    split_ratio: float = 0.8
    format: str = "chat"
    use_llm: bool = False  # 是否用 LLM 生成完整参考答案


@app.get("/")
async def root():
    return {"message": "LoRAForge API", "version": "1.0.0"}


@app.post("/data/prepare")
async def prepare_data(request: DataRequest):
    """准备训练数据"""
    from ..data.loader import DataLoader
    from ..data.augment import DataAugmentor
    from ..data.splitter import DataSplitter

    loader = DataLoader()
    questions = loader.load()

    # 可选用 LLM 生成完整参考答案
    llm_client = None
    if request.use_llm:
        try:
            from ..api.service import _get_client_from_env
            llm_client = _get_client_from_env()
        except Exception:
            pass

    data = loader.to_chat_format(questions, llm_client=llm_client)

    if request.augment > 0:
        augmentor = DataAugmentor()
        data = augmentor.augment(data, request.augment)

    splitter = DataSplitter()
    train_data, test_data = splitter.split(data, request.split_ratio)

    return {
        "total": len(data),
        "train": len(train_data),
        "test": len(test_data),
        "augment": request.augment,
        "llm_generated": llm_client is not None,
    }


@app.post("/train")
async def train_model(request: TrainRequest):
    """启动真实训练（需要 GPU 环境）"""
    from ..data.loader import DataLoader
    from ..data.augment import DataAugmentor
    from ..data.splitter import DataSplitter

    # 准备数据
    loader = DataLoader()
    questions = loader.load()
    data = loader.to_chat_format(questions)

    if request.augment > 0:
        augmentor = DataAugmentor()
        data = augmentor.augment(data, request.augment)

    splitter = DataSplitter()
    train_data, test_data = splitter.split(data)

    # 选择训练器
    if request.qlora:
        from ..train.qlora import QLoRATrainer
        trainer = QLoRATrainer(request.model_name, r=request.lora_r, lora_alpha=request.lora_alpha)
    else:
        from ..train.lora import LoRATrainer
        trainer = LoRATrainer(request.model_name, r=request.lora_r, lora_alpha=request.lora_alpha)

    # 执行训练
    try:
        result = trainer.train(
            train_data, test_data,
            epochs=request.epochs,
            batch_size=request.batch_size,
            lr=request.learning_rate,
            output_dir=request.output_dir,
        )
        return {
            "status": "completed",
            "model": request.model_name,
            "qlora": request.qlora,
            "train_loss": result["train_loss"],
            "save_path": result["save_path"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"训练失败: {e}")


@app.post("/eval")
async def evaluate_model(request: EvalRequest):
    """评测模型（需要 GPU 环境）"""
    from ..eval.compare import ModelComparator
    from ..data.loader import DataLoader
    from ..data.splitter import DataSplitter

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        # 加载基座模型
        tokenizer = AutoTokenizer.from_pretrained(request.base_model, trust_remote_code=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            request.base_model, trust_remote_code=True,
            torch_dtype=torch.float16, device_map="auto"
        )

        # 加载 adapter
        model = PeftModel.from_pretrained(base_model, request.adapter_path)
        model.eval()

        # 加载测试数据
        loader = DataLoader()
        questions = loader.load()

        if request.max_samples:
            questions = questions[:request.max_samples]

        # 评测（简化版：用 rule_score 评估模型生成）
        from ..eval.stats import rule_score
        import numpy as np

        scores = []
        details = []
        for q in questions:
            prompt = f"问题：{q['question']}\n请回答："
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=512, do_sample=False)
            answer = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            score = rule_score(answer, q.get("expected_hint", ""))
            scores.append(score)
            details.append({"id": q["id"], "dimension": q["dimension"], "score": score})

        return {
            "status": "completed",
            "base_model": request.base_model,
            "adapter_path": request.adapter_path,
            "mean_score": round(float(np.mean(scores)), 2),
            "std_score": round(float(np.std(scores)), 2),
            "n_samples": len(scores),
            "details": details[:10],  # 返回前 10 条
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"评测失败: {e}")


@app.get("/models")
async def list_models():
    """列出支持的模型"""
    return {
        "models": [
            {"name": "Qwen/Qwen2.5-3B", "lora_vram": "8GB", "qlora_vram": "6GB"},
            {"name": "Qwen/Qwen2.5-7B", "lora_vram": "16GB", "qlora_vram": "8GB"},
            {"name": "Qwen/Qwen2.5-14B", "lora_vram": "28GB", "qlora_vram": "12GB"},
            {"name": "LLaMA-3-8B", "lora_vram": "16GB", "qlora_vram": "8GB"},
        ]
    }
