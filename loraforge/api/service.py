"""
FastAPI 服务

提供 REST API 接口，支持远程调用微调和评测。
"""

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
    qlora: bool = False
    augment: int = 3


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
    data = loader.to_chat_format(questions)

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
    }


@app.post("/train")
async def train_model(request: TrainRequest):
    """启动训练"""
    # 实际使用时需要 GPU 环境
    return {
        "status": "submitted",
        "model": request.model_name,
        "epochs": request.epochs,
        "qlora": request.qlora,
        "message": "训练任务已提交（需要 GPU 环境执行）",
    }


@app.post("/eval")
async def evaluate_model(request: EvalRequest):
    """评测模型"""
    # 实际使用时加载模型并评测
    return {
        "status": "submitted",
        "base_model": request.base_model,
        "adapter_path": request.adapter_path,
        "message": "评测任务已提交",
    }


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
