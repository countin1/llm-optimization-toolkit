"""
FastAPI 服务

提供 REST API 接口，支持远程调用 Prompt 优化。
需要设置 MIMO_API_KEY 环境变量。
"""

import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
import numpy as np

app = FastAPI(title="PromptForge API", version="1.0.0")

# 全局客户端（惰性初始化）
_client = None


def _get_client():
    """获取或创建 API 客户端"""
    global _client
    if _client is None:
        from ..api.client import create_client_from_env
        _client = create_client_from_env()
    return _client


class SearchRequest(BaseModel):
    """搜索请求"""
    method: str = "grid"  # grid, bayesian, genetic
    questions: List[Dict]
    max_questions: Optional[int] = None
    iterations: int = 20  # bayesian/genetic
    generations: int = 10  # genetic


class SearchResponse(BaseModel):
    """搜索响应"""
    best_config: Dict[str, str]
    best_score: float
    method: str
    iterations: int
    summary: str


@app.get("/")
async def root():
    return {"message": "PromptForge API", "version": "1.0.0"}


@app.post("/search", response_model=SearchResponse)
async def search_prompts(request: SearchRequest):
    """搜索最优 prompt 模板（使用真实 API）"""
    from ..search.grid import GridSearch
    from ..search.bayesian import BayesianSearch
    from ..search.genetic import GeneticSearch

    client = _get_client()

    if request.method == "grid":
        searcher = GridSearch()
        result = searcher.search(request.questions, client,
                                max_questions=request.max_questions)
    elif request.method == "bayesian":
        searcher = BayesianSearch()
        result = searcher.search(request.questions, client,
                                n_iterations=request.iterations,
                                max_questions=request.max_questions)
    elif request.method == "genetic":
        searcher = GeneticSearch()
        result = searcher.search(request.questions, client,
                                generations=request.generations,
                                max_questions=request.max_questions)
    else:
        raise HTTPException(status_code=400, detail=f"未知方法: {request.method}")

    return SearchResponse(
        best_config=result.best_config,
        best_score=result.best_score,
        method=request.method,
        iterations=request.iterations,
        summary=result.summary(),
    )


@app.post("/evaluate")
async def evaluate_prompt(config: Dict[str, str], questions: List[Dict]):
    """评估单个 prompt 配置（使用真实 API）"""
    from ..core.builder import PromptBuilder
    from ..core.scorer import Scorer

    client = _get_client()
    builder = PromptBuilder()
    scorer = Scorer()

    scores = []
    for q in questions:
        prompt = builder.build(q["question"], config)
        answer = client(prompt)
        score = scorer.score(q["question"], answer, q.get("expected_hint", ""))
        scores.append(score)

    return {
        "config": config,
        "mean_score": round(float(np.mean(scores)), 2),
        "std_score": round(float(np.std(scores)), 2),
        "scores": scores,
    }


@app.get("/templates")
async def list_templates():
    """列出所有模板选项"""
    from ..core.templates import PromptTemplates
    templates = PromptTemplates()
    return {
        "components": {
            name: list(comp.options.keys())
            for name, comp in templates.components.items()
        }
    }
