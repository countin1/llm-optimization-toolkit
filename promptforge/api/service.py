"""
FastAPI 服务

提供 REST API 接口，支持远程调用 Prompt 优化。
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
import json

app = FastAPI(title="PromptForge API", version="1.0.0")


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
    """搜索最优 prompt 模板"""
    from ..core.templates import PromptTemplates
    from ..core.builder import PromptBuilder
    from ..core.scorer import Scorer
    from ..search.grid import GridSearch
    from ..search.bayesian import BayesianSearch
    from ..search.genetic import GeneticSearch

    # 创建模拟模型函数（实际使用时替换为真实 API 调用）
    def mock_model_fn(prompt: str) -> str:
        return f"模拟回答：{prompt[:50]}..."

    searcher = None
    if request.method == "grid":
        searcher = GridSearch()
        result = searcher.search(request.questions, mock_model_fn,
                                max_questions=request.max_questions)
    elif request.method == "bayesian":
        searcher = BayesianSearch()
        result = searcher.search(request.questions, mock_model_fn,
                                n_iterations=request.iterations,
                                max_questions=request.max_questions)
    elif request.method == "genetic":
        searcher = GeneticSearch()
        result = searcher.search(request.questions, mock_model_fn,
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
    """评估单个 prompt 配置"""
    from ..core.builder import PromptBuilder
    from ..core.scorer import Scorer

    builder = PromptBuilder()
    scorer = Scorer()

    scores = []
    for q in questions:
        prompt = builder.build(q["question"], config)
        # 实际使用时调用真实模型
        answer = f"模拟回答"
        score = scorer.score(q["question"], answer, q.get("expected_hint", ""))
        scores.append(score)

    import numpy as np
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
