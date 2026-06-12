"""
成本优化器

在给定 API 预算内找到最优 prompt 配置。
"""

from typing import Dict, List, Optional
import numpy as np


class CostOptimizer:
    """成本优化器"""

    def __init__(self, cost_per_token: float = 0.0001):
        """
        Args:
            cost_per_token: 每 token 成本（元）
        """
        self.cost_per_token = cost_per_token

    def estimate_cost(self, n_questions: int, n_templates: int,
                      avg_tokens_per_call: int = 500) -> float:
        """
        估算 API 成本

        Args:
            n_questions: 题目数
            n_templates: 模板数
            avg_tokens_per_call: 每次调用平均 token 数

        Returns:
            预估成本（元）
        """
        total_calls = n_questions * n_templates
        total_tokens = total_calls * avg_tokens_per_call
        return total_tokens * self.cost_per_token

    def optimize_budget(self, questions: List[Dict], budget: float,
                        model_fn, scorer, builder, templates,
                        strategy: str = "greedy") -> Dict:
        """
        在预算内搜索最优配置

        Args:
            questions: 评测题目
            budget: 预算（元）
            model_fn: 模型调用函数
            scorer: 评分器
            builder: Prompt 构建器
            templates: 模板管理器
            strategy: 策略（greedy/random/bayesian）

        Returns:
            优化结果
        """
        # 计算预算允许的最大调用次数
        max_calls = int(budget / (self.cost_per_call * 500))  # 假设每次 500 tokens

        if strategy == "greedy":
            return self._greedy_search(questions, max_calls, model_fn, scorer, builder, templates)
        elif strategy == "random":
            return self._random_search(questions, max_calls, model_fn, scorer, builder, templates)
        else:
            return {"error": f"未知策略: {strategy}"}

    def _greedy_search(self, questions, max_calls, model_fn, scorer, builder, templates):
        """贪心搜索：先测每个组件的最佳选项"""
        best_config = {"role": "none", "format": "none", "reasoning": "none", "fewshot": "none"}
        calls_used = 0

        for param in ["role", "format", "reasoning", "fewshot"]:
            options = templates.get_options(param)
            best_option = "none"
            best_score = 0

            for option in options:
                if calls_used >= max_calls:
                    break

                test_config = best_config.copy()
                test_config[param] = option

                scores = []
                for q in questions:
                    if calls_used >= max_calls:
                        break
                    prompt = builder.build(q["question"], test_config)
                    answer = model_fn(prompt)
                    score = scorer.score(q["question"], answer, q.get("expected_hint", ""))
                    scores.append(score)
                    calls_used += 1

                if scores:
                    mean_score = np.mean(scores)
                    if mean_score > best_score:
                        best_score = mean_score
                        best_option = option

            best_config[param] = best_option

        return {
            "config": best_config,
            "calls_used": calls_used,
            "budget_used": calls_used * self.cost_per_token * 500,
        }

    def _random_search(self, questions, max_calls, model_fn, scorer, builder, templates):
        """随机搜索：随机采样配置"""
        best_config = None
        best_score = 0
        calls_used = 0

        while calls_used < max_calls:
            # 随机生成配置
            config = {}
            for param in ["role", "format", "reasoning", "fewshot"]:
                options = templates.get_options(param)
                config[param] = np.random.choice(options)

            # 评估（用部分题目）
            eval_questions = questions[:min(5, len(questions))]
            scores = []
            for q in eval_questions:
                if calls_used >= max_calls:
                    break
                prompt = builder.build(q["question"], config)
                answer = model_fn(prompt)
                score = scorer.score(q["question"], answer, q.get("expected_hint", ""))
                scores.append(score)
                calls_used += 1

            if scores:
                mean_score = np.mean(scores)
                if mean_score > best_score:
                    best_score = mean_score
                    best_config = config

        return {
            "config": best_config,
            "best_score": best_score,
            "calls_used": calls_used,
            "budget_used": calls_used * self.cost_per_token * 500,
        }
