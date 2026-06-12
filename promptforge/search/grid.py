"""
网格搜索

遍历所有模板组合，找到最优配置。
"""

import time
from typing import Dict, List, Optional
import numpy as np
from ..core.templates import PromptTemplates
from ..core.builder import PromptBuilder
from ..core.scorer import Scorer, rule_score


class GridSearchResult:
    """网格搜索结果"""

    def __init__(self, all_results: Dict, best_name: str, comparisons: List[Dict]):
        self.all_results = all_results
        self.best_name = best_name
        self.best_config = all_results[best_name]["config"]
        self.best_score = all_results[best_name]["mean"]
        self.comparisons = comparisons

    def summary(self) -> str:
        """返回摘要"""
        lines = [
            f"最优模板: {self.best_name}",
            f"最优得分: {self.best_score:.2f}",
            f"搜索空间: {len(self.all_results)} 种组合",
            "",
            "Top 5:",
        ]
        sorted_results = sorted(self.all_results.items(), key=lambda x: x[1]["mean"], reverse=True)
        for i, (name, data) in enumerate(sorted_results[:5], 1):
            lines.append(f"  {i}. {name}: {data['mean']:.2f}")
        return "\n".join(lines)


class GridSearch:
    """网格搜索"""

    def __init__(self, templates: Optional[PromptTemplates] = None,
                 builder: Optional[PromptBuilder] = None,
                 scorer: Optional[Scorer] = None):
        self.templates = templates or PromptTemplates()
        self.builder = builder or PromptBuilder(self.templates)
        self.scorer = scorer or Scorer()

    def search(self, questions: List[Dict], model_fn, *,
               roles: Optional[List[str]] = None,
               formats: Optional[List[str]] = None,
               reasonings: Optional[List[str]] = None,
               fewshots: Optional[List[str]] = None,
               max_questions: Optional[int] = None) -> GridSearchResult:
        """
        网格搜索

        Args:
            questions: 评测题目列表
            model_fn: 模型调用函数 fn(prompt) -> answer
            roles/formats/reasonings/fewshots: 搜索空间
            max_questions: 最大题目数

        Returns:
            GridSearchResult
        """
        if max_questions:
            questions = questions[:max_questions]

        combos = self.templates.get_all_combinations(roles, formats, reasonings, fewshots)
        print(f"搜索空间: {len(combos)} 个模板 × {len(questions)} 题 = {len(combos) * len(questions)} 次调用")

        all_results = {}

        for t_idx, config in enumerate(combos):
            name = self.templates.config_to_name(config)
            print(f"\n  [{t_idx+1}/{len(combos)}] 模板: {name}")

            scores = []
            for q in questions:
                prompt = self.builder.build(q["question"], config)
                answer = model_fn(prompt)
                score = self.scorer.score(q["question"], answer, q.get("expected_hint", ""))
                scores.append(score)

            mean_score = float(np.mean(scores))
            std_score = float(np.std(scores))
            all_results[name] = {
                "scores": scores,
                "mean": round(mean_score, 2),
                "std": round(std_score, 2),
                "config": config,
            }
            print(f"    均分: {mean_score:.2f} ± {std_score:.2f}")

        # 统计比较
        from scipy import stats as sp_stats
        from scipy.stats import norm

        best_name = max(all_results, key=lambda k: all_results[k]["mean"])
        baseline_scores = np.array(all_results.get("baseline", list(all_results.values())[0])["scores"])

        comparisons = []
        for name, data in all_results.items():
            if name == "baseline":
                continue
            scores = np.array(data["scores"])
            t_stat, p_val = sp_stats.ttest_rel(scores, baseline_scores)
            d = (scores.mean() - baseline_scores.mean()) / np.sqrt((scores.var() + baseline_scores.var()) / 2) if (scores.var() + baseline_scores.var()) > 0 else 0
            comparisons.append({
                "template": name,
                "mean": round(float(scores.mean()), 2),
                "diff": round(float(scores.mean() - baseline_scores.mean()), 2),
                "t_stat": round(t_stat, 3),
                "p_value": round(p_val, 4),
                "cohens_d": round(d, 3),
                "significant": "***" if p_val < 0.01 else ("**" if p_val < 0.05 else ("*" if p_val < 0.1 else "")),
            })

        comparisons.sort(key=lambda x: x["mean"], reverse=True)
        return GridSearchResult(all_results, best_name, comparisons)
