"""
模型对比评测

微调前后对比：配对 t 检验 + Cohen's d + 按维度分析。
"""

import os
import datetime
import numpy as np
from scipy import stats as sp_stats
from typing import Dict, List
from .stats import paired_t_test, cohens_d, rule_score


class CompareResult:
    """对比结果"""

    def __init__(self, analysis: Dict, details: List[Dict]):
        self.analysis = analysis
        self.details = details
        self.base_mean = analysis["base_mean"]
        self.ft_mean = analysis["ft_mean"]
        self.improvement = analysis["improvement"]
        self.cohens_d = analysis["cohens_d"]
        self.p_value = analysis["p_value"]

    def summary(self) -> str:
        return (
            f"基座模型: {self.base_mean:.2f}\n"
            f"微调模型: {self.ft_mean:.2f}\n"
            f"提升: {self.improvement:+.2f}\n"
            f"Cohen's d: {self.cohens_d}\n"
            f"p 值: {self.p_value}\n"
            f"显著性: {'是' if self.p_value < 0.05 else '否'}"
        )


class ModelComparator:
    """模型对比器"""

    def compare(self, base_scores: np.ndarray, ft_scores: np.ndarray,
                details: List[Dict]) -> CompareResult:
        """
        对比两个模型

        Args:
            base_scores: 基座模型得分
            ft_scores: 微调模型得分
            details: 详细信息

        Returns:
            CompareResult
        """
        # 整体统计
        tt = paired_t_test(ft_scores, base_scores)
        cd = cohens_d(ft_scores, base_scores)

        # 按维度分析
        dim_analysis = {}
        for i, detail in enumerate(details):
            dim = detail["dimension"]
            if dim not in dim_analysis:
                dim_analysis[dim] = {"base": [], "ft": []}
            dim_analysis[dim]["base"].append(base_scores[i])
            dim_analysis[dim]["ft"].append(ft_scores[i])

        by_dimension = {}
        for dim, scores in dim_analysis.items():
            b = np.array(scores["base"])
            f = np.array(scores["ft"])
            dim_t, dim_p = sp_stats.ttest_rel(f, b) if len(b) > 1 else (0, 1)
            by_dimension[dim] = {
                "n": len(b),
                "base_mean": round(float(b.mean()), 2),
                "ft_mean": round(float(f.mean()), 2),
                "improvement": round(float(f.mean() - b.mean()), 2),
                "p_value": round(dim_p, 4),
            }

        analysis = {
            "base_mean": round(float(base_scores.mean()), 2),
            "base_std": round(float(base_scores.std()), 2),
            "ft_mean": round(float(ft_scores.mean()), 2),
            "ft_std": round(float(ft_scores.std()), 2),
            "improvement": round(float(ft_scores.mean() - base_scores.mean()), 2),
            "t_stat": tt["t_stat"],
            "p_value": tt["p_value"],
            "significant": tt["significant"],
            "cohens_d": cd["d"],
            "magnitude": cd["magnitude"],
            "by_dimension": by_dimension,
        }

        return CompareResult(analysis, details)

    def generate_report(self, result: CompareResult, model_name: str,
                        output_dir: str = "outputs/reports") -> str:
        """生成对比报告"""
        os.makedirs(output_dir, exist_ok=True)
        report_path = os.path.join(output_dir, f"finetune_comparison_{model_name.replace('/', '_')}.md")

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"# 微调前后对比报告 — {model_name}\n\n")
            f.write(f"**生成时间**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("## 一、整体对比\n\n")
            f.write("| 指标 | 微调前 | 微调后 | 变化 |\n")
            f.write("|------|--------|--------|------|\n")
            f.write(f"| 均分 | {result.base_mean} | {result.ft_mean} | {result.improvement:+.2f} |\n\n")

            f.write("## 二、统计检验\n\n")
            f.write(f"| 统计量 | 值 |\n")
            f.write(f"|--------|----|\n")
            f.write(f"| t 统计量 | {result.analysis['t_stat']} |\n")
            f.write(f"| p 值 | {result.p_value} |\n")
            f.write(f"| Cohen's d | {result.cohens_d} ({result.analysis['magnitude']}) |\n\n")

            f.write("## 三、各维度对比\n\n")
            f.write("| 维度 | N | 微调前 | 微调后 | 提升 | p 值 |\n")
            f.write("|------|---|--------|--------|------|------|\n")
            for dim, data in sorted(result.analysis["by_dimension"].items()):
                f.write(f"| {dim} | {data['n']} | {data['base_mean']} | {data['ft_mean']} | "
                        f"{data['improvement']:+.2f} | {data['p_value']} |\n")
            f.write("\n")

        return report_path
