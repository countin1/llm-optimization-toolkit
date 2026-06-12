"""
报告生成模块

生成 Markdown 格式的优化报告。
"""

import os
import datetime
from typing import Dict, List, Optional
import numpy as np


def generate_report(all_results: Dict, best_name: str, comparisons: List[Dict],
                    model_name: str, output_dir: str = "reports") -> str:
    """
    生成优化报告

    Args:
        all_results: 所有模板结果
        best_name: 最优模板名
        comparisons: 统计比较结果
        model_name: 模型名称
        output_dir: 输出目录

    Returns:
        报告文件路径
    """
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, f"prompt_optimization_{model_name}.md")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# Prompt 模板优化报告 — {model_name}\n\n")
        f.write(f"**生成时间**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # 最优模板
        best = all_results[best_name]
        f.write("## 一、最优模板\n\n")
        f.write(f"**模板**: `{best_name}`\n\n")
        f.write(f"**均分**: {best['mean']} ± {best['std']}\n\n")
        f.write("配置:\n")
        for k, v in best["config"].items():
            f.write(f"- {k}: {v}\n")
        f.write("\n")

        # 模板排名
        f.write("## 二、模板排名\n\n")
        f.write("| 排名 | 模板 | 均分 | 标准差 | vs baseline | 显著性 |\n")
        f.write("|------|------|------|--------|-------------|--------|\n")
        sorted_results = sorted(all_results.items(), key=lambda x: x[1]["mean"], reverse=True)
        baseline_mean = all_results.get("baseline", {}).get("mean", 0)
        for rank, (name, data) in enumerate(sorted_results, 1):
            diff = data["mean"] - baseline_mean
            comp = next((c for c in comparisons if c["template"] == name), None)
            sig = comp["significant"] if comp else ""
            diff_str = f"{diff:+.2f}" if name != "baseline" else "-"
            f.write(f"| {rank} | {name} | {data['mean']} | {data['std']} | {diff_str} | {sig} |\n")
        f.write("\n")

        # 统计检验
        if comparisons:
            f.write("## 三、统计检验（vs baseline）\n\n")
            f.write("| 模板 | 差值 | t 值 | p 值 | Cohen's d | 显著性 |\n")
            f.write("|------|------|------|------|-----------|--------|\n")
            for c in comparisons:
                f.write(f"| {c['template']} | {c['diff']:+.2f} | {c['t_stat']} | "
                        f"{c['p_value']} | {c['cohens_d']} | {c['significant']} |\n")
            f.write("\n")

        # 结论
        f.write("## 四、结论\n\n")
        f.write(f"1. **最优模板**: `{best_name}`（均分 {best['mean']}）\n")
        sig_improvements = [c for c in comparisons if c["p_value"] < 0.05 and c["diff"] > 0]
        if sig_improvements:
            f.write(f"2. **显著优于 baseline 的模板**: {', '.join(c['template'] for c in sig_improvements)}\n")
        else:
            f.write("2. 各模板间无统计显著差异\n")
        f.write("\n")

    return report_path
