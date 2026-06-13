"""
可视化图表模块

生成 Prompt 优化相关的图表：
- 模板对比柱状图
- 组件贡献分析图
- 贝叶斯优化收敛曲线
- 统计检验热力图
"""

import os
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


def plot_template_comparison(all_results: Dict, model_name: str, output_dir: str = "reports") -> str:
    """
    模板对比柱状图

    Args:
        all_results: 所有模板结果
        model_name: 模型名称
        output_dir: 输出目录

    Returns:
        图表文件路径
    """
    os.makedirs(output_dir, exist_ok=True)

    sorted_results = sorted(all_results.items(), key=lambda x: x[1]["mean"], reverse=True)
    names = [r[0][:20] for r in sorted_results]
    means = [r[1]["mean"] for r in sorted_results]
    stds = [r[1]["std"] for r in sorted_results]

    fig, ax = plt.subplots(figsize=(max(10, len(names) * 0.8), 6))
    colors = ["#2ecc71" if i == 0 else "#667eea" for i in range(len(names))]
    bars = ax.bar(range(len(names)), means, yerr=stds, color=colors, edgecolor="white", capsize=3)

    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{val:.1f}", ha="center", fontsize=9)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("平均分", fontsize=12)
    ax.set_title(f"{model_name} — Prompt 模板对比", fontsize=14)
    ax.set_ylim(0, 10)
    plt.tight_layout()

    path = os.path.join(output_dir, f"template_comparison_{model_name}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_component_analysis(component_results: Dict, model_name: str, output_dir: str = "reports") -> str:
    """
    组件贡献分析图

    Args:
        component_results: 组件分析结果
        model_name: 模型名称
        output_dir: 输出目录

    Returns:
        图表文件路径
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for idx, (comp_name, options) in enumerate(component_results.items()):
        ax = axes[idx // 2][idx % 2]
        opts = sorted(options.items(), key=lambda x: x[1]["mean"], reverse=True)
        labels = [o[0] for o in opts]
        vals = [o[1]["mean"] for o in opts]
        ax.bar(labels, vals, color=["#2ecc71" if i == 0 else "#667eea" for i in range(len(labels))])
        ax.set_title(comp_name, fontsize=12)
        ax.set_ylim(0, 10)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.1, f"{v:.1f}", ha="center", fontsize=9)

    plt.suptitle(f"{model_name} — 组件贡献分析", fontsize=14)
    plt.tight_layout()

    path = os.path.join(output_dir, f"component_analysis_{model_name}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_convergence(history: List[float], method: str, model_name: str, output_dir: str = "reports") -> str:
    """
    优化收敛曲线

    Args:
        history: 每轮最优得分历史
        method: 搜索方法名称
        model_name: 模型名称
        output_dir: 输出目录

    Returns:
        图表文件路径
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, len(history) + 1), history, "b-o", markersize=4, linewidth=2)
    ax.fill_between(range(1, len(history) + 1), history, alpha=0.1)
    ax.set_xlabel("迭代次数", fontsize=12)
    ax.set_ylabel("当前最优得分", fontsize=12)
    ax.set_title(f"{model_name} — {method} 收敛曲线", fontsize=14)
    ax.grid(True, alpha=0.3)

    # 标注起终点
    ax.annotate(f"{history[0]:.2f}", xy=(1, history[0]), fontsize=10, color="red")
    ax.annotate(f"{history[-1]:.2f}", xy=(len(history), history[-1]), fontsize=10, color="green")

    plt.tight_layout()
    path = os.path.join(output_dir, f"convergence_{method}_{model_name}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_dimension_heatmap(by_dimension: Dict, model_name: str, output_dir: str = "reports") -> str:
    """
    维度对比热力图

    Args:
        by_dimension: 按维度分析结果
        model_name: 模型名称
        output_dir: 输出目录

    Returns:
        图表文件路径
    """
    os.makedirs(output_dir, exist_ok=True)

    dims = sorted(by_dimension.keys())
    metrics = ["base_mean", "ft_mean", "improvement"]
    data = np.array([[by_dimension[d].get(m, 0) for m in metrics] for d in dims])

    fig, ax = plt.subplots(figsize=(8, max(4, len(dims) * 0.6)))
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto")

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(["微调前", "微调后", "提升"], fontsize=10)
    ax.set_yticks(range(len(dims)))
    ax.set_yticklabels(dims, fontsize=10)

    for i in range(len(dims)):
        for j in range(len(metrics)):
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=10)

    plt.colorbar(im)
    ax.set_title(f"{model_name} — 维度对比热力图", fontsize=14)
    plt.tight_layout()

    path = os.path.join(output_dir, f"dimension_heatmap_{model_name}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path

