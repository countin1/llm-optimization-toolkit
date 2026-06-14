"""
超参搜索

搜索最优的 LoRA 配置（r, alpha, lr, epochs）。

注意：当前 search() 返回模拟结果（随机 eval_loss），
实际使用时需要接入 LoRATrainer 执行真实训练。
"""

import os
import json
import datetime
import warnings
import numpy as np
from itertools import product
from typing import Dict, List, Optional


# 超参搜索空间
SEARCH_SPACE = {
    "r": [4, 8, 16, 32],
    "lora_alpha": [8, 16, 32],
    "learning_rate": [1e-4, 2e-4, 5e-4],
    "epochs": [2, 3, 5],
}

QUICK_SPACE = {
    "r": [8, 16],
    "lora_alpha": [16, 32],
    "learning_rate": [2e-4],
    "epochs": [3],
}


class HyperparamResult:
    """超参搜索结果"""

    def __init__(self, results: List[Dict]):
        self.results = sorted(results, key=lambda x: x["eval_loss"])
        self.best = self.results[0]

    def summary(self) -> str:
        lines = [
            f"最优配置: r={self.best['params']['r']}, alpha={self.best['params']['lora_alpha']}, "
            f"lr={self.best['params']['learning_rate']}, epochs={self.best['params']['epochs']}",
            f"最优 eval_loss: {self.best['eval_loss']:.4f}",
            f"总实验数: {len(self.results)}",
        ]
        return "\n".join(lines)


class HyperparamSearch:
    """超参搜索器"""

    def search(self, train_data: List[Dict], test_data: List[Dict],
               model_name: str = "Qwen/Qwen2.5-7B",
               quick: bool = False,
               output_dir: str = "outputs") -> HyperparamResult:
        """
        网格搜索超参

        Args:
            train_data: 训练数据
            test_data: 测试数据
            model_name: 基座模型
            quick: 快速模式
            output_dir: 输出目录

        Returns:
            HyperparamResult
        """
        search_space = QUICK_SPACE if quick else SEARCH_SPACE
        combinations = list(product(
            search_space["r"],
            search_space["lora_alpha"],
            search_space["learning_rate"],
            search_space["epochs"],
        ))

        print(f"搜索空间: {len(combinations)} 种组合")
        warnings.warn("HyperparamSearch.search() 使用模拟数据，未执行真实训练", UserWarning, stacklevel=2)

        results = []
        for i, (r, alpha, lr, epochs) in enumerate(combinations):
            params = {"r": r, "lora_alpha": alpha, "learning_rate": lr, "epochs": epochs}
            print(f"\n[{i+1}/{len(combinations)}] r={r}, alpha={alpha}, lr={lr}, epochs={epochs}")

            # 这里实际训练需要 GPU，暂时返回模拟结果
            # 实际使用时取消注释并调用 LoRATrainer
            result = {
                "params": params,
                "train_loss": np.random.uniform(0.5, 2.0),
                "eval_loss": np.random.uniform(0.6, 2.5),
                "output_dir": os.path.join(output_dir, f"hp_r{r}_a{alpha}"),
            }
            results.append(result)
            print(f"  eval_loss: {result['eval_loss']:.4f}")

        return HyperparamResult(results)

    def generate_report(self, result: HyperparamResult, model_name: str,
                        output_dir: str = "outputs/reports") -> str:
        """生成超参搜索报告"""
        os.makedirs(output_dir, exist_ok=True)
        report_path = os.path.join(output_dir, f"hyperparam_{model_name.replace('/', '_')}.md")

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"# LoRA 超参搜索报告 — {model_name}\n\n")
            f.write(f"**生成时间**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**总实验数**: {len(result.results)}\n\n")

            f.write("## 一、最优配置\n\n")
            best = result.best
            f.write(f"| 参数 | 值 |\n")
            f.write(f"|------|----|\n")
            for k, v in best["params"].items():
                f.write(f"| {k} | {v} |\n")
            f.write(f"| eval_loss | {best['eval_loss']:.4f} |\n\n")

            f.write("## 二、所有实验结果\n\n")
            f.write("| 排名 | r | alpha | lr | epochs | eval_loss |\n")
            f.write("|------|---|-------|-----|--------|----------|\n")
            for i, r in enumerate(result.results, 1):
                p = r["params"]
                f.write(f"| {i} | {p['r']} | {p['lora_alpha']} | {p['learning_rate']} | {p['epochs']} | {r['eval_loss']:.4f} |\n")
            f.write("\n")

        return report_path
