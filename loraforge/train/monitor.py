"""
训练监控模块

- 训练 loss 曲线
- Checkpoint 管理
- 训练日志
"""

import os
import json
import datetime
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


class TrainingMonitor:
    """训练监控器"""

    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = output_dir
        self.log_file = os.path.join(output_dir, "training_log.jsonl")
        self.history = {"train_loss": [], "eval_loss": [], "epoch": [], "step": []}
        os.makedirs(output_dir, exist_ok=True)

    def log_step(self, step: int, epoch: int, train_loss: float, eval_loss: float = None):
        """记录训练步骤"""
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "step": step,
            "epoch": epoch,
            "train_loss": train_loss,
            "eval_loss": eval_loss,
        }

        self.history["step"].append(step)
        self.history["epoch"].append(epoch)
        self.history["train_loss"].append(train_loss)
        if eval_loss is not None:
            self.history["eval_loss"].append(eval_loss)

        # 写入日志文件
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def plot_loss_curve(self, title: str = "训练 Loss 曲线") -> str:
        """
        绘制 loss 曲线

        Returns:
            图表文件路径
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(self.history["step"], self.history["train_loss"],
                "b-o", markersize=3, label="Train Loss")

        if self.history["eval_loss"]:
            # eval_loss 的 step 可能不同
            eval_steps = self.history["step"][:len(self.history["eval_loss"])]
            ax.plot(eval_steps, self.history["eval_loss"],
                    "r-s", markersize=3, label="Eval Loss")

        ax.set_xlabel("Step", fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(self.output_dir, "loss_curve.png")
        plt.savefig(path, dpi=150)
        plt.close()
        return path

    def get_summary(self) -> dict:
        """获取训练摘要"""
        if not self.history["train_loss"]:
            return {}

        return {
            "total_steps": len(self.history["train_loss"]),
            "initial_loss": self.history["train_loss"][0],
            "final_loss": self.history["train_loss"][-1],
            "best_loss": min(self.history["train_loss"]),
            "loss_reduction": self.history["train_loss"][0] - self.history["train_loss"][-1],
        }


def plot_comparison_bar(base_scores, ft_scores, dimensions, model_name, output_dir="outputs/reports"):
    """
    微调前后对比柱状图

    Args:
        base_scores: 基座模型得分
        ft_scores: 微调模型得分
        dimensions: 维度列表
        model_name: 模型名称
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)

    x = range(len(dimensions))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar([i - width/2 for i in x], base_scores, width, label="微调前", color="#667eea")
    bars2 = ax.bar([i + width/2 for i in x], ft_scores, width, label="微调后", color="#2ecc71")

    ax.set_xticks(x)
    ax.set_xticklabels(dimensions, rotation=20, ha="right")
    ax.set_ylabel("平均分", fontsize=12)
    ax.set_title(f"{model_name} — 微调前后对比", fontsize=14)
    ax.legend()

    for bar, val in zip(bars1, base_scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f"{val:.1f}", ha="center", fontsize=9)
    for bar, val in zip(bars2, ft_scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f"{val:.1f}", ha="center", fontsize=9)

    plt.tight_layout()
    path = os.path.join(output_dir, f"comparison_bar_{model_name.replace('/', '_')}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path
