"""
训练脚本

用法：
    python scripts/train.py --model Qwen/Qwen2.5-7B --epochs 3
    python scripts/train.py --model Qwen/Qwen2.5-7B --epochs 3 --qlora
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loraforge.train.lora import LoRATrainer
from loraforge.train.qlora import QLoRATrainer


def main():
    parser = argparse.ArgumentParser(description="LoRA/QLoRA 微调")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B", help="基座模型")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮次")
    parser.add_argument("--batch-size", type=int, default=4, help="批大小")
    parser.add_argument("--lr", type=float, default=2e-4, help="学习率")
    parser.add_argument("--r", type=int, default=8, help="LoRA 秩")
    parser.add_argument("--qlora", action="store_true", help="使用 QLoRA (4bit)")
    args = parser.parse_args()

    # 加载数据
    if not os.path.exists("data/train.json"):
        print("错误: 训练数据不存在，请先运行 prepare_data.py")
        sys.exit(1)

    with open("data/train.json", "r", encoding="utf-8") as f:
        train_data = json.load(f)
    with open("data/test.json", "r", encoding="utf-8") as f:
        test_data = json.load(f)

    print(f"训练集: {len(train_data)} 条")
    print(f"测试集: {len(test_data)} 条")

    # 训练
    if args.qlora:
        trainer = QLoRATrainer(args.model, r=args.r)
    else:
        trainer = LoRATrainer(args.model, r=args.r)

    result = trainer.train(
        train_data, test_data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )

    print(f"\n下一步:")
    print(f"  python scripts/evaluate.py --base-model {args.model} --adapter-path {result['save_path']}")


if __name__ == "__main__":
    main()
