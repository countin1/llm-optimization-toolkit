"""
LoRAForge CLI

用法：
    python -m loraforge data --augment 3
    python -m loraforge train --model Qwen/Qwen2.5-7B --qlora
    python -m loraforge eval --base-model Qwen/Qwen2.5-7B --adapter-path outputs/adapters/lora_adapter
"""

import argparse
import json
import os
import sys

from .data.loader import DataLoader
from .data.augment import DataAugmentor
from .data.splitter import DataSplitter


def main():
    parser = argparse.ArgumentParser(description="LoRAForge — 大模型微调 Pipeline")
    subparsers = parser.add_subparsers(dest="command")

    # data 命令
    data_parser = subparsers.add_parser("data", help="准备训练数据")
    data_parser.add_argument("--input", type=str, default="data/questions.json", help="输入文件")
    data_parser.add_argument("--augment", type=int, default=0, help="增强变体数")
    data_parser.add_argument("--split", type=float, default=0.8, help="训练集比例")
    data_parser.add_argument("--format", choices=["chat", "alpaca"], default="chat")

    # train 命令
    train_parser = subparsers.add_parser("train", help="训练模型")
    train_parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B")
    train_parser.add_argument("--epochs", type=int, default=3)
    train_parser.add_argument("--batch-size", type=int, default=4)
    train_parser.add_argument("--lr", type=float, default=2e-4)
    train_parser.add_argument("--r", type=int, default=8)
    train_parser.add_argument("--qlora", action="store_true")

    # eval 命令
    eval_parser = subparsers.add_parser("eval", help="评测模型")
    eval_parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-7B")
    eval_parser.add_argument("--adapter-path", type=str, required=True)
    eval_parser.add_argument("--max-samples", type=int)

    args = parser.parse_args()

    if args.command == "data":
        loader = DataLoader(args.input)
        questions = loader.load()
        print(f"加载 {len(questions)} 道题目")

        data = loader.to_chat_format(questions) if args.format == "chat" else loader.to_alpaca_format(questions)

        if args.augment > 0:
            augmentor = DataAugmentor()
            data = augmentor.augment(data, args.augment)
            print(f"增强后: {len(data)} 条")

        splitter = DataSplitter()
        train_data, test_data = splitter.split(data, args.split)
        print(f"训练集: {len(train_data)} 条, 测试集: {len(test_data)} 条")

        os.makedirs("data", exist_ok=True)
        with open("data/train.json", "w", encoding="utf-8") as f:
            json.dump(train_data, f, indent=2, ensure_ascii=False)
        with open("data/test.json", "w", encoding="utf-8") as f:
            json.dump(test_data, f, indent=2, ensure_ascii=False)
        print("已保存: data/train.json, data/test.json")

    elif args.command == "train":
        if not os.path.exists("data/train.json"):
            print("错误: 请先运行 loraforge data")
            sys.exit(1)

        with open("data/train.json", "r", encoding="utf-8") as f:
            train_data = json.load(f)
        with open("data/test.json", "r", encoding="utf-8") as f:
            test_data = json.load(f)

        if args.qlora:
            from .train.qlora import QLoRATrainer
            trainer = QLoRATrainer(args.model, r=args.r)
        else:
            from .train.lora import LoRATrainer
            trainer = LoRATrainer(args.model, r=args.r)

        trainer.train(train_data, test_data, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)

    elif args.command == "eval":
        print("请运行: python scripts/evaluate.py")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
