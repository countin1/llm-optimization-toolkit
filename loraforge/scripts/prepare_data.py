"""
数据准备脚本

用法：
    python scripts/prepare_data.py --augment 3 --split 0.8
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loraforge.data.loader import DataLoader
from loraforge.data.augment import DataAugmentor
from loraforge.data.splitter import DataSplitter


def main():
    parser = argparse.ArgumentParser(description="准备微调数据")
    parser.add_argument("--data", type=str, default="data/questions.json", help="题目文件")
    parser.add_argument("--augment", type=int, default=0, help="每题增强变体数")
    parser.add_argument("--split", type=float, default=0.8, help="训练集比例")
    parser.add_argument("--format", choices=["chat", "alpaca"], default="chat", help="输出格式")
    args = parser.parse_args()

    # 加载
    loader = DataLoader(args.data)
    questions = loader.load()
    print(f"加载 {len(questions)} 道题目")

    # 转换格式
    if args.format == "chat":
        data = loader.to_chat_format(questions)
    else:
        data = loader.to_alpaca_format(questions)
    print(f"转换为 {args.format} 格式: {len(data)} 条")

    # 增强
    if args.augment > 0:
        augmentor = DataAugmentor()
        data = augmentor.augment(data, args.augment)
        print(f"增强后: {len(data)} 条")

    # 划分
    splitter = DataSplitter()
    train_data, test_data = splitter.split(data, args.split)
    print(f"训练集: {len(train_data)} 条, 测试集: {len(test_data)} 条")

    # 保存
    os.makedirs("data", exist_ok=True)
    with open("data/train.json", "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    with open("data/test.json", "w", encoding="utf-8") as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)

    print(f"\n已保存:")
    print(f"  data/train.json")
    print(f"  data/test.json")


if __name__ == "__main__":
    main()
