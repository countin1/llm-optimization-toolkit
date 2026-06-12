"""
PromptForge CLI

用法：
    python -m promptforge search --method grid --max-questions 10
    python -m promptforge search --method bayesian --iterations 20
    python -m promptforge search --method genetic --generations 10
"""

import argparse
import json
import sys
import os

from .api.client import create_client_from_env
from .search.grid import GridSearch
from .search.bayesian import BayesianSearch
from .search.genetic import GeneticSearch
from .analysis.report import generate_report


def main():
    parser = argparse.ArgumentParser(description="PromptForge — Prompt 自动优化框架")
    subparsers = parser.add_subparsers(dest="command")

    # search 命令
    search_parser = subparsers.add_parser("search", help="搜索最优 prompt 模板")
    search_parser.add_argument("--method", choices=["grid", "bayesian", "genetic"],
                               default="grid", help="搜索方法")
    search_parser.add_argument("--data", type=str, default="data/questions.json",
                               help="评测数据路径")
    search_parser.add_argument("--max-questions", type=int, help="最大题目数")
    search_parser.add_argument("--iterations", type=int, default=20,
                               help="贝叶斯迭代次数")
    search_parser.add_argument("--generations", type=int, default=10,
                               help="遗传算法代数")
    search_parser.add_argument("--model", type=str, default="mimo-v2.5-pro",
                               help="模型名称")

    args = parser.parse_args()

    if args.command == "search":
        # 加载题目
        if not os.path.exists(args.data):
            print(f"错误: 数据文件不存在: {args.data}")
            sys.exit(1)

        with open(args.data, "r", encoding="utf-8") as f:
            questions = json.load(f)
        print(f"题目总数: {len(questions)}")

        # 创建客户端
        client = create_client_from_env(args.model)

        # 搜索
        if args.method == "grid":
            searcher = GridSearch()
            result = searcher.search(questions, client, max_questions=args.max_questions)
        elif args.method == "bayesian":
            searcher = BayesianSearch()
            result = searcher.search(questions, client,
                                     n_iterations=args.iterations,
                                     max_questions=args.max_questions)
        elif args.method == "genetic":
            searcher = GeneticSearch()
            result = searcher.search(questions, client,
                                     generations=args.generations,
                                     max_questions=args.max_questions)

        # 输出结果
        print(f"\n{'='*60}")
        print(result.summary())
        print(f"{'='*60}")

        # 生成报告
        if hasattr(result, "all_results"):
            report_path = generate_report(
                result.all_results, result.best_name, result.comparisons,
                args.model, "reports"
            )
            print(f"\n报告已生成: {report_path}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
