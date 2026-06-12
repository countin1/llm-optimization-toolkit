"""
评测脚本

用法：
    python scripts/evaluate.py --base-model Qwen/Qwen2.5-7B --adapter-path outputs/adapters/lora_adapter
"""

import argparse
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loraforge.eval.compare import ModelComparator
from loraforge.eval.stats import rule_score


def main():
    parser = argparse.ArgumentParser(description="微调前后对比评测")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-7B", help="基座模型")
    parser.add_argument("--adapter-path", type=str, help="LoRA adapter 路径")
    parser.add_argument("--max-samples", type=int, help="最大评测样本数")
    args = parser.parse_args()

    # 检查依赖
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
    except ImportError as e:
        print(f"缺少依赖: {e}")
        print("请运行: pip install torch transformers peft")
        return

    # 加载测试数据
    if not os.path.exists("data/test.json"):
        print("错误: 测试数据不存在，请先运行 prepare_data.py")
        return

    with open("data/test.json", "r", encoding="utf-8") as f:
        test_data = json.load(f)

    if args.max_samples:
        test_data = test_data[:args.max_samples]

    print(f"测试集: {len(test_data)} 条")

    # 加载 tokenizer
    print(f"\n加载 tokenizer: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    # 评估基座模型
    print(f"\n{'='*60}")
    print("评估基座模型...")
    print(f"{'='*60}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, trust_remote_code=True, torch_dtype=torch.float16, device_map="auto"
    )

    base_scores = []
    details = []
    for item in test_data:
        messages = item["messages"]
        question = messages[1]["content"]
        expected = messages[2]["content"]

        input_text = tokenizer.apply_chat_template(messages[:2], tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors="pt").to(base_model.device)

        with torch.no_grad():
            outputs = base_model.generate(**inputs, max_new_tokens=512, temperature=0.7, do_sample=True)

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        score = rule_score(response, expected)
        base_scores.append(score)
        details.append({"question": question[:80], "dimension": item["metadata"]["dimension"]})

    base_scores = np.array(base_scores)
    print(f"  基座模型均分: {base_scores.mean():.2f}")
    del base_model

    # 评估微调模型
    if args.adapter_path:
        print(f"\n{'='*60}")
        print("评估微调模型...")
        print(f"{'='*60}")
        ft_model = AutoModelForCausalLM.from_pretrained(
            args.base_model, trust_remote_code=True, torch_dtype=torch.float16, device_map="auto"
        )
        ft_model = PeftModel.from_pretrained(ft_model, args.adapter_path)
        ft_model.eval()

        ft_scores = []
        for item in test_data:
            messages = item["messages"]
            expected = messages[2]["content"]

            input_text = tokenizer.apply_chat_template(messages[:2], tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(input_text, return_tensors="pt").to(ft_model.device)

            with torch.no_grad():
                outputs = ft_model.generate(**inputs, max_new_tokens=512, temperature=0.7, do_sample=True)

            response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            score = rule_score(response, expected)
            ft_scores.append(score)

        ft_scores = np.array(ft_scores)
        print(f"  微调模型均分: {ft_scores.mean():.2f}")
        del ft_model
    else:
        ft_scores = base_scores

    # 对比
    comparator = ModelComparator()
    result = comparator.compare(base_scores, ft_scores, details)

    report_path = comparator.generate_report(result, args.base_model)

    print(f"\n{'='*60}")
    print(result.summary())
    print(f"{'='*60}")
    print(f"\n报告: {report_path}")


if __name__ == "__main__":
    main()
