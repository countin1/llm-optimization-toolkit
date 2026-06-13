"""
快速实验：100 题 × 3 模板，验证 PromptForge 完整流程
"""
import json
import time
import sys
import numpy as np
from scipy import stats

sys.stdout.reconfigure(encoding='utf-8')

from promptforge.api.client import create_client_from_env
from promptforge.core.builder import PromptBuilder
from promptforge.core.scorer import rule_score


def main():
    # 加载题目
    with open('promptforge/data/questions.json', encoding='utf-8') as f:
        questions = json.load(f)

    print(f'=== PromptForge 完整实验 ===')
    print(f'题目数: {len(questions)}')
    print(f'模板数: 3 (baseline / cot / expert)')
    print(f'总调用: {len(questions) * 3} 次')
    print()

    # 创建客户端和构建器
    client = create_client_from_env()
    builder = PromptBuilder()

    # 3 种模板配置
    templates = {
        'baseline': {'role': 'none', 'format': 'none', 'reasoning': 'none', 'fewshot': 'none'},
        'cot':      {'role': 'none', 'format': 'none', 'reasoning': 'cot', 'fewshot': 'none'},
        'expert':   {'role': 'expert', 'format': 'structured', 'reasoning': 'cot', 'fewshot': 'none'},
    }

    all_results = {}

    for tmpl_name, config in templates.items():
        print(f'--- 模板: {tmpl_name} ---')
        scores = []
        start = time.time()

        for i, q in enumerate(questions):
            prompt = builder.build(q['question'], config)
            try:
                response = client(prompt)
                score = rule_score(response, q.get('expected_hint', ''))
            except Exception as e:
                print(f'  Q{q["id"]}: ERROR - {e}')
                score = 0

            scores.append(score)

            if (i + 1) % 20 == 0:
                elapsed = time.time() - start
                avg_so_far = np.mean(scores)
                print(f'  进度: {i+1}/{len(questions)}, 当前均分: {avg_so_far:.2f}, 耗时: {elapsed:.0f}s')

        elapsed = time.time() - start
        all_results[tmpl_name] = {
            'scores': scores,
            'mean': np.mean(scores),
            'std': np.std(scores),
            'median': np.median(scores),
            'min': np.min(scores),
            'max': np.max(scores),
            'time': elapsed,
        }
        print(f'  完成! 均分: {np.mean(scores):.2f}, 标准差: {np.std(scores):.2f}, 耗时: {elapsed:.0f}s')
        print()

    # ========== 统计分析 ==========
    print('=' * 50)
    print('=== 统计分析 ===')
    print('=' * 50)
    print()

    # 排序
    ranked = sorted(all_results.items(), key=lambda x: x[1]['mean'], reverse=True)
    print('排名:')
    for i, (name, r) in enumerate(ranked, 1):
        print(f'  {i}. {name}: {r["mean"]:.2f} ± {r["std"]:.2f}')

    print()

    # 配对 t 检验
    print('配对 t 检验:')
    baseline_scores = all_results['baseline']['scores']
    for name in ['cot', 'expert']:
        other_scores = all_results[name]['scores']
        t_stat, p_value = stats.ttest_rel(baseline_scores, other_scores)
        # Cohen's d
        diff = np.array(other_scores) - np.array(baseline_scores)
        cohens_d = np.mean(diff) / np.std(diff) if np.std(diff) > 0 else 0

        sig = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
        print(f'  baseline vs {name}:')
        print(f'    t = {t_stat:.3f}, p = {p_value:.4f} {sig}')
        print(f'    Cohen\'s d = {cohens_d:.3f} ({effect_size_label(cohens_d)})')
        print()

    # 保存结果
    output = {
        'model': 'mimo-v2.5-pro',
        'num_questions': len(questions),
        'results': {k: {**v, 'scores': v['scores']} for k, v in all_results.items()},
    }
    with open('quick_test_results.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print('结果已保存到 quick_test_results.json')


def effect_size_label(d):
    d = abs(d)
    if d < 0.2:
        return '可忽略'
    elif d < 0.5:
        return '小'
    elif d < 0.8:
        return '中等'
    else:
        return '大'


if __name__ == '__main__':
    main()
