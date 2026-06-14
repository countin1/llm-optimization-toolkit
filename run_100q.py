"""
PromptForge 实验脚本 — 100 题 x N 模板

功能：
- 加载题目 → 构建 Prompt → 调用 API → 规则评分 → 统计分析
- 输出：均值、标准差、中位数、95% CI、配对 t 检验、Cohen's d、按维度分析
- 结果保存到 JSON 文件

用法：
    python run_100q.py                    # 默认 100 题
    python run_100q.py --max-questions 20 # 只跑 20 题
    python run_100q.py --output my.json   # 指定输出文件
"""
import json
import time
import sys
import argparse
from pathlib import Path
import numpy as np
from scipy import stats

sys.stdout.reconfigure(encoding='utf-8')

# 使用相对于脚本位置的路径，支持从任意目录运行
SCRIPT_DIR = Path(__file__).parent
QUESTIONS_PATH = SCRIPT_DIR / "promptforge" / "data" / "questions.json"

from promptforge.api.client import create_client_from_env
from promptforge.core.builder import PromptBuilder
from promptforge.core.scorer import rule_score


# ========== 配置 ==========

TEMPLATES = {
    'baseline': {'role': 'none', 'format': 'none', 'reasoning': 'none', 'fewshot': 'none'},
    'cot':      {'role': 'none', 'format': 'none', 'reasoning': 'cot', 'fewshot': 'none'},
    'expert':   {'role': 'expert', 'format': 'structured', 'reasoning': 'cot', 'fewshot': 'none'},
}

MAX_RETRIES = 2       # API 调用失败重试次数
RETRY_DELAY = 3       # 重试间隔（秒）
PROGRESS_INTERVAL = 10 # 每 N 题打印进度


# ========== 工具函数 ==========

def log(msg: str) -> None:
    """打印并立即刷新"""
    print(msg, flush=True)


def call_with_retry(client, prompt: str, retries: int = MAX_RETRIES) -> str:
    """
    带重试的 API 调用

    Args:
        client: 模型客户端
        prompt: 输入 prompt
        retries: 最大重试次数

    Returns:
        模型响应文本

    Raises:
        Exception: 重试耗尽后抛出最后一次异常
    """
    for attempt in range(retries + 1):
        try:
            return client(prompt)
        except Exception as e:
            if attempt < retries:
                log(f'    retry {attempt+1}/{retries}: {e}')
                time.sleep(RETRY_DELAY)
            else:
                raise


def compute_ci95(scores: list) -> float:
    """
    计算 95% 置信区间半宽

    Args:
        scores: 分数列表

    Returns:
        95% CI 的半宽值
    """
    n = len(scores)
    if n < 2:
        return 0.0
    se = np.std(scores, ddof=1) / np.sqrt(n)
    return float(stats.t.ppf(0.975, n - 1) * se)


def effect_size_label(d: float) -> str:
    """Cohen's d 效应量标签"""
    d = abs(d)
    if d < 0.2:
        return 'negligible'
    elif d < 0.5:
        return 'small'
    elif d < 0.8:
        return 'medium'
    else:
        return 'large'


# ========== 核心逻辑 ==========

def run_experiment(questions: list, client, builder: PromptBuilder) -> dict:
    """
    运行完整实验

    Args:
        questions: 题目列表
        client: 模型客户端
        builder: Prompt 构建器

    Returns:
        各模板的实验结果
    """
    all_results = {}

    for tmpl_name, config in TEMPLATES.items():
        scores = []
        errors = 0
        start = time.time()

        for i, q in enumerate(questions):
            prompt = builder.build(q['question'], config)
            try:
                response = call_with_retry(client, prompt)
                score = rule_score(response, q.get('expected_hint', ''))
            except Exception as e:
                log(f'  Q{q["id"]}: FAIL - {e}')
                score = 0
                errors += 1

            scores.append(score)

            if (i + 1) % PROGRESS_INTERVAL == 0:
                log(f'  [{tmpl_name}] {i+1}/{len(questions)} avg={np.mean(scores):.2f}')

        elapsed = time.time() - start
        all_results[tmpl_name] = {
            'scores': scores,
            'mean': float(np.mean(scores)),
            'std': float(np.std(scores)),
            'median': float(np.median(scores)),
            'ci95': compute_ci95(scores),
            'min': int(np.min(scores)),
            'max': int(np.max(scores)),
            'errors': errors,
            'time': elapsed,
        }
        log(f'  [{tmpl_name}] DONE mean={np.mean(scores):.2f} +/- {np.std(scores):.2f} '
            f'ci95=[{np.mean(scores)-compute_ci95(scores):.2f}, {np.mean(scores)+compute_ci95(scores):.2f}] '
            f'errors={errors} time={elapsed:.0f}s')

    return all_results


def statistical_analysis(all_results: dict) -> None:
    """
    统计分析：排名 + 配对 t 检验 + Cohen's d

    Args:
        all_results: 各模板实验结果
    """
    log('')
    log('=' * 60)
    log('Statistical Analysis')
    log('=' * 60)

    # 排名
    ranked = sorted(all_results.items(), key=lambda x: x[1]['mean'], reverse=True)
    for i, (name, r) in enumerate(ranked, 1):
        log(f'  {i}. {name}: {r["mean"]:.2f} +/- {r["std"]:.2f} '
            f'95%CI=[{r["mean"]-r["ci95"]:.2f}, {r["mean"]+r["ci95"]:.2f}]')

    log('')

    # 配对 t 检验
    baseline = all_results['baseline']['scores']
    for name in ['cot', 'expert']:
        other = all_results[name]['scores']
        t_stat, p_value = stats.ttest_rel(baseline, other)
        diff = np.array(other) - np.array(baseline)
        d = float(np.mean(diff) / np.std(diff)) if np.std(diff) > 0 else 0
        sig = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
        log(f'  baseline vs {name}: t={t_stat:.3f} p={p_value:.4f} {sig} d={d:.3f} ({effect_size_label(d)})')

    log('')


def dimension_analysis(questions: list, all_results: dict) -> None:
    """
    按维度分析各模板表现

    Args:
        questions: 题目列表
        all_results: 各模板实验结果
    """
    log('--- By Dimension ---')

    # 预构建维度 → 索引映射
    dims = {}
    for i, q in enumerate(questions):
        dim = q.get('dimension', 'unknown')
        dims.setdefault(dim, []).append(i)

    for dim, indices in dims.items():
        dim_scores = {}
        for tmpl_name, r in all_results.items():
            dim_scores[tmpl_name] = [r['scores'][i] for i in indices]
        best = max(dim_scores.items(), key=lambda x: np.mean(x[1]))
        log(f'  {dim}: best={best[0]}({np.mean(best[1]):.2f}), '
            + ', '.join(f'{k}={np.mean(v):.2f}' for k, v in dim_scores.items()))

    log('')


# ========== 主入口 ==========

def main():
    parser = argparse.ArgumentParser(description='PromptForge Experiment Runner')
    parser.add_argument('--max-questions', type=int, default=0,
                        help='最多跑多少题（0=全部）')
    parser.add_argument('--output', type=str, default='experiment_results.json',
                        help='输出 JSON 文件路径')
    args = parser.parse_args()

    # 加载题目
    with open(QUESTIONS_PATH, encoding='utf-8') as f:
        questions = json.load(f)
    if args.max_questions > 0:
        questions = questions[:args.max_questions]

    log(f'=== PromptForge Experiment: {len(questions)} questions x {len(TEMPLATES)} templates ===')
    log(f'API calls: {len(questions) * len(TEMPLATES)} (retries: {MAX_RETRIES})')
    log('')

    # 初始化
    client = create_client_from_env()
    builder = PromptBuilder()

    # 运行实验
    all_results = run_experiment(questions, client, builder)

    # 统计分析
    statistical_analysis(all_results)

    # 维度分析
    dimension_analysis(questions, all_results)

    # 保存结果
    output = {
        'model': 'mimo-v2.5-pro',
        'n': len(questions),
        'templates': list(TEMPLATES.keys()),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'results': {
            k: {
                'mean': v['mean'], 'std': v['std'], 'median': v['median'],
                'ci95': v['ci95'], 'min': v['min'], 'max': v['max'],
                'scores': v['scores'], 'errors': v['errors'], 'time': v['time'],
            } for k, v in all_results.items()
        },
    }
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    log(f'Done! Saved to {args.output}')


if __name__ == '__main__':
    main()
