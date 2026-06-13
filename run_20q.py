import json, time, sys
import numpy as np
from scipy import stats

sys.stdout.reconfigure(encoding='utf-8')

from promptforge.api.client import create_client_from_env
from promptforge.core.builder import PromptBuilder
from promptforge.core.scorer import rule_score

with open('promptforge/data/questions.json', encoding='utf-8') as f:
    questions = json.load(f)[:20]

print(f'=== PromptForge: {len(questions)} 题 x 3 模板 ===')

client = create_client_from_env()
builder = PromptBuilder()

templates = {
    'baseline': {'role': 'none', 'format': 'none', 'reasoning': 'none', 'fewshot': 'none'},
    'cot':      {'role': 'none', 'format': 'none', 'reasoning': 'cot', 'fewshot': 'none'},
    'expert':   {'role': 'expert', 'format': 'structured', 'reasoning': 'cot', 'fewshot': 'none'},
}

all_results = {}

for tmpl_name, config in templates.items():
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
        sys.stdout.write(f'\r  [{tmpl_name}] {i+1}/{len(questions)}')
        sys.stdout.flush()

    elapsed = time.time() - start
    all_results[tmpl_name] = {'scores': scores, 'mean': float(np.mean(scores)), 'std': float(np.std(scores)), 'time': elapsed}
    print(f'\n  [{tmpl_name}] mean={np.mean(scores):.2f} std={np.std(scores):.2f} time={elapsed:.0f}s\n')

print('=== 统计分析 ===')
baseline = all_results['baseline']['scores']
for name in ['cot', 'expert']:
    other = all_results[name]['scores']
    t_stat, p_value = stats.ttest_rel(baseline, other)
    diff = np.array(other) - np.array(baseline)
    d = float(np.mean(diff) / np.std(diff)) if np.std(diff) > 0 else 0
    sig = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
    print(f'baseline vs {name}: t={t_stat:.3f}, p={p_value:.4f} {sig}, d={d:.3f}')

output = {'model': 'mimo-v2.5-pro', 'n': len(questions), 'results': {k: {'mean': v['mean'], 'std': v['std'], 'scores': v['scores']} for k, v in all_results.items()}}
with open('quick_test_results.json', 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=2)
print('\nDone! Results saved to quick_test_results.json')
