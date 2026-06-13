import json, time, sys
import numpy as np
from scipy import stats

sys.stdout.reconfigure(encoding='utf-8')

from promptforge.api.client import create_client_from_env
from promptforge.core.builder import PromptBuilder
from promptforge.core.scorer import rule_score

def log(msg):
    print(msg, flush=True)

with open('promptforge/data/questions.json', encoding='utf-8') as f:
    questions = json.load(f)

log(f'=== PromptForge: {len(questions)} x 3 ===')

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
            log(f'  Q{q["id"]}: ERR - {e}')
            score = 0
        scores.append(score)
        if (i + 1) % 10 == 0:
            log(f'  [{tmpl_name}] {i+1}/{len(questions)} avg={np.mean(scores):.2f}')

    elapsed = time.time() - start
    all_results[tmpl_name] = {
        'scores': scores, 'mean': float(np.mean(scores)),
        'std': float(np.std(scores)), 'median': float(np.median(scores)), 'time': elapsed,
    }
    log(f'  [{tmpl_name}] DONE mean={np.mean(scores):.2f} std={np.std(scores):.2f}')

log('')
log('=== Stats ===')

ranked = sorted(all_results.items(), key=lambda x: x[1]['mean'], reverse=True)
for i, (name, r) in enumerate(ranked, 1):
    log(f'  {i}. {name}: {r["mean"]:.2f} +/- {r["std"]:.2f}')

baseline = all_results['baseline']['scores']
for name in ['cot', 'expert']:
    other = all_results[name]['scores']
    t_stat, p_value = stats.ttest_rel(baseline, other)
    diff = np.array(other) - np.array(baseline)
    d = float(np.mean(diff) / np.std(diff)) if np.std(diff) > 0 else 0
    sig = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
    log(f'  base vs {name}: t={t_stat:.3f} p={p_value:.4f} {sig} d={d:.3f}')

log('')
log('--- By Dimension ---')
dims = {}
for q in questions:
    dim = q.get('dimension', 'unknown')
    dims.setdefault(dim, []).append(q['id'])

for dim, qids in dims.items():
    dim_scores = {}
    for t, r in all_results.items():
        dim_scores[t] = [r['scores'][i] for i in range(len(questions)) if questions[i]['id'] in qids]
    best = max(dim_scores.items(), key=lambda x: np.mean(x[1]))
    log(f'  {dim}: best={best[0]}({np.mean(best[1]):.2f}), ' + ', '.join(f'{k}={np.mean(v):.2f}' for k, v in dim_scores.items()))

output = {
    'model': 'mimo-v2.5-pro', 'n': len(questions),
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'results': {k: {'mean': v['mean'], 'std': v['std'], 'median': v['median'], 'scores': v['scores'], 'time': v['time']} for k, v in all_results.items()},
}
with open('experiment_results_100q.json', 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=2)
log('Done!')
