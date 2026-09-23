#!/usr/bin/env python3
"""Compare two completed C80 runs with explicit labels and provenance."""
import argparse
import datetime
import json
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('baseline', type=Path)
p.add_argument('run', type=Path)
p.add_argument('--reference-label', default='reference')
p.add_argument('--candidate-label', default='candidate')
p.add_argument('--output-dir', type=Path, required=True)
a = p.parse_args()
old = json.loads((a.baseline/'c80/agentx_conc80.json').read_text())
new = json.loads((a.run/'c80/agentx_conc80.json').read_text())
assert old['conc'] == new['conc'] == 80
assert (a.run/'c80-completed.txt').exists()
out = a.output_dir
out.mkdir(parents=True, exist_ok=True)
def at(obj, path):
    for key in path.split('.'):
        obj = obj[key]
    return obj

fields = {
    '成功请求数': 'num_requests_successful',
    'Total tokens/s/GPU': 'request_metrics.throughput.per_gpu.total_tput_tps',
    'Output tokens/s/GPU': 'request_metrics.throughput.per_gpu.output_tput_tps',
    '输出吞吐 (token/s)': 'request_metrics.throughput.output.tokens_per_second',
    '输入吞吐 (token/s，含命中)': 'request_metrics.throughput.input.tokens_per_second',
    'TTFT mean (s)': 'request_metrics.latency.ttft.mean',
    'TTFT p50 (s)': 'request_metrics.latency.ttft.p50',
    'TTFT p90 (s)': 'request_metrics.latency.ttft.p90',
    'TTFT p95 (s)': 'request_metrics.latency.ttft.p95',
    'ITL mean (s)': 'request_metrics.latency.itl.mean',
    '实际平均输入 tokens': 'request_metrics.tokens.input.mean',
    '实际平均输出 tokens': 'request_metrics.tokens.output_actual.mean',
    'AIPerf GPU cache hit（聚合诊断值）': 'server_metrics.cache.gpu_cache_hit_rate',
    'AIPerf Host cache hit（聚合诊断值）': 'server_metrics.cache.cpu_cache_hit_rate',
}
rows = []
for label, path in fields.items():
    x, y = at(old, path), at(new, path)
    rows.append({'metric': label, 'reference': x, 'candidate': y,
                 'change_pct': (y/x-1)*100 if x else None})
for root, name in ((a.baseline, 'old'), (a.run, 'new')):
    summary = json.loads((root/'analysis/summary.json').read_text())
    if name == 'old': old_phase = summary['points']['80']['phases']['profiling']
    else: new_phase = summary['points']['80']['phases']['profiling']
# Request-cohort cache accounting is the evidence used for mechanism comparisons.
for label, numerator in [('请求级 device hit rate', 'cached_device'), ('请求级 host hit rate', 'cached_host'), ('请求级 miss rate', 'miss_tokens')]:
    x = old_phase['cache'][numerator] / old_phase['cache']['input_tokens']
    y = new_phase['cache'][numerator] / new_phase['cache']['input_tokens']
    rows.append({'metric': label, 'reference': x, 'candidate': y, 'change_pct': (y/x-1)*100 if x else None})
for key in ('prefill/queue_ms', 'prefill/forward_envelope_ms', 'decode/alloc_wait_ms'):
    for stat in ('mean', 'p50', 'p90', 'p99'):
        x, y = old_phase['stages_ms'][key][stat], new_phase['stages_ms'][key][stat]
        rows.append({'metric': f'{key} {stat}', 'reference': x, 'candidate': y,
                     'change_pct': (y/x-1)*100 if x else None})
# Matched context/miss/output bins reduce differences in the closed-loop workload.
strata = []
for key, x in old_phase['stratified_stages_ms'].items():
    y = new_phase['stratified_stages_ms'].get(key)
    if not y or min(x.get('n', 0), y.get('n', 0)) < 20:
        continue
    strata.append({'stratum': key, 'baseline': x, 'current': y})
report = {'generated_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'baseline': str(a.baseline), 'run': str(a.run), 'reference_label': a.reference_label, 'candidate_label': a.candidate_label, 'metrics': rows,
          'matched_strata': strata, 'baseline_coverage': old_phase['coverage'],
          'current_coverage': new_phase['coverage'],
          'baseline_accounting': old['request_accounting'],
          'current_accounting': new['request_accounting'],
          'limitations': ['Inspect node/configuration provenance; even same-node sequential runs can have workload and temporal drift.',
                         'Closed-loop trace progress and cache workload can differ; headline throughput alone does not establish a parameter benefit.',
                         'AIPerf aggregate server cache counters may be inconsistent/reset; use the request-cohort cache accounting for mechanism claims.',
                         'Forward envelopes include scheduling gaps and are not exclusive GPU time.']}
(out/'comparison.json').write_text(json.dumps(report, indent=2)+'\n')
lines = [f'# C80 {a.candidate_label} 与 {a.reference_label} 自动比较', '',
         '以下为已完成运行的自动汇总。请结合节点、配置与缓存初始化记录；闭环轨迹进度可能不同，单次对比不能独立建立稳定收益。', '',
         f'| 指标 | {a.reference_label} | {a.candidate_label} | 变化 |', '|---|---:|---:|---:|']
for r in rows:
    change = f"{r['change_pct']:+.2f}%" if r['change_pct'] is not None else 'n/a'
    lines.append(f"| {r['metric']} | {r['reference']:.5g} | {r['candidate']:.5g} | {change} |")
lines += ['', 'AIPerf 聚合缓存值存在 counter reset/口径不一致，只保留作诊断；缓存收益采用请求级 cohort 统计。',
          '详细匹配分层、请求关联覆盖率和错误计数见 `comparison.json`。',
          '需要结合 miss/context/output 分层服务时间、取消/错误、缓存命中与工作量解释结果。', '']
(out/'COMPARISON.zh-CN.md').write_text('\n'.join(lines))
print(out/'COMPARISON.zh-CN.md')
