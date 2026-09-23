#!/usr/bin/env python3
"""Compare completed C80 runs; preserve the limits of a historical comparison."""
import argparse
import datetime
import json
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('baseline', type=Path)
p.add_argument('run', type=Path)
a = p.parse_args()
old = json.loads((a.baseline/'c80/agentx_conc80.json').read_text())
new = json.loads((a.run/'c80/agentx_conc80.json').read_text())
assert old['conc'] == new['conc'] == 80
assert (a.run/'c80-completed.txt').exists()
out = a.run/'analysis'
out.mkdir(exist_ok=True)
def at(obj, path):
    for key in path.split('.'):
        obj = obj[key]
    return obj

fields = {
    '成功请求数': 'num_requests_successful',
    '输出吞吐 (token/s)': 'request_metrics.throughput.output.tokens_per_second',
    '输入吞吐 (token/s，含命中)': 'request_metrics.throughput.input.tokens_per_second',
    'TTFT mean (s)': 'request_metrics.latency.ttft.mean',
    'TTFT p50 (s)': 'request_metrics.latency.ttft.p50',
    'TTFT p90 (s)': 'request_metrics.latency.ttft.p90',
    'TTFT p95 (s)': 'request_metrics.latency.ttft.p95',
    'ITL mean (s)': 'request_metrics.latency.itl.mean',
    '实际平均输入 tokens': 'request_metrics.tokens.input.mean',
    '实际平均输出 tokens': 'request_metrics.tokens.output_actual.mean',
    'GPU cache hit rate': 'server_metrics.cache.gpu_cache_hit_rate',
    'Host cache hit rate': 'server_metrics.cache.cpu_cache_hit_rate',
}
rows = []
for label, path in fields.items():
    x, y = at(old, path), at(new, path)
    rows.append({'metric': label, 'baseline_4k': x, 'current_8k': y,
                 'change_pct': (y/x-1)*100 if x else None})
for root, name in ((a.baseline, 'old'), (a.run, 'new')):
    summary = json.loads((root/'analysis/summary.json').read_text())
    if name == 'old': old_phase = summary['points']['80']['phases']['profiling']
    else: new_phase = summary['points']['80']['phases']['profiling']
for key in ('prefill/queue_ms', 'prefill/forward_envelope_ms', 'decode/alloc_wait_ms'):
    for stat in ('mean', 'p50', 'p90', 'p99'):
        x, y = old_phase['stages_ms'][key][stat], new_phase['stages_ms'][key][stat]
        rows.append({'metric': f'{key} {stat}', 'baseline_4k': x, 'current_8k': y,
                     'change_pct': (y/x-1)*100 if x else None})
# Matched context/miss/output bins reduce differences in the closed-loop workload.
strata = []
for key, x in old_phase['stratified_stages_ms'].items():
    y = new_phase['stratified_stages_ms'].get(key)
    if not y or min(x.get('n', 0), y.get('n', 0)) < 20:
        continue
    strata.append({'stratum': key, 'baseline': x, 'current': y})
report = {'generated_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'baseline': str(a.baseline), 'run': str(a.run), 'metrics': rows,
          'matched_strata': strata, 'baseline_coverage': old_phase['coverage'],
          'current_coverage': new_phase['coverage'],
          'baseline_accounting': old['request_accounting'],
          'current_accounting': new['request_accounting'],
          'limitations': ['Prefill node changed; this is a historical comparison, not a same-node paired experiment.',
                         'Closed-loop trace progress and cache workload can differ; headline throughput alone does not establish a chunk-size benefit.',
                         'Forward envelopes include scheduling gaps and are not exclusive GPU time.']}
(out/'chunk8k-comparison.json').write_text(json.dumps(report, indent=2)+'\n')
lines = ['# C80 8K 与历史 4K 自动比较', '',
         '以下为已完成运行的自动汇总。Prefill 节点发生变化，且闭环轨迹进度可能不同；单次历史对比不能独立证明 chunk-size 收益。', '',
         '| 指标 | 历史 4K | 当前 8K | 变化 |', '|---|---:|---:|---:|']
for r in rows:
    change = f"{r['change_pct']:+.2f}%" if r['change_pct'] is not None else 'n/a'
    lines.append(f"| {r['metric']} | {r['baseline_4k']:.5g} | {r['current_8k']:.5g} | {change} |")
lines += ['', '详细匹配分层、请求关联覆盖率和错误计数见 `chunk8k-comparison.json`。',
          '需要结合 miss/context/output 分层服务时间、取消/错误、缓存命中与工作量判断是否应在同两节点回测 4K。', '']
(out/'CHUNK8K-COMPARISON.zh-CN.md').write_text('\n'.join(lines))
print(out/'CHUNK8K-COMPARISON.zh-CN.md')
