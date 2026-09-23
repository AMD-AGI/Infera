#!/usr/bin/env python3
"""Create auditable matched-stratum summaries; not a causal estimator."""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('run', type=Path)
args = parser.parse_args()
summary = json.loads((args.run/'analysis/summary.json').read_text())
phases = {c: summary['points'][c]['phases']['profiling'] for c in ('80', '112')}
result = {'matched_strata': {}, 'ten_minute_arrival_cohorts': {},
          'method': 'Common input/miss/host bins for Prefill and input/output bins for Decode; '
                    'at least 20 samples in each point; weight=min(n80,n112). '
                    'This reduces coarse workload differences but does not establish causality.'}
for stage in phases['80']['stages_ms']:
    role, name = stage.split('/')
    pairs = []
    for key, a in phases['80']['stratified_stages_ms'].items():
        b = phases['112']['stratified_stages_ms'].get(key)
        if key.startswith(role+'/') and key.endswith('/'+name) and b and min(a['n'], b['n']) >= 20:
            pairs.append((min(a['n'], b['n']), a['mean'], b['mean']))
    weight = sum(x[0] for x in pairs)
    result['matched_strata'][stage] = dict(strata=len(pairs), weight=weight,
        means_ms={c: sum(x[0]*x[i] for x in pairs)/weight if weight else None
                  for c, i in [('80', 1), ('112', 2)]})
for c, phase in phases.items():
    groups = {}
    for start in range(0, 60, 10):
        group = {}
        for stage in phase['stages_ms']:
            values = [v for k, v in phase['minute_stage_ms'].items()
                      if start <= int(k.split('/')[0]) < start+10 and k.split('/', 1)[1] == stage]
            n = sum(v['n'] for v in values)
            group[stage] = dict(n=n, mean_ms=sum(v['n']*v['mean'] for v in values)/n if n else None)
        groups[str(start)] = group
    result['ten_minute_arrival_cohorts'][c] = groups
for c in ('80', '112'):
    result[c] = json.loads((args.run/f'c{c}/agentx_conc{c}.json').read_text())['request_metrics']
(args.run/'analysis/comparison.json').write_text(json.dumps(result, indent=2)+'\n')
print(args.run/'analysis/comparison.json')
