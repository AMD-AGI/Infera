"""Compare completed cohorts and sampled resources in the first 43 minutes."""
import collections
import datetime
import json
import math
import statistics
from pathlib import Path

OUT = Path(__file__).resolve().parent
RUNS = {
    'A0': Path('/perf_apps/liyingli/bench_agentx/p8d8-adaptive-31625-20260923/runs/a0-guard-decode'),
    'R2': Path('/perf_apps/liyingli/bench_agentx/r2-4k-31705-31706-20260924/runs/r2-on-4k-31705-31706'),
}


def rows(path):
    with path.open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def stats(values):
    v = sorted(x for x in values if math.isfinite(x))
    if not v:
        return {'n': 0}
    def q(p):
        i = (len(v)-1)*p
        lo = int(i)
        return v[lo] + (v[min(lo+1, len(v)-1)]-v[lo])*(i-lo)
    return dict(n=len(v), mean=statistics.fmean(v), p50=q(.5), p95=q(.95),
                p99=q(.99), max=v[-1])


def cv(v):
    mean = statistics.fmean(v)
    return statistics.pstdev(v)/mean if mean else 0


def cohort_summary(cohort):
    stages = collections.defaultdict(list)
    ranks = {role: {str(i): collections.Counter() for i in range(8)} for role in ('prefill', 'decode')}
    cache = collections.Counter()
    for row in cohort:
        for role in ranks:
            d = row.get(role)
            if not d:
                continue
            rank = ranks[role][str(d['dp_rank'])]
            rank.update(requests=1, input_tokens=d['input_tokens'], output_tokens=d['output_tokens'])
            for key, value in d['durations_ms'].items():
                if value is not None:
                    stages[role+'/'+key].append(value)
            if role == 'prefill':
                for key in ('input_tokens', 'cached_device', 'cached_host'):
                    cache[key] += d.get(key, 0)
    return dict(n=len(cohort), stages={k: stats(v) for k,v in stages.items()},
                stage_gt1s={k: sum(x>1000 for x in v) for k,v in stages.items()},
                stage_gt10s={k: sum(x>10000 for x in v) for k,v in stages.items()},
                ranks=ranks, rank_total_cv={role: {k: cv([r[k] for r in rr.values()])
                for k in ('requests','input_tokens','output_tokens')} for role,rr in ranks.items()}, cache=dict(cache))


all_cohorts = {}
result = {'window_seconds': 2580, 'cohort': 'requests starting and completing inside window; missing diagnostics excluded per stage', 'runs': {}}
for label, root in RUNS.items():
    summary = json.loads((root/'analysis/summary.json').read_text())
    start = summary['points']['80']['phases']['profiling']['start_ns']
    end = start + 2580*10**9
    def inside(ns):
        return start <= ns < end
    cohort = [r for r in rows(root/'analysis/joined-requests.jsonl') if r['phase']=='profiling'
              and inside(r['metadata'].get('request_start_ns', 0))
              and inside(r['metadata'].get('request_end_ns', 0))]
    all_cohorts[label] = cohort
    item = result['runs'][label] = cohort_summary(cohort)
    item.update(start_ns=start, cutoff_ns=end)
    ids = {r['rid'] for r in cohort}
    blocked = collections.defaultdict(set)
    for path in (root/'diagnostics/decode').glob('*.jsonl'):
        for row in rows(path):
            if row.get('rid') in ids and row.get('event')=='admission' and row['reason']!='admitted':
                blocked[row['reason']].add(row['rid'])
    item['blocked_requests'] = {k: sorted(v) for k,v in blocked.items()}
    samples = collections.defaultdict(list)
    per_rank = collections.defaultdict(list)
    counts = collections.Counter()
    for row in rows(root/'sampling/engine.jsonl'):
        if row.get('record_type') != 'sample':
            continue
        ns = int(datetime.datetime.fromisoformat(row['captured_at']).timestamp()*1e9)
        if not inside(ns):
            continue
        role = row['endpoint']
        if row.get('error'):
            counts[role+'/errors'] += 1
            continue
        counts[role+'/samples'] += 1
        metrics = collections.defaultdict(dict)
        for s in row.get('series', []):
            rank = s['labels'].get('dp_rank')
            if rank is not None:
                metrics[s['metric']][rank] = s['value']
        keys = ['token_usage','kv_used_tokens','num_running_reqs','num_queue_reqs',
                'num_decode_prealloc_queue_reqs','num_decode_transfer_queue_reqs',
                'pending_prealloc_token_usage']
        for key in keys:
            m = metrics.get('sglang:'+key, {})
            if len(m) != 8:
                continue
            v = list(m.values())
            prefix = role+'/'+key
            samples[prefix+'/mean_rank'].append(statistics.fmean(v))
            samples[prefix+'/max_rank'].append(max(v))
            samples[prefix+'/cv'].append(cv(v))
            samples[prefix+'/max_minus_min'].append(max(v)-min(v))
            for rank, value in m.items():
                per_rank[prefix+'/'+rank].append(value)
        usage = list(metrics.get('sglang:token_usage', {}).values())
        if len(usage)==8:
            counts[role+'/usage_samples'] += 1
            counts[role+'/any_rank_gt90pct'] += max(usage)>.9
            counts[role+'/hot_gt90_spare_lt50'] += max(usage)>.9 and min(usage)<.5
    item['engine_sample_counts'] = dict(counts)
    item['sampled_resources'] = {k: stats(v) for k,v in samples.items()}
    item['sampled_per_rank'] = {k: stats(v) for k,v in per_rank.items()}
    gpu = collections.defaultdict(list)
    for row in rows(root/'sampling/nodes.jsonl'):
        if row.get('record_type')!='sample' or row.get('error'):
            continue
        if not inside(int(datetime.datetime.fromisoformat(row['captured_at']).timestamp()*1e9)):
            continue
        for card, fields in row.get('sample',{}).get('gpu',{}).get('json',{}).items():
            for key in ('GPU use (%)','GPU Memory Read/Write Activity (%)'):
                try:
                    gpu[row['role']+'/'+key].append(float(fields[key]))
                except (KeyError, ValueError):
                    pass
    item['gpu'] = {k: stats(v) for k,v in gpu.items()}

# Match unique source turns, then restrict input/output sizes to avoid treating
# differences in trajectory progress as a scheduling effect.
def index(cohort):
    grouped = collections.defaultdict(list)
    for r in cohort:
        m = r['metadata']
        key = tuple(m.get(k) for k in ('source_trace_id','conversation_id','turn_index','source_outer_idx','source_inner_idx'))
        grouped[key].append(r)
    return {k: v[0] for k,v in grouped.items() if len(v)==1}

a, b = (index(all_cohorts[k]) for k in ('A0','R2'))
pairs = []
for key in a.keys() & b.keys():
    x,y = a[key],b[key]
    if not all(r.get('prefill') and r.get('decode') for r in (x,y)):
        continue
    ix,iy = (r['prefill']['input_tokens'] for r in (x,y))
    if abs(ix-iy)>8 or abs(ix-iy)>max(ix,iy)*.001:
        continue
    if x['decode']['output_tokens'] != y['decode']['output_tokens']:
        continue
    pairs.append((x,y))
result['matched'] = {k: cohort_summary([p[i] for p in pairs]) for i,k in enumerate(('A0','R2'))}
(OUT/'comparison.json').write_text(json.dumps(result, indent=2)+'\n')
print(OUT/'comparison.json')
