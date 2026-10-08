"""Read archived logs/scrapes; never launch or modify inference services."""
import collections
import datetime as dt
import hashlib
import json
import re
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RUNTIME = Path('/perf_apps/liyingli/bench_agentx')
RUNS = {
    '历史 C80': 'p8d8-tracing-aus-20260922/runs/main-20260922',
    '历史 C112': 'p8d8-tracing-aus-20260922/runs/main-20260922',
    '4K复测': 'p8d8-baseline4k-31625-20260923/runs/baseline4k-c80-31625',
    '8K': 'p8d8-chunk8k-31625-20260923/runs/chunk8k-c80-31625',
    'A0': 'p8d8-adaptive-31625-20260923/runs/a0-guard-decode',
    'G0 / R1': 'p8d8-adaptive-31625-20260923/runs/g0-guard-completion',
    'C1': 'p8d8-adaptive-31625-20260923/runs/c1-fixed-host-31688',
    'C1G1': 'p8d8-adaptive-31625-20260923/runs/c1g1r-guard-completion-31688',
    'R1+R4': 'r1r4-4k-20260924/runs/r1r4-31719-performance-attempt4',
    'P affinity / Base': 'session-affinity-31999-20260928/runs/session-affinity-31999-performance',
    'B1': 'router-capacity-20260928/runs/campaign-b1-dynamo-p',
    'B2': 'router-capacity-20260928/runs/campaign-b2-radix',
    'B3*': 'router-capacity-20260928/runs/campaign-b3-decode-affinity',
    'RB': 'router-capacity-20260928/runs/campaign-rb-rebaseline',
    'B4': 'router-capacity-20260928/runs/campaign-b4-triton',
}


def stats(values):
    if not values:
        return {}
    xs = sorted(values)
    def q(p):
        i = (len(xs) - 1) * p
        j = int(i)
        return xs[j] + (xs[min(j + 1, len(xs) - 1)] - xs[j]) * (i - j)
    return dict(n=len(xs), mean=statistics.fmean(xs), p50=q(.5), p90=q(.9),
                p99=q(.99), max=xs[-1])


def timestamp(value):
    value = re.sub(r'(\.\d{6})\d+', r'\1', value)
    return dt.datetime.fromisoformat(value.replace('Z', '+00:00')).timestamp()


def read_rows(path, sources):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for line in f:
            digest.update(line)
            yield json.loads(line)
    sources[str(path)] = digest.hexdigest()


def recover_quantiles(run, sources):
    metrics = collections.defaultdict(list)
    count = 0
    input_tokens = output_tokens = 0
    starts, ends = [], []
    for row in read_rows(run/'c80/aiperf_artifacts/profile_export.jsonl', sources):
        if row['metadata']['benchmark_phase'] != 'profiling':
            continue
        assert not row.get('error')
        count += 1
        input_tokens += row['metrics']['input_sequence_length']['value']
        output_tokens += row['metrics']['output_token_count']['value']
        starts.append(row['metadata']['request_start_ns'])
        ends.append(row['metadata']['request_end_ns'])
        for key in ['time_to_first_token', 'inter_token_latency']:
            value = row['metrics'].get(key, {}).get('value')
            if value is not None and (key != 'inter_token_latency' or value > 0):
                metrics[key].append(value)
    elapsed = (max(ends)-min(starts))/1e9
    result = {'client_count': count, 'input_tps': input_tokens/elapsed,
              'total_tps_per_gpu': (input_tokens+output_tokens)/elapsed/16}
    for kind, key in [('ttft', 'time_to_first_token'), ('itl', 'inter_token_latency')]:
        s = stats(metrics[key])
        for p in ['p50', 'p90']:
            result[kind+'_'+p+('_s' if kind == 'ttft' else '_ms')] = s[p] / (1000 if kind == 'ttft' else 1)
            if kind == 'itl':
                result['intvty_'+p+'_tps'] = 1000 / s[p]
    return result


def extract(row):
    run = RUNTIME/RUNS[row['old_code']]
    # C1G1's compact source intentionally omits timestamps; use the archived full summary.
    summary_path = run/'analysis/summary.json'
    summary = json.loads(summary_path.read_text())
    point = summary['points'][str(row['concurrency'])]
    phase = point['phases']['profiling']
    if point.get('headline'):
        assert point['headline']['request_accounting']['records_profiled'] == row['metrics']['completed']
    start = phase['start_ns'] / 1e9
    end = start + 3600
    sources = {str(summary_path): hashlib.sha256(summary_path.read_bytes()).hexdigest()}
    values = collections.defaultdict(list)
    samples = collections.Counter()
    spans = collections.defaultdict(list)
    rank_ids = collections.defaultdict(set)
    previous = {}
    rate_tokens = rate_seconds = 0.0
    keys = ['kv_used_tokens', 'kv_evictable_tokens', 'max_total_num_tokens',
            'num_running_reqs', 'num_queue_reqs', 'num_decode_prealloc_queue_reqs',
            'num_decode_transfer_queue_reqs']
    for frame in read_rows(run/'sampling/engine.jsonl', sources):
        endpoint = frame.get('endpoint', '')
        role = 'prefill' if endpoint.startswith('prefill') else 'decode' if endpoint.startswith('decode') else None
        if role is None or frame.get('record_type') != 'sample':
            continue
        t = timestamp(frame['captured_at'])
        if not start <= t < end:
            continue
        samples[role+'_scrapes'] += 1
        if frame.get('error'):
            samples[role+'_errors'] += 1
            continue
        vectors = collections.defaultdict(dict)
        effective = {}
        for series in frame.get('series', []):
            labels = series.get('labels', {})
            rank = labels.get('dp_rank')
            if rank is None:
                continue
            name = series['metric'].removeprefix('sglang:')
            if name in keys:
                rank = int(rank)
                assert rank not in vectors[name], (name, rank)
                vectors[name][rank] = series['value']
            if role == 'prefill' and name == 'prefill_effective_tokens_total' and labels.get('mode') == 'input':
                effective[int(rank)] = series['value']
        if set(effective) == set(range(8)):
            total = sum(effective.values())
            old = previous.get(endpoint)
            if old and total >= old[1] and t > old[0]:
                rate_tokens += total - old[1]
                rate_seconds += t - old[0]
            previous[endpoint] = (t, total)
        if any(set(vectors[k]) != set(range(8)) for k in keys):
            samples[role+'_incomplete_vectors'] += 1
            continue
        samples[role+'_complete'] += 1
        spans[role].append(t)
        v = {k: [vectors[k][i] for i in range(8)] for k in keys}
        cap = sum(v['max_total_num_tokens'])
        assert cap > 0
        used = v['kv_used_tokens']; evict = v['kv_evictable_tokens']
        active = sum(used)/cap*100
        resident = sum(a+b for a,b in zip(used,evict))/cap*100
        assert 0 <= active <= resident <= 100.001
        fractions = [a/b*100 for a,b in zip(used,v['max_total_num_tokens'])]
        batch = v['num_running_reqs']; queue = v['num_queue_reqs']
        measurements = {
            'active_pct': active, 'resident_pct': resident,
            'evictable_pct': sum(evict)/cap*100, 'free_pct': 100-resident,
            'max_rank_active_pct': max(fractions),
            'kv_cv': statistics.pstdev(used)/statistics.fmean(used) if sum(used) else 0,
            'running_total': sum(batch), 'queue_total': sum(queue),
            'batch_cv': statistics.pstdev(batch)/statistics.fmean(batch) if sum(batch) else 0,
            'prealloc_total': sum(v['num_decode_prealloc_queue_reqs']),
            'transfer_total': sum(v['num_decode_transfer_queue_reqs']),
            'hot_with_spare_pct': 100*int(max(fractions)>90 and min(fractions)<50),
            'queue4_other0_pct': 100*int(max(queue)>=4 and min(queue)==0),
            'capacity_tokens': cap,
        }
        for k,value in measurements.items():
            values[role+'_'+k].append(value)
        values[role+'_running_rank'].extend(batch)
    if rate_seconds:
        values['prefill_effective_compute_tps'].append(rate_tokens/rate_seconds)

    # One final log per worker; never combine launch/ and final copies.
    for role in ['prefill', 'decode']:
        paths = sorted((run/'server-logs').glob(role+'*.log'))
        for path in paths:
            digest = hashlib.sha256()
            with path.open('rb') as f:
                for raw in f:
                    digest.update(raw)
                    line = raw.decode(errors='replace')
                    marker = 'Prefill batch' if role == 'prefill' else 'Decode batch'
                    if marker not in line:
                        continue
                    match = re.search(r'(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?Z)', line)
                    if not match:
                        samples[role+'_log_missing_timestamp'] += 1
                        continue
                    t = timestamp(match[1])
                    if not start <= t < end:
                        continue
                    rank = re.search(r'\bDP(\d+)\b', line)
                    if rank:
                        rank_ids[role].add(int(rank[1]))
                    if role == 'prefill':
                        for label,key in [('new-token','prefill_batch_new_tokens'),('new-seq','prefill_batch_sequences')]:
                            match = re.search('#'+label+r': (\d+)',line)
                            if match:
                                value = int(match[1]); values[key].append(value)
                                if label == 'new-token':
                                    limit = 8192 if row['old_code']=='8K' else 4096
                                    values['prefill_batch_full_pct'].append(100*int(value>=limit))
                    else:
                        match = re.search(r'#running-req: (\d+)',line)
                        if match:
                            values['decode_logged_batch_requests'].append(int(match[1]))
            sources[str(path)] = digest.hexdigest()
    extra = {}
    if row['old_code']=='B3*':
        extra = recover_quantiles(run,sources)
        assert extra['client_count']==row['metrics']['completed']
    return dict(run=str(run), window_start_utc=dt.datetime.fromtimestamp(start,dt.timezone.utc).isoformat(),
                window_seconds=3600, sources=sources, samples=dict(samples),
                sampled_spans={k:dict(first=min(v),last=max(v),span_s=max(v)-min(v)) for k,v in spans.items()},
                logged_ranks={k:sorted(v) for k,v in rank_ids.items()},
                stats={k:stats(v) for k,v in values.items()}, recovered_client_metrics=extra)


if __name__ == '__main__':
    rows = json.loads((HERE/'data.json').read_text())['rows']
    result = {}
    for row in rows:
        print('Reading',row['old_code'],flush=True)
        result[row['old_code']] = extract(row)
        (HERE/'dimensions.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    reference_sources = {}
    recovered = recover_quantiles(RUNTIME/RUNS['B2'], reference_sources)
    b2 = json.loads((REPO/'llying/router-capacity-campaign-20260928/b2/agentx_conc80.json').read_text())['request_metrics']['latency']
    for kind,unit,factor in [('ttft','s',1),('itl','ms',1000),('intvty','tps',1)]:
        for p in ['p50','p90']:
            expected = b2[kind][p]*factor
            tolerance = .0051 if kind=='itl' else .0000051
            assert abs(recovered[f'{kind}_{p}_{unit}']-expected)<=tolerance,(kind,p)
    throughput = json.loads((REPO/'llying/router-capacity-campaign-20260928/b2/agentx_conc80.json').read_text())['request_metrics']['throughput']
    assert abs(recovered['input_tps']-throughput['input']['tokens_per_second']) <= .0000051
    assert abs(recovered['total_tps_per_gpu']-throughput['per_gpu']['total_tput_tps']) <= .0000051
    (HERE/'quantile-recovery-validation.json').write_text(json.dumps(dict(reference='B2',sources=reference_sources,recovered=recovered,checks='p50/p90 TTFT, positive ITL, InferenceX intvty, input TPS and total TPS/GPU match official export'),indent=2)+'\n')
    print('All dimensions extracted; recovered B3 quantiles validated using B2 official export.',flush=True)
