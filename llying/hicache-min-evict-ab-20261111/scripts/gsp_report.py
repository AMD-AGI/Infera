#!/usr/bin/env python3
"""Summarize e1ab gsp A/B runs (round 2 = scored): prefill /metrics deltas between r2 start and end, plus bench_serving
latency. Usage: gsp_report.py RUN_DIR... (logs/gsp/gsp-me<N>-<ts>). Prints one markdown table row per run."""
import json
import re
import sys
from pathlib import Path

LINE = re.compile(r'^(sglang:\w+)\{([^}]*)\} (\S+)$')


def load(p):
    out = {}
    for line in Path(p).read_text().splitlines():
        m = LINE.match(line)
        if not m:
            continue
        labels = dict(kv.split('=', 1) for kv in re.findall(r'\w+="[^"]*"', m.group(2)))
        labels = {k: v.strip('"') for k, v in labels.items()}
        key = (m.group(1), labels.get('mode', ''), labels.get('pp_rank', ''), labels.get('le', ''))
        out[key] = out.get(key, 0.0) + float(m.group(3))
    return out


def delta(a, b, name, mode='', pp='0', le=''):
    k = (name, mode, pp, le)
    return b.get(k, 0.0) - a.get(k, 0.0)


def per_pp(a, b, name):
    return [delta(a, b, name, pp=str(r)) for r in range(4)]


rows = []
for d in map(Path, sys.argv[1:]):
    a, b = load(d / 'prefill_r2_start.prom'), load(d / 'prefill_r2_end.prom')
    dur = float((d / 'r2_end').read_text()) - float((d / 'r2_start').read_text())
    r = json.loads((d / 'r2.jsonl').read_text().splitlines()[-1])
    me = re.search(r'-me(\d+)-', d.name).group(1)
    inp = delta(a, b, 'sglang:prefill_effective_tokens_total', 'input')
    dev = delta(a, b, 'sglang:prefill_effective_tokens_total', 'device_hit')
    host = delta(a, b, 'sglang:prefill_effective_tokens_total', 'host_hit')
    tot = inp + dev + host
    ev_n = per_pp(a, b, 'sglang:eviction_duration_seconds_count')
    ev_s = per_pp(a, b, 'sglang:eviction_duration_seconds_sum')
    rows.append(dict(
        run=d.name, me=me, dur=r['duration'], wall=dur, completed=r['completed'],
        computed=inp, dev=dev, host=host, tot=tot,
        evicted=delta(a, b, 'sglang:evicted_tokens_total'),
        ev_n=ev_n[0], ev_s=ev_s[0], ev_s_max=max(ev_s),
        backup=delta(a, b, 'sglang:hicache_backup_tokens_total'),
        dropped=delta(a, b, 'sglang:hicache_dropped_tokens_total'),
        lb=delta(a, b, 'sglang:load_back_tokens_total'),
        lb_n=delta(a, b, 'sglang:load_back_duration_seconds_count'),
        lb_s=delta(a, b, 'sglang:load_back_duration_seconds_sum'),
        kvt=delta(a, b, 'sglang:kv_transfer_latency_ms_sum') / max(delta(a, b, 'sglang:kv_transfer_latency_ms_count'), 1),
        ttft=r['mean_ttft_ms'], ttft50=r['median_ttft_ms'], ttft99=r['p99_ttft_ms'],
        tpot=r['mean_tpot_ms'], e2e=r['mean_e2e_latency_ms'], rps=r['request_throughput'],
    ))

print('| MIN_EVICT | 时长 s | 请求/s | 计算 token | 计算 tok/s | device 命中 | host 命中 | 命中率 dev/host | 驱逐次数 | 驱逐 token | 每次驱逐 token | 驱逐耗时 s (pp0 / 各 pp 最大) | 备份 token | 丢弃 token | load-back token (次数, 耗时 s) | KV 传输均值 ms (pp0) | TTFT 均值/中位/P99 ms | TPOT ms | E2E 均值 ms |')
print('|' + '---|' * 19)
for x in rows:
    per = x['evicted'] / x['ev_n'] if x['ev_n'] else 0
    print(f"| {x['me']} | {x['dur']:.1f} | {x['rps']:.3f} | {x['computed'] / 1e6:.3f}M | {x['computed'] / x['dur']:,.0f} | "
          f"{x['dev'] / 1e6:.3f}M | {x['host'] / 1e6:.3f}M | {x['dev'] / x['tot']:.1%} / {x['host'] / x['tot']:.1%} | "
          f"{x['ev_n']:.0f} | {x['evicted'] / 1e6:.3f}M | {per:,.0f} | {x['ev_s']:.3f} / {x['ev_s_max']:.3f} | "
          f"{x['backup'] / 1e6:.3f}M | {x['dropped']:.0f} | {x['lb'] / 1e6:.3f}M ({x['lb_n']:.0f}, {x['lb_s']:.3f}) | "
          f"{x['kvt']:,.0f} | {x['ttft']:,.0f} / {x['ttft50']:,.0f} / {x['ttft99']:,.0f} | {x['tpot']:.2f} | {x['e2e']:,.0f} |")
