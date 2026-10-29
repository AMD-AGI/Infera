"""How the prefill prefix-cache miss share (new-computed / all prompt tokens) of the AgentX replay depends on the total
unique KV capacity (device + host), to price cache policies that trade capacity for cheaper evictions.
Model: per conversation, the cached part of its last prompt is a prefix of its next prompt (hit = min(ISL, cached));
one LRU over conversations, and when over capacity the least recently used conversation is trimmed from its tail
(the radix tree evicts leaves, i.e. a context's newest tokens, first). Cross-conversation sharing is ignored.
Requests are taken from aiperf's profile_export.jsonl in start order (warmup + profiling, the same span the
sglang:prefill_effective_tokens_total counters cover). Measured shares of real runs are printed for calibration.
  python3 cache_capacity_sim.py PROFILE_EXPORT.jsonl [--prom LABEL=PREFILL_PROM ...] [--caps 10 15 20 ...] (M tokens)"""

import argparse
import collections
import json
import re

ap = argparse.ArgumentParser()
ap.add_argument("export")
ap.add_argument("--prom", nargs="*", default=[])
ap.add_argument("--caps", nargs="*", type=float, default=[8, 12, 16, 20, 23, 26, 29, 32.4, 36.7, 45, 54.1, 80, 1e6])
a = ap.parse_args()

for spec in a.prom:
    label, path = spec.split("=", 1)
    v = {}
    for line in open(path):
        m = re.match(r'sglang:prefill_effective_tokens_total\{.*mode="(\w+)".*pp_rank="0".*\} (\S+)', line)
        if m:
            v[m.group(1)] = float(m.group(2))
        m = re.match(r'sglang:hicache_host_total_tokens\{.*pp_rank="0".*\} (\S+)', line)
        if m:
            v["host"] = float(m.group(1))
        m = re.match(r'sglang:max_total_num_tokens\{.*pp_rank="0".*\} (\S+)', line)
        if m:
            v["dev"] = float(m.group(1))
    tot = v["input"] + v["device_hit"] + v["host_hit"]
    print(f"measured {label:14s}: host {v.get('host', 0) / 1e6:5.1f}M + device {v.get('dev', 0) / 1e6:4.1f}M; "
          f"new {v['input'] / tot:.2%}, device hit {v['device_hit'] / tot:.1%}, host hit {v['host_hit'] / tot:.1%}")

reqs = []
for line in open(a.export):
    r = json.loads(line)
    md, mt = r["metadata"], r["metrics"]
    isl = mt.get("input_sequence_length", {}).get("value")
    if isl is None or md.get("was_cancelled"):
        continue
    reqs.append((md["request_start_ns"], md["conversation_id"], int(isl), md.get("benchmark_phase")))
reqs.sort()
convs = len(set(r[1] for r in reqs))
grow = collections.defaultdict(int)
print(f"{len(reqs)} requests, {convs} conversations, {sum(r[2] for r in reqs) / 1e6:.0f}M prompt tokens")


def sim(cap):
    cache = collections.OrderedDict()
    total = 0
    new = allin = 0
    for _, c, isl, ph in reqs:
        hit = min(isl, cache.get(c, 0))
        new += isl - hit
        allin += isl
        total += isl - cache.get(c, 0)
        cache[c] = isl
        cache.move_to_end(c)
        while total > cap:
            k, v = next(iter(cache.items()))
            if k == c:
                break
            cut = min(v, total - cap)
            total -= cut
            if v - cut <= 0:
                del cache[k]
            else:
                cache[k] = v - cut
    return new / allin


for cap in a.caps:
    print(f"  capacity {cap:7.1f}M tokens: new-computed share {sim(cap * 1e6):.2%}")
