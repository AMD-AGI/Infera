"""TTFT of the AgentX profiling phase, overall and by uncached prompt tokens (<2k, 2-8k, 8-32k, >=32k).
  python3 ttft_buckets.py LABEL=RUN_DIR/agentx/aiperf_artifacts/profile_export.jsonl ..."""
import json
import sys


def q(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(len(v) * p))]


for arg in sys.argv[1:]:
    label, path = arg.split("=", 1)
    b = {}
    for line in open(path):
        d = json.loads(line)
        m, md = d.get("metrics", {}), d.get("metadata", {})
        if md.get("benchmark_phase") != "profiling" or "time_to_first_token" not in m:
            continue
        new = m["input_sequence_length"]["value"] - m.get("usage_prompt_cache_read_tokens", {}).get("value", 0)
        k = "<2k" if new < 2048 else "2-8k" if new < 8192 else "8-32k" if new < 32768 else ">=32k"
        b.setdefault(k, []).append(m["time_to_first_token"]["value"] / 1e3)
    allv = [x for v in b.values() for x in v]
    print(f"{label:<12} all n={len(allv)} p50 {q(allv, .5):.2f} p90 {q(allv, .9):.2f} | " + " | ".join(
        f"{k} n={len(b[k])} p50 {q(b[k], .5):.2f} p90 {q(b[k], .9):.2f}"
        for k in ("<2k", "2-8k", "8-32k", ">=32k") if k in b))
