#!/usr/bin/env python3
"""One line per AgentX result JSON: per-GPU throughput, interactivity, TTFT, ITL, cache, server token split."""
import json, sys
def g(d, *ks):
    for k in ks:
        if not isinstance(d, dict) or k not in d: return None
        d = d[k]
    return d
hdr = "file conc gpus P(nw x tp) D(nw x tp) ok/total tot/gpu out/gpu in/gpu intvty_p50 intvty_p90 ttft_p50 ttft_p90 ttft_mean itl_p50 itl_p90 in_mean out_mean dur hit_srv hit_theo computed_frac"
print(hdr)
for f in sys.argv[1:]:
    d = json.load(open(f))
    m = d.get("request_metrics", {})
    L = m.get("latency", {}); T = m.get("throughput", {})
    sm = d.get("server_metrics", {}) or {}
    pbs = g(sm, "tokens", "prompt_by_source") or {}
    pt = g(sm, "tokens", "prompt_total") or 0
    comp = pbs.get("computed")
    row = [f.split("/")[-1][:40], d.get("conc"), d.get("num_gpus"),
           f"{d.get('prefill_num_workers')}x{d.get('prefill_tp')}", f"{d.get('decode_num_workers')}x{d.get('decode_tp')}",
           f"{d.get('num_requests_successful')}/{d.get('num_requests_total')}",
           g(T, "per_gpu", "total_tput_tps"), g(T, "per_gpu", "output_tput_tps"), g(T, "per_gpu", "input_tput_tps"),
           g(L, "intvty", "p50"), g(L, "intvty", "p90"), g(L, "ttft", "p50"), g(L, "ttft", "p90"), g(L, "ttft", "mean"),
           g(L, "itl", "p50"), g(L, "itl", "p90"),
           g(m, "tokens", "input", "mean"), g(m, "tokens", "output_actual", "mean"), T.get("duration_seconds"),
           g(sm, "cache", "overall_cache_hit_rate"), g(m, "cache", "theoretical_cache_hit_rate"),
           (comp / pt) if (comp is not None and pt) else None]
    out = []
    for x in row:
        if isinstance(x, float): out.append(f"{x:.4g}")
        else: out.append(str(x))
    print(" ".join(out))
