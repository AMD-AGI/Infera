"""AgentX cc-traces-weka-062126 statistics (whole traces, not the harness replay window).
Usage: python3 -I trace_stats.py <traces.jsonl> <out.json>
ISL/OSL in the trace's own units ('in' = 64-token hash blocks x 64, Claude tokenizer).
Theoretical prefix hit per request = leading hash_ids already seen earlier in the same trace
(hash ids are chained prefix-block ids, scope local to a trace)."""
import json, sys, statistics as st

def pct(a, q):
    a = sorted(a); return a[min(len(a) - 1, int(q * len(a)))] if a else None

src, out = sys.argv[1], sys.argv[2]
isl, osl, new, hitfrac, turns, sub_isl, main_isl = [], [], [], [], [], [], []
tot_in = tot_hit = 0
traces = 0
for line in open(src):
    d = json.loads(line); traces += 1
    seen = set(); n_main = 0
    reqs = []
    for r in d["requests"]:
        if r.get("type") == "subagent":
            for s in r.get("requests", []):
                if "in" in s: reqs.append((s, True))
        elif "in" in r:
            reqs.append((r, False)); n_main += 1
    reqs.sort(key=lambda x: x[0]["t"])  # sub-agent inner 't' assumed on the same clock
    for r, is_sub in reqs:
        h = r.get("hash_ids", [])
        k = 0
        while k < len(h) and h[k] in seen: k += 1
        seen.update(h)
        i = r["in"]; hit = min(i, k * 64)
        isl.append(i); osl.append(r.get("out", 0)); new.append(i - hit)
        hitfrac.append(hit / i if i else 0.0)
        (sub_isl if is_sub else main_isl).append(i)
        tot_in += i; tot_hit += hit
    turns.append(n_main)

def summ(a):
    return dict(n=len(a), mean=round(st.mean(a), 1), p10=pct(a, .1), p50=pct(a, .5), p90=pct(a, .9),
                p99=pct(a, .99), max=max(a))
res = dict(traces=traces, requests=len(isl), isl=summ(isl), main_isl=summ(main_isl), sub_isl=summ(sub_isl),
           osl=summ(osl), new_tokens=summ(new), main_turns_per_trace=summ(turns),
           token_weighted_prefix_hit=round(tot_hit / tot_in, 4),
           request_mean_prefix_hit=round(st.mean(hitfrac), 4),
           frac_isl_gt_128k=round(sum(x > 131072 for x in isl) / len(isl), 4),
           frac_isl_gt_256k=round(sum(x > 262144 for x in isl) / len(isl), 4),
           frac_isl_gt_512k=round(sum(x > 524288 for x in isl) / len(isl), 4),
           isl_weighted_mean=round(sum(x * x for x in isl) / sum(isl), 1),
           input_share=round(sum(isl) / (sum(isl) + sum(osl)), 5))
json.dump(res, open(out, "w"), indent=1); print(json.dumps(res, indent=1))
