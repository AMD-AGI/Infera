import json
rows=[]
print("# Task 2 — single-conc (conc=1) agentic latency, 10 reps/shape, warmed cache")
print("| shape | ISL | OSL | cache-hit | TTFT p50 (ms) | TTFT p90 (ms) | E2E p50 (ms) | E2E p90 (ms) | E2E mean (ms) | TPOT p50 (ms) |")
print("|---|---|---|---|---|---|---|---|---|---|")
for name,ISL,OSL in [("p50",74000,320),("p90",155000,3300),("p99",235000,17000)]:
    recs=[json.loads(l) for l in open("/tmp/mix_lat/lat_%s.jsonl"%name)]
    def pc(k,p):
        v=sorted(x[k] for x in recs); return v[min(len(v)-1,int(round((len(v)-1)*p)))]
    def mn(k): return sum(x[k] for x in recs)/len(recs)
    ch=[x["cached_tokens"] for x in recs if x.get("cached_tokens")]; pt=[x["prompt_tokens"] for x in recs if x.get("prompt_tokens")]
    hit=100*sum(ch)/sum(pt) if ch else 0
    print("| %s | %d | %d | %.1f%% | %.0f | %.0f | %.0f | %.0f | %.0f | %.2f |"%(
        name,ISL,OSL,hit,pc("ttft_ms",.5),pc("ttft_ms",.9),pc("e2e_ms",.5),pc("e2e_ms",.9),mn("e2e_ms"),pc("tpot_ms",.5)))
