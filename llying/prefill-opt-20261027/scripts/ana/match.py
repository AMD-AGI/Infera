"""Same request (conversation_id, turn_index) in two runs: per-request TTFT / decode / latency difference.
python3 match.py RUN_A RUN_B"""
import json, sys, statistics as st
def load(run):
    out = {}
    for line in open(f"{run}/agentx/aiperf_artifacts/profile_export.jsonl"):
        r = json.loads(line); m = r["metadata"]
        if m.get("benchmark_phase") != "profiling": continue
        g = lambda k: (r["metrics"].get(k) or {}).get("value", 0)
        out[(m["conversation_id"], m["turn_index"])] = (g("time_to_first_token") / 1e3, g("request_latency") / 1e3,
                                                        g("output_sequence_length"), g("input_sequence_length") - g("usage_prompt_cache_read_tokens"))
    return out
a, b = load(sys.argv[1]), load(sys.argv[2])
k = [x for x in a if x in b]
print(f"matched {len(k)} of {len(a)} / {len(b)}")
for i, name in ((0, "ttft"), (1, "latency")):
    print(f"  {name}: A mean {st.mean(a[x][i] for x in k):.2f}  B mean {st.mean(b[x][i] for x in k):.2f}")
da = [a[x][1] - a[x][0] for x in k]; db = [b[x][1] - b[x][0] for x in k]
print(f"  decode: A {st.mean(da):.2f}  B {st.mean(db):.2f}; per output token A {sum(da) / sum(a[x][2] for x in k) * 1e3:.1f} ms  B {sum(db) / sum(b[x][2] for x in k) * 1e3:.1f} ms")
print(f"  uncached tokens A {st.mean(a[x][3] for x in k):,.0f}  B {st.mean(b[x][3] for x in k):,.0f}")
