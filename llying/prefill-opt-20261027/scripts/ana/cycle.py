"""Lane cycle decomposition in the first W s of profiling: server latency vs recorded inter-turn delay.
python3 cycle.py RUN_DIR [W]"""
import json, sys, statistics as st
run, W = sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 1200
recs = []
for line in open(f"{run}/agentx/aiperf_artifacts/profile_export.jsonl"):
    r = json.loads(line); m = r["metadata"]
    if m.get("benchmark_phase") != "profiling": continue
    g = lambda k: (r["metrics"].get(k) or {}).get("value", 0)
    recs.append(dict(s=m["request_start_ns"] / 1e9, e=m["request_end_ns"] / 1e9, c=m["credit_issued_ns"] / 1e9,
                     ttft=g("time_to_first_token") / 1e3, conv=m["conversation_id"], turn=m["turn_index"],
                     depth=m.get("agent_depth", 0), root=m.get("root_correlation_id"), isl=g("input_sequence_length"),
                     osl=g("output_sequence_length")))
t0 = min(r["s"] for r in recs); t1 = t0 + W
by = {}
for r in recs: by.setdefault(r["conv"], []).append(r)
gaps, lats = [], []
for v in by.values():
    v.sort(key=lambda r: r["turn"])
    for a, b in zip(v, v[1:]):
        if t0 <= b["s"] < t1 and b["turn"] == a["turn"] + 1: gaps.append(b["s"] - a["e"])
win = [r for r in recs if t0 <= r["s"] < t1]
q = lambda xs, p: sorted(xs)[int(p * (len(xs) - 1))]
print(f"{run.split('/')[-1]}: in-window reqs {len(win)} (depth0 {sum(r['depth'] == 0 for r in win)}, subagent {sum(r['depth'] > 0 for r in win)})")
print(f"   same-conversation gap end->next start: n {len(gaps)} mean {st.mean(gaps):.2f} p50 {q(gaps, .5):.2f} p90 {q(gaps, .9):.2f} max {max(gaps):.0f} s; sum {sum(gaps):,.0f} s")
print(f"   latency mean {st.mean(r['e'] - r['s'] for r in win):.2f} s; credit->start mean {st.mean(r['s'] - r['c'] for r in win) * 1e3:.1f} ms")
# per-root tree activity: fraction of window with >=1 request of the tree in flight
roots = {}
for r in recs: roots.setdefault(r["root"], []).append((max(r["s"], t0), min(r["e"], t1)))
busy = 0
for iv in roots.values():
    iv = sorted(x for x in iv if x[1] > x[0]); cur = None
    for s, e in iv:
        if cur and s <= cur[1]: cur[1] = max(cur[1], e)
        else:
            if cur: busy += cur[1] - cur[0]
            cur = [s, e]
    if cur: busy += cur[1] - cur[0]
print(f"   trees seen {len(roots)}; mean trees with a request in flight {busy / W:.1f} of 104 lanes")
