"""Per-run split of where in-flight time goes, first W s of the profiling phase.
python3 lat_split.py RUN_DIR [W]"""
import json, sys, statistics as st
run, W = sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 1200
recs = []
for line in open(f"{run}/agentx/aiperf_artifacts/profile_export.jsonl"):
    r = json.loads(line); m = r["metadata"]
    if m.get("benchmark_phase") != "profiling": continue
    g = lambda k: (r["metrics"].get(k) or {}).get("value", 0)
    recs.append(dict(s=m["request_start_ns"] / 1e9, e=m["request_end_ns"] / 1e9, ttft=g("time_to_first_token") / 1e3,
                     lat=g("request_latency") / 1e3, isl=g("input_sequence_length"), osl=g("output_sequence_length"),
                     cr=g("usage_prompt_cache_read_tokens"), sess=m.get("session_num"), turn=m.get("turn_index")))
t0 = min(r["s"] for r in recs); t1 = t0 + W
win = [r for r in recs if t0 <= r["s"] < t1]
# in-flight seconds clipped to the window -> mean concurrency actually in flight
infl = sum(max(0, min(r["e"], t1) - max(r["s"], t0)) for r in recs) / W
ttft_inflight = sum(max(0, min(r["s"] + r["ttft"], t1) - max(r["s"], t0)) for r in recs) / W
n = len(win); tok = sum(r["isl"] + r["osl"] for r in win)
print(f"{run.split('/')[-1]}: reqs {n}  tok/s/GPU {tok / W / 8:,.0f}  mean in-flight {infl:.1f} of 104"
      f"  (in prefill/TTFT {ttft_inflight:.1f}, in decode {infl - ttft_inflight:.1f})")
q = lambda xs, p: sorted(xs)[int(p * (len(xs) - 1))]
for k in ("ttft", "lat"):
    xs = [r[k] for r in win]; print(f"   {k}: mean {st.mean(xs):.2f}  p50 {q(xs, .5):.2f}  p90 {q(xs, .9):.2f}  sum {sum(xs):,.0f} s")
dec = [r["lat"] - r["ttft"] for r in win]; osl = [r["osl"] for r in win]
print(f"   decode time: mean {st.mean(dec):.2f} s  sum {sum(dec):,.0f} s ; osl mean {st.mean(osl):.0f} ; isl mean {st.mean(r['isl'] for r in win):,.0f}"
      f" ; uncached mean {st.mean(r['isl'] - r['cr'] for r in win):,.0f}")
# idle gap between a lane's consecutive requests (same session): end -> next start
by = {}
for r in sorted(recs, key=lambda r: r["s"]): by.setdefault(r["sess"], []).append(r)
gaps = [b["s"] - a["e"] for v in by.values() for a, b in zip(v, v[1:]) if t0 <= b["s"] < t1]
if gaps: print(f"   gap between a session's requests: n {len(gaps)} mean {st.mean(gaps):.2f} p50 {q(gaps, .5):.2f} p90 {q(gaps, .9):.2f} s ; sessions {len(by)}")
