"""Where TTFT goes in a P4D4 run: engine-side latencies (prefill / decode vLLM histograms, deltas over the AgentX
measurement window from metrics_sampler output) next to the router's own view (AIPerf server-metrics export).
   ttft_breakdown.py RUN_DIR [PREFILL_URL_SUFFIX=27201] [DECODE_URL_SUFFIX=27301]
With several API servers per engine the vLLM histograms come from one of them, so engine numbers are a sample."""
import datetime
import json
import re
import sys

run = sys.argv[1].rstrip("/")
p_sfx = sys.argv[2] if len(sys.argv) > 2 else "27201"
d_sfx = sys.argv[3] if len(sys.argv) > 3 else "27301"
log = open(f"{run}/agentx-c104/benchmark.log", errors="replace").read()
m = re.search(r"(\d\d:\d\d:\d\d)\.\d+ NOTICE\s+Phase profiling \(profiling\) started.*?target: ([\d.]+)s", log)
if not m:
    sys.exit("no profiling phase in benchmark.log")
names = ["vllm:time_to_first_token_seconds", "vllm:request_queue_time_seconds", "vllm:request_prefill_time_seconds"]
snaps = {}
for line in open(f"{run}/samples.jsonl"):
    d = json.loads(line)
    if not d.get("m"):
        continue
    tot = {}
    for per_rank in d["m"].values():
        for n in names:
            for suf in ("_sum", "_count"):
                v = per_rank.get(n + suf)
                if v is not None:
                    tot[n + suf] = tot.get(n + suf, 0.0) + v
    snaps.setdefault(d["url"][-5:], []).append((d["t"], tot))
day = datetime.datetime.fromtimestamp(min(t for s in snaps.values() for t, _ in s), datetime.UTC).date()
t0 = datetime.datetime.combine(day, datetime.time.fromisoformat(m.group(1)), datetime.UTC).timestamp()
t1 = t0 + float(m.group(2))
print(f"measurement {m.group(1)} + {float(m.group(2)):.0f} s")
for label, sfx in (("prefill", p_sfx), ("decode", d_sfx)):
    s = snaps.get(sfx)
    if not s:
        continue
    a = min(s, key=lambda x: abs(x[0] - t0))[1]
    b = min(s, key=lambda x: abs(x[0] - t1))[1]
    cells = []
    for n in names:
        c = b.get(n + "_count", 0) - a.get(n + "_count", 0)
        v = b.get(n + "_sum", 0) - a.get(n + "_sum", 0)
        if c:
            cells.append(f"{n.split(':')[1].replace('_seconds', '')} {v / c:.2f} s")
    print(f"  {label} engine: " + ", ".join(cells))
try:
    sm = json.load(open(f"{run}/agentx-c104/aiperf_artifacts/server_metrics_export.json"))["metrics"]
except (OSError, KeyError, ValueError):
    sm = {}
for name in ("infera_time_to_first_token_seconds", "infera_pd_dispatch_duration_seconds", "infera_request_duration_seconds"):
    for series in sm.get(name, {}).get("series", []):
        st = series.get("stats", {})
        if st.get("count"):
            lab = ",".join(f"{k}={v}" for k, v in series.get("labels", {}).items() if k in ("leg", "outcome"))
            print(f"  router {name.replace('infera_', '').replace('_seconds', '')}{'[' + lab + ']' if lab else ''}: "
                  f"n {st['count']}, mean {st['avg']:.2f} s, p50 {st.get('p50_estimate', 0):.2f}, p90 {st.get('p90_estimate', 0):.2f}")
