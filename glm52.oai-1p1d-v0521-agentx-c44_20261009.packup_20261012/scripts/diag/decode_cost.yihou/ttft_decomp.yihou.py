# TTFT decomposition of an AgentX run from aiperf artifacts (offline, read-only).
# usage: ttft_decomp.yihou.py <run_dir/agentx-c40> <out_prefix> [bin_s=300]
# Server side (server_metrics_export.json, 1-s timeslices of sglang histograms/gauges, summed over DP ranks):
#   prefill: prefill_bootstrap (bootstrap q -> wait q), queue_time (wait q -> 1st forward), prefill_forward
#            (1st forward -> prefill done, all chunks), prefill_transfer_kv_cache (transfer q -> done)
#   decode:  decode_prepare (recv -> prealloc q), decode_bootstrap (prealloc q -> transfer q = bootstrap+alloc),
#            decode_transferred (transfer q -> wait q = KV transfer wait), decode_waiting (wait q -> 1st forward),
#            fake_output (1st forward -> prebuilt/first token)
#   (stage boundaries: sglang srt/observability/req_time_stats.py in the v0521 image)
# Client side (profile_export.jsonl): per-request TTFT/ISL/turn binned by request start.
import json, sys, math, collections

run, outp = sys.argv[1], sys.argv[2]
B = float(sys.argv[3]) if len(sys.argv) > 3 else 300.0
A = run + "/aiperf_artifacts/"
sm = json.load(open(A + "server_metrics_export.json"))
pr = sm["summary"]["phase_time_ranges"]["profiling"]; t0 = pr["start_ns"]; t1 = pr["end_ns"]
nb = int(math.ceil((t1 - t0) / 1e9 / B))
M = sm["metrics"]

def b_of(ns):
    i = int((ns - t0) / 1e9 // B); return i if 0 <= i < nb else None

hist = collections.defaultdict(lambda: [[0.0, 0.0, None] for _ in range(nb)])  # key -> bins of [count,sum,buckets]
def addh(key, ts):
    H = hist[key]
    for s in ts:
        i = b_of(s["start_ns"])
        if i is None: continue
        H[i][0] += s.get("count", 0); H[i][1] += s.get("sum", 0.0)
        bk = s.get("buckets")
        if bk:
            if H[i][2] is None: H[i][2] = collections.Counter()
            H[i][2].update(bk)
gau = collections.defaultdict(lambda: [[0.0, 0] for _ in range(nb)])
def addg(key, ts):
    G = gau[key]
    for s in ts:
        i = b_of(s["start_ns"])
        if i is None: continue
        G[i][0] += s.get("avg", 0.0); G[i][1] += 1

HN = {"sglang:per_stage_req_latency_seconds", "sglang:queue_time_seconds", "sglang:kv_transfer_latency_ms",
      "sglang:kv_transfer_bootstrap_ms", "sglang:kv_transfer_alloc_ms", "sglang:load_back_duration_seconds",
      "sglang:time_to_first_token_seconds", "sglang:e2e_request_latency_seconds",
      "sglang:uncached_prompt_tokens_histogram", "sglang:prompt_tokens_histogram"}
GN = {"sglang:num_queue_reqs", "sglang:num_prefill_bootstrap_queue_reqs", "sglang:num_prefill_inflight_queue_reqs",
      "sglang:num_running_reqs", "sglang:num_decode_prealloc_queue_reqs", "sglang:num_decode_transfer_queue_reqs",
      "sglang:token_usage", "sglang:pending_prealloc_token_usage", "sglang:hicache_host_used_tokens",
      "sglang:num_decode_host_receive_queue_reqs", "sglang:cache_hit_rate"}
for name, m in M.items():
    if name not in HN and name not in GN: continue
    for s in m.get("series", []):
        lb = s.get("labels", {}); eng = lb.get("engine_type", "router")
        key = (name.replace("sglang:", ""), eng, lb.get("stage", ""))
        ts = s.get("timeslices") or []
        if m["type"] == "histogram": addh(key, ts)
        elif name in GN:
            # sum of per-rank averages: average per-rank gauge then x ranks
            addg(key + (lb.get("dp_rank", ""),), ts)

def pct(bk, q):
    if not bk: return float("nan")
    items = sorted(((float(k) if k != "+Inf" else float("inf")), v) for k, v in bk.items())
    tot = items[-1][1]
    if tot <= 0: return float("nan")
    tgt = q * tot; prev_le, prev_c = 0.0, 0
    for le, c in items:
        if c >= tgt:
            if le == float("inf"): return prev_le
            return prev_le + (le - prev_le) * ((tgt - prev_c) / max(c - prev_c, 1e-9))
        prev_le, prev_c = le, c
    return items[-2][0]

with open(outp + ".server.tsv", "w") as f:
    f.write("metric\tengine\tstage\tbin_min\tcount\tmean\tp95\n")
    for key in sorted(hist):
        for i, (c, s, bk) in enumerate(hist[key]):
            f.write(f"{key[0]}\t{key[1]}\t{key[2]}\t{i*B/60:.0f}\t{c:.0f}\t{(s/c if c else float('nan')):.4f}\t{pct(bk,0.95):.4f}\n")
        c = sum(x[0] for x in hist[key]); s = sum(x[1] for x in hist[key])
        allbk = collections.Counter()
        for x in hist[key]:
            if x[2]: allbk.update(x[2])
        f.write(f"{key[0]}\t{key[1]}\t{key[2]}\tALL\t{c:.0f}\t{(s/c if c else float('nan')):.4f}\t{pct(allbk,0.95):.4f}\n")
gsum = collections.defaultdict(lambda: [0.0] * nb)
for key, G in gau.items():
    for i, (s, n) in enumerate(G):
        if n: gsum[key[:3]][i] += s / n
with open(outp + ".gauges.tsv", "w") as f:
    f.write("metric\tengine\tbin_min\tsum_over_ranks_avg\n")
    for key in sorted(gsum):
        for i, v in enumerate(gsum[key]):
            f.write(f"{key[0]}\t{key[1]}\t{i*B/60:.0f}\t{v:.3f}\n")

# client side
rows = []
for line in open(A + "profile_export.jsonl"):
    r = json.loads(line); md = r["metadata"]; mt = r.get("metrics", {})
    if md.get("benchmark_phase") != "profiling" or md.get("was_cancelled"): continue
    tt = mt.get("time_to_first_token", {}).get("value")
    if tt is None: continue
    rows.append((md["request_start_ns"], tt / 1e3, mt.get("input_sequence_length", {}).get("value", 0),
                 md.get("turn_index", -1), md.get("agent_depth", -1)))
def q(v, p):
    v = sorted(v); return v[min(len(v) - 1, int(p * len(v)))] if v else float("nan")
with open(outp + ".client.tsv", "w") as f:
    f.write("bin_min\tn\tttft_mean\tttft_p50\tttft_p95\tttft_max\tisl_mean\tturn0_frac\tsub_frac\tn_ttft_gt20\n")
    for i in list(range(nb)) + ["ALL"]:
        rs = [r for r in rows if i == "ALL" or b_of(r[0]) == i]
        if not rs: continue
        t = [r[1] for r in rs]
        f.write(f"{i if i=='ALL' else int(i*B/60)}\t{len(rs)}\t{sum(t)/len(t):.2f}\t{q(t,.5):.2f}\t{q(t,.95):.2f}\t{max(t):.1f}\t"
                f"{sum(r[2] for r in rs)/len(rs):.0f}\t{sum(r[3]==0 for r in rs)/len(rs):.3f}\t"
                f"{sum(r[4]>0 for r in rs)/len(rs):.3f}\t{sum(r[1]>20 for r in rs)}\n")
# slow-tail requests (TTFT > 20 s): start offset, ttft, isl, turn
with open(outp + ".tail.tsv", "w") as f:
    f.write("t_rel_s\tttft_s\tisl\tturn\tdepth\n")
    for r in sorted(rows):
        if r[1] > 20: f.write(f"{(r[0]-t0)/1e9:.1f}\t{r[1]:.2f}\t{r[2]}\t{r[3]}\t{r[4]}\n")
print("done", outp, "profiling", t0, t1, "bins", nb)
