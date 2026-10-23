# pdbal (task dcpx 2026-10-22): per-role / per-DP means of selected gauges from a metrics sampler tsv
# (cols: ts role dp metric value). Usage: pdbal.yihou.py <tsv> <t_start_epoch> <t_end_epoch>
import sys, collections as co
f, t0, t1 = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
want = {"num_running_reqs", "num_queue_reqs", "num_prefill_inflight_queue_reqs", "num_prefill_bootstrap_queue_reqs",
        "num_decode_transfer_queue_reqs", "num_decode_prealloc_queue_reqs", "token_usage", "gen_throughput"}
s = co.defaultdict(lambda: [0, 0.0])
for line in open(f):
    p = line.rstrip("\n").split("\t")
    if len(p) < 5 or p[3] not in want: continue
    try: ts = float(p[0]); v = float(p[4])
    except ValueError: continue
    if not (t0 <= ts <= t1): continue
    k = (p[1], p[2], p[3]); s[k][0] += 1; s[k][1] += v
for (role, dp, m), (n, tot) in sorted(s.items()):
    if n: print(f"{role:8s} dp={dp:2s} {m:36s} mean={tot/n:9.3f} n={n}")
