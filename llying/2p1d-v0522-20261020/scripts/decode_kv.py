"""Decode KV usage of one run over the AgentX window, 10 s bins: tokens in use, busiest-rank usage, prealloc waits,
and whether the busiest rank by running requests is also the busiest by KV.
  python3 decode_kv.py RUN_DIR WINDOW_START WINDOW_END TOKENS_PER_RANK"""
import re, sys, datetime, collections, statistics as st
run, t0, t1, cap = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
# DP-attention ranks log as "DP<r> TP<t>"; a DCP decode logs its one batch from TP 0 only (rank 0).
P = re.compile(r'^(\S+)Z \[\S+ \S+ (?:DP(\d+) TP\d+|TP0)\] Decode batch, #running-req: (\d+), #token: (\d+), token usage: ([0-9.]+).*?pre-allocated usage: ([0-9.]+), #prealloc-req: (\d+), #transfer-req: (\d+), #retracted-req: (\d+)')
b = collections.defaultdict(dict); nr = 0
for line in open(run + '/logs/glm52_2p1d-decode.log', errors='replace'):
    m = P.match(line)
    if not m or not (t0 <= m.group(1) <= t1): continue
    ts = datetime.datetime.fromisoformat(m.group(1)[:26]); k = ts.replace(second=ts.second // 10 * 10, microsecond=0)
    r = int(m.group(2) or 0); nr = max(nr, r + 1)
    b[k][r] = (int(m.group(3)), int(m.group(4)), float(m.group(5)), float(m.group(6)), int(m.group(7)), int(m.group(9)))
full = [v for v in b.values() if len(v) >= 3 * nr // 4]
tok = sorted(sum(x[1] for x in v.values()) for v in full)
umax = sorted(max(x[2] for x in v.values()) for v in full)
umean = [st.mean(x[2] for x in v.values()) for v in full]
upre = sorted(max(x[2] + x[3] for x in v.values()) for v in full)
same = sum(1 for v in full if max(v, key=lambda r: v[r][0]) == max(v, key=lambda r: v[r][1])) / len(full)
hi = sum(1 for u in umax if u > 0.9) / len(umax)
q = lambda a, p: a[min(len(a) - 1, int(len(a) * p))]
print('ranks %d, bins %d | decode KV tokens in use: mean %.2fM p90 %.2fM max %.2fM (pool %.1fM) | usage mean of ranks %.2f, busiest-rank usage p50 %.2f p90 %.2f, with prealloc p90 %.2f | bins with a rank > 0.9: %.0f%% | prealloc-req total max %d, retracted max %d | busiest-by-count == busiest-by-KV in %.0f%% of bins' % (
    nr, len(full), st.mean(tok) / 1e6, q(tok, .9) / 1e6, tok[-1] / 1e6, nr * cap / 1e6, st.mean(umean), q(umax, .5), q(umax, .9), q(upre, .9), 100 * hi,
    max(sum(x[4] for x in v.values()) for v in full), max(max(x[5] for x in v.values()) for v in full), 100 * same))
