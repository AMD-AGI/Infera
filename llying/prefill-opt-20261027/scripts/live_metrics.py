"""Live mechanism check during a run: per-PP-rank eviction count and mean duration over the interval since the
previous call (state in STATE_FILE), from the prefill /metrics. Run on the prefill node.
  python3 live_metrics.py METRICS_URL STATE_FILE"""
import json, os, re, sys, time, urllib.request

url, state = sys.argv[1], sys.argv[2]
txt = urllib.request.urlopen(url, timeout=10).read().decode()
cur = {}
for name in ("eviction_duration_seconds_count", "eviction_duration_seconds_sum", "evicted_tokens_total",
             "hicache_backup_duration_seconds_sum"):
    for m in re.finditer(r"^sglang:%s\{[^}]*pp_rank=\"(\d)\"[^}]*\} ([0-9.e+]+)" % name, txt, re.M):
        cur.setdefault(m.group(1), {})[name] = float(m.group(2))
prev = json.load(open(state)) if os.path.exists(state) else {}
json.dump(cur, open(state, "w"))
out = []
for r in sorted(cur):
    c, p = cur[r], prev.get(r, {})
    dn = c.get("eviction_duration_seconds_count", 0) - p.get("eviction_duration_seconds_count", 0)
    ds = c.get("eviction_duration_seconds_sum", 0) - p.get("eviction_duration_seconds_sum", 0)
    db = c.get("hicache_backup_duration_seconds_sum", 0) - p.get("hicache_backup_duration_seconds_sum", 0)
    out.append(f"pp{r}: evict {dn:.0f} avg {1e3 * ds / dn:.0f}ms backup {1e3 * db / dn:.0f}ms" if dn else f"pp{r}: evict 0")
print(time.strftime("%H:%M:%S", time.gmtime()), "live:", " | ".join(out), flush=True)
