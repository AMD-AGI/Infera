# Read-only: summarize sglang histograms from aiperf server_metrics_export.csv (profiling window) per role (task v0521cap2, 2026-10-19).
# Count-weighted avg over DPs; p50/p90 = mean of per-DP estimates (approximation).
import sys, csv, collections as co, json
lab, fp = sys.argv[1], sys.argv[2]
want = {"kv_transfer_latency_ms", "kv_transfer_total_mb", "kv_transfer_speed_gb_s", "kv_transfer_bootstrap_ms", "kv_transfer_alloc_ms",
        "per_stage_req_latency_seconds", "queue_time_seconds", "dp_attention_sync_wait_seconds", "time_to_first_token_seconds"}
agg = co.defaultdict(lambda: [0, 0.0, [], []])
hdr = None
with open(fp, newline="") as f:
    for row in csv.reader(f):
        if not row or row[0].startswith("#"): continue
        if row[0] == "Endpoint": hdr = row; continue
        if hdr is None or "count" not in hdr or row[1] != "histogram": continue
        d = dict(zip(hdr, row)); m = d["Metric"].split(":", 1)[-1]
        if m not in want: continue
        role = d.get("engine_type") or ("prefill" if d["Endpoint"].endswith("29101") else "decode")
        key = (role, m, d.get("stage", ""))
        try: c = float(d["count"]); s = float(d["sum"])
        except ValueError: continue
        a = agg[key]; a[0] += c; a[1] += s
        for col, i in (("p50_estimate", 2), ("p90_estimate", 3)):
            try: a[i].append(float(d[col]))
            except (ValueError, KeyError): pass
out = {"label": lab}
for (role, m, st), (c, s, p50, p90) in sorted(agg.items()):
    if c == 0: continue
    out[f"{role}|{m}|{st}"] = {"n": int(c), "avg": round(s / c, 3), "p50": round(sum(p50)/len(p50), 3) if p50 else None, "p90": round(sum(p90)/len(p90), 3) if p90 else None}
print(json.dumps(out))
