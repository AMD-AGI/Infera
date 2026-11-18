"""Mooncake KV transfer stats per engine over the AgentX measurement window (metrics_sampler output).
   mc_xfer.py RUN_DIR"""
import datetime, json, re, sys
run = sys.argv[1].rstrip("/")
log = open(f"{run}/agentx-c104/benchmark.log", errors="replace").read()
m = re.search(r"(\d\d:\d\d:\d\d)\.\d+ NOTICE\s+Phase profiling \(profiling\) started.*?target: ([\d.]+)s", log)
keys = ["vllm:mooncake_xfer_time_seconds_sum", "vllm:mooncake_xfer_time_seconds_count",
        "vllm:mooncake_bytes_transferred_sum", "vllm:mooncake_bytes_transferred_count",
        "vllm:mooncake_num_failed_transfers_total", "vllm:mooncake_num_kv_expired_reqs_total"]
snaps = {}
for line in open(f"{run}/samples.jsonl"):
    d = json.loads(line)
    if not d.get("m"):
        continue
    tot = {}
    for r in d["m"].values():
        for k in keys:
            if k in r:
                tot[k] = tot.get(k, 0.0) + r[k]
    snaps.setdefault(d["url"][-5:], []).append((d["t"], tot))
day = datetime.datetime.fromtimestamp(min(t for s in snaps.values() for t, _ in s), datetime.UTC).date()
t0 = datetime.datetime.combine(day, datetime.time.fromisoformat(m.group(1)), datetime.UTC).timestamp()
t1 = t0 + float(m.group(2))
for u, s in sorted(snaps.items()):
    a = min(s, key=lambda x: abs(x[0] - t0))[1]
    b = min(s, key=lambda x: abs(x[0] - t1))[1]
    dd = {k: b.get(k, 0) - a.get(k, 0) for k in keys}
    n = dd[keys[1]]
    nb = dd[keys[3]]
    if not n and not nb:
        print(u, "no mooncake transfer samples"); continue
    t = dd[keys[0]] / n if n else 0
    gb = dd[keys[2]] / nb / 1e9 if nb else 0
    print(f"{u}: transfers {n:.0f}, mean xfer {t:.3f} s, mean {gb:.2f} GB per transfer, "
          f"{gb / t if t else 0:.1f} GB/s per transfer, total {dd[keys[2]] / 1e12:.2f} TB in {t1 - t0:.0f} s "
          f"({dd[keys[2]] / (t1 - t0) / 1e9:.1f} GB/s), failed {dd[keys[4]]:.0f}, expired {dd[keys[5]]:.0f}")
