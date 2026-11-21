#!/usr/bin/env python3
"""One JSON line for a point: InferenceX's aggregated AgentX result, the decode's memory headroom (free VRAM after graph
capture, KV capacity, peak VRAM over the run), fatal errors, and the ionic traffic per NIC during the benchmark.
Usage: summarize.py RUN_DIR STATUS LABEL CONFIG CONC MODE   (stdlib only)
"""
import json
import re
import sys
from pathlib import Path

run, status, label, config, conc, mode = Path(sys.argv[1]), *sys.argv[2:7]
out = {"label": label, "config": config, "conc": int(conc), "mode": mode, "status": status}


def find(tree, pattern):
    """Leaves whose dotted key path matches ``pattern``."""
    hits = {}

    def walk(node, path):
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, f"{path}.{k}" if path else k)
        elif isinstance(node, (int, float)) and re.search(pattern, path):
            hits[path] = node

    walk(tree, "")
    return hits


results = sorted((run / "result").glob(f"agentx_{label}*.json"))
if results:
    agg = json.loads(results[0].read_text())
    rm = agg.get("request_metrics", {})
    out["tok_s_gpu"] = rm.get("throughput", {}).get("per_gpu", {}).get("total_tput_tps")
    out["qps_mean"] = rm.get("qps", {}).get("mean")
    out["latency"] = {k: round(v, 3) for k, v in find(rm, r"(intvty|interactivity|ttft|itl|tpot).*\.(p50|p90|median)$").items()}
    out["requests"] = find(agg, r"request_accounting\.")
    out["kv"] = find(agg.get("server_metrics", {}), r"kv_cache\.")
    out["cache"] = find(agg.get("server_metrics", {}), r"cache\.(gpu|cpu)_cache_hit_rate$")

# SGLang's "avail mem" comes from hipMemGetInfo, which does not count GEM BOs, so with the GEM KV pools it overstates
# free VRAM by the KV size; the free VRAM left after graph capture is the last capture stage's figure minus that.
mem = {}
for log in sorted((run / "logs").glob("decode*.log")):
    if log.stem.endswith(".startup"):
        continue
    text = log.read_text(errors="replace")
    ranks = len(re.findall(r"Capture target verify CUDA graph end", text)) or 1
    kv_per_rank = sum(float(x) for x in re.findall(r"KV Cache is allocated\..*?KV size: ([0-9.]+) GB", text)) / ranks
    last = [float(x) for x in re.findall(r"Capture draft extend CUDA graph end\..*?avail mem=([0-9.]+) GB", text)]
    verify = [float(x) for x in re.findall(r"Capture target verify CUDA graph end\..*?mem usage=([0-9.]+) GB", text)]
    mem[log.stem] = {
        "kv_tokens_per_rank": max(((float(s), int(t)) for t, s in re.findall(
            r"KV Cache is allocated\..*?#tokens: (\d+), KV size: ([0-9.]+) GB", text)), default=(0, None))[1],
        "kv_gb_per_rank": round(kv_per_rank, 2),
        "verify_graph_gb": max(verify) if verify else None,
        "free_after_capture_gb_min": round(min(last) - kv_per_rank, 2) if last else None,
        "fatal": sorted(set(re.findall(r"(HSA_STATUS_ERROR\w*|Fatal Python error|Memory access fault)", text))),
    }
out["decode_memory"] = mem
headroom = {}
for f in run.glob("vram-*.txt"):
    lines = f.read_text().splitlines()
    totals = next((list(map(int, x.split()[1:])) for x in lines if x.startswith("total")), None)
    rows = [list(map(int, x.split()[1:])) for x in lines if x.strip() and not x.startswith("total")]
    if rows:
        peak = [max(col) for col in zip(*rows)]
        headroom[f.stem.removeprefix("vram-")] = {"peak_used_mib": peak, "min_free_mib": [t - p for t, p in zip(totals, peak)] if totals else None}
out["vram"] = headroom


def counters(path):
    vals = {}
    if path.exists():
        for line in path.read_text().splitlines():
            nic, key, value = line.split()
            vals[(nic, key)] = int(value)
    return vals


traffic = {}
for start in run.glob("ionic-bench-start-*.txt"):
    host = start.stem.removeprefix("ionic-bench-start-")
    a, b = counters(start), counters(run / f"ionic-after-{host}.txt")
    per = {}
    for (nic, key), v in b.items():
        if (nic, key) in a and key in ("tx_rdma_ucast_bytes", "tx_rdma_retx_pkts", "tx_rdma_ack_timeout"):
            d = v - a[(nic, key)]
            per.setdefault(nic, {})[key] = round(d / 1e12, 3) if key.endswith("bytes") else d
    traffic[host] = per
out["ionic_tx_tb_retx_acktimeout"] = traffic
notes = run / "notes.txt"
out["notes"] = notes.read_text().splitlines()[-8:] if notes.exists() else []
print(json.dumps(out))
