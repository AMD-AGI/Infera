#!/usr/bin/env python3
"""Decode KV pressure over a run from samples.jsonl: per minute, KV cache usage and waiting requests per DP engine,
and preemptions (cumulative, all engines). Also the mooncake failed-transfer / expired counters.
  decode_pressure.py RUN_DIR [DECODE_PORT=27301] [EVERY_S=60]
"""
import json
import sys
import time

run = sys.argv[1]
port = sys.argv[2] if len(sys.argv) > 2 else "27301"
every = float(sys.argv[3]) if len(sys.argv) > 3 else 60.0


def val(m, name):
    return sum(v for k, v in m.items() if k == name or k.startswith(name + "{") or k.startswith(name + "["))


last = 0.0
preempt0 = None
rows = []
for line in open(f"{run}/samples.jsonl"):
    try:
        s = json.loads(line)
    except json.JSONDecodeError:
        continue
    if not s.get("url", "").endswith(port) or "m" not in s:
        continue
    engines = s["m"]
    pre = sum(val(m, "vllm:num_preemptions_total") for m in engines.values())
    if preempt0 is None:
        preempt0 = pre
    if s["t"] - last < every:
        continue
    last = s["t"]
    kv = [m.get("vllm:kv_cache_usage_perc", 0) * 100 for _, m in sorted(engines.items())]
    wait = [int(m.get("vllm:num_requests_waiting", 0)) for _, m in sorted(engines.items())]
    run_ = [int(m.get("vllm:num_requests_running", 0)) for _, m in sorted(engines.items())]
    fail = sum(val(m, "vllm:mooncake_num_failed_transfers_total") for m in engines.values())
    exp = sum(val(m, "vllm:mooncake_num_kv_expired_reqs_total") for m in engines.values())
    rows.append((s["t"], kv, run_, wait, pre - preempt0, fail, exp))

for t, kv, run_, wait, pre, fail, exp in rows:
    print(
        time.strftime("%H:%M:%S", time.gmtime(t)),
        "kv%", " ".join(f"{x:5.1f}" for x in kv),
        "| run", " ".join(f"{x:3d}" for x in run_),
        "| wait", " ".join(f"{x:3d}" for x in wait),
        f"| preempt {pre:6.0f} | mc_fail {fail:.0f} expired {exp:.0f}",
    )
