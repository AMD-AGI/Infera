#!/usr/bin/env python3
"""Background sampler for bottleneck analysis: every INTERVAL seconds, append one JSON line with
(1) the latency histograms (sum/count) of every live engine, and (2) cumulative CPU seconds of
every process in every engine container (docker top), so any window can be broken down later
with sample_window.py (TTFT stages, per-process CPU %).

Usage: sampler.py OUT_JSONL [INTERVAL=60]   (run detached on the control node; env from
config.3p1d.sh: TOPOLOGY, CONTAINER_PREFIX, ENGINE_PORT_BASE, ENGINE_PORT_STRIDE, SSH_OPTS)
"""

import json
import os
import re
import shlex
import subprocess
import sys
import time
import urllib.request

HISTS = ("time_to_first_token_seconds", "e2e_request_latency_seconds", "queue_time_seconds",
         "inter_token_latency_seconds", "kv_transfer_alloc_ms", "kv_transfer_bootstrap_ms",
         "kv_transfer_latency_ms", "kv_transfer_total_mb")
LINE = re.compile(r"^sglang:(" + "|".join(HISTS) + r")_(sum|count)(\{[^}]*\})? ([0-9.eE+-]+)$")


def topology():
    here = os.path.dirname(os.path.abspath(__file__))
    out = subprocess.run([sys.executable, os.path.join(here, "bench-harness/tools/topology.py"), "rows",
                          os.environ["TOPOLOGY"]], capture_output=True, text=True, check=True).stdout
    base, stride = int(os.environ["ENGINE_PORT_BASE"]), int(os.environ.get("ENGINE_PORT_STRIDE") or 1)
    rows = []
    for line in out.splitlines():
        index, instance, role, node, ip = line.split("\t")
        rows.append({"instance": instance, "role": role, "node": node,
                     "url": f"http://{ip}:{base + int(index) * stride}",
                     "container": f"{os.environ['CONTAINER_PREFIX']}-{instance}"})
    return rows


def metrics(url):
    values = {}
    try:
        text = urllib.request.urlopen(url + "/metrics", timeout=10).read().decode()
    except Exception as exc:  # noqa: BLE001 - a down engine is recorded, not fatal
        return {"error": str(exc)[:200]}
    for line in text.splitlines():
        m = LINE.match(line)
        if m:
            key = f"{m.group(1)}_{m.group(2)}"
            values[key] = values.get(key, 0.0) + float(m.group(4))
    return values


def cpu(node, containers):
    ssh = ["ssh"] + shlex.split(os.environ.get("SSH_OPTS", "-o BatchMode=yes")) + ["-n", node]
    cmd = " ; ".join(f"echo '## {c}'; docker top {c} -eo pid,comm,time 2>/dev/null" for c in containers)
    try:
        out = subprocess.run(ssh + [cmd], capture_output=True, text=True, timeout=30).stdout
    except subprocess.TimeoutExpired:
        return {}
    result, current = {}, None
    for line in out.splitlines():
        if line.startswith("## "):
            current = line[3:].strip()
            result[current] = []
            continue
        f = line.split()
        if current and len(f) >= 3 and f[0].isdigit():
            t = f[-1]
            days = 0
            if "-" in t:
                d, t = t.split("-", 1)
                days = int(d)
            h, m, s = (int(x) for x in t.split(":"))
            result[current].append([int(f[0]), " ".join(f[1:-1]), days * 86400 + h * 3600 + m * 60 + s])
    return result


def main():
    out_path = sys.argv[1]
    interval = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    pending = []
    while True:
        t0 = time.time()
        rows = topology()
        rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(t0)), "engines": {}, "cpu": {}}
        for r in rows:
            rec["engines"][r["instance"]] = {"role": r["role"], **metrics(r["url"])}
        by_node = {}
        for r in rows:
            by_node.setdefault(r["node"], []).append(r["container"])
        for node, containers in by_node.items():
            rec["cpu"].update(cpu(node, containers))
        # The shared /home refuses writes while momentarily full (EDQUOT); keep the samples and retry
        # on the next round instead of dying (the sampler died that way on 2026-10-14 22:00).
        pending.append(json.dumps(rec, separators=(",", ":")) + "\n")
        try:
            with open(out_path, "a") as stream:
                stream.write("".join(pending))
            pending.clear()
        except OSError:
            del pending[:-1000]
        time.sleep(max(1, interval - (time.time() - t0)))


if __name__ == "__main__":
    main()
