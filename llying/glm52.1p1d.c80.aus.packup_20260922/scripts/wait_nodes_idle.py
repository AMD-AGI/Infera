#!/usr/bin/env python3
"""Wait for every GPU on both benchmark nodes to release memory before launch."""
import argparse
import concurrent.futures
import datetime
import json
import os
import shlex
import subprocess
import time

PROBE = '''
from pathlib import Path
import json
rows=[]
for d in sorted(Path("/sys/class/drm").glob("card[0-9]*/device")):
    used=d/"mem_info_vram_used"
    if not used.exists():
        continue
    total=int((d/"mem_info_vram_total").read_text())
    rows.append({"card":d.parent.name,
                 "vram_pct":round(100*int(used.read_text())/total,3),
                 "busy_pct":int((d/"gpu_busy_percent").read_text())})
print(json.dumps(rows))
'''

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("nodes", nargs="+")
parser.add_argument("--timeout", type=float, default=5400)
parser.add_argument("--interval", type=float, default=30)
parser.add_argument("--gpu-count", type=int, default=8)
parser.add_argument("--vram-pct", type=float, default=2)
parser.add_argument("--busy-pct", type=float, default=5)
args = parser.parse_args()
if not (0 < args.interval <= 60 and args.timeout > 0 and args.gpu_count > 0
        and 0 <= args.vram_pct <= 100 and 0 <= args.busy_pct <= 100):
    parser.error("invalid timeout, interval, GPU count or thresholds")
ssh_options = shlex.split(os.environ.get(
    "SSH_OPTS", "-F /dev/null -o BatchMode=yes -o ConnectTimeout=10 "
    "-o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/tmp/agentx_known_hosts"))

def probe(node):
    try:
        result = subprocess.run(["ssh", *ssh_options, node, "python3", "-"],
                                input=PROBE, text=True, capture_output=True,
                                timeout=25, check=True)
        rows = json.loads(result.stdout)
        ready = len(rows) == args.gpu_count and all(
            0 <= r["vram_pct"] <= args.vram_pct
            and 0 <= r["busy_pct"] <= args.busy_pct for r in rows)
        return node, {"ready": ready, "gpus": rows}
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        return node, {"ready": False, "error": str(exc)}

started = time.monotonic()
while True:
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(args.nodes)) as pool:
        states = dict(pool.map(probe, args.nodes))
    print(json.dumps({"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      "elapsed_s": round(time.monotonic()-started),
                      "nodes": states}), flush=True)
    if all(state["ready"] for state in states.values()):
        print("ALL_NODES_IDLE: every GPU passed memory and utilization thresholds", flush=True)
        break
    if time.monotonic()-started >= args.timeout:
        raise SystemExit("GPU release timeout: do not launch")
    time.sleep(args.interval)
