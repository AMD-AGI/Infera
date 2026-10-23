#!/usr/bin/env bash
# Sample per-GPU VRAM used (GiB) every 15 s for $2 seconds into $1 (node-side, detached).
OUT=$1; DUR=${2:-5400}; end=$(( $(date +%s) + DUR ))
while [ "$(date +%s)" -lt "$end" ]; do
  rocm-smi --showmeminfo vram --json 2>/dev/null | python3 -c 'import sys,json,time;d=json.load(sys.stdin);print(time.strftime("%H:%M:%S",time.gmtime()), " ".join(str(round(int(v["VRAM Total Used Memory (B)"])/2**30,2)) for k,v in sorted(d.items())))' >> "$OUT"
  sleep 15
done
