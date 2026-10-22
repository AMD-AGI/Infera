#!/usr/bin/env bash
# Task v0521cap2: read-only VRAM sampler (rocm-smi every 5 s) -> TSV; per-GPU used GiB. Usage: vram_peak.yihou.sh <out.tsv> <seconds>
out=$1; dur=$2; end=$(( $(date +%s) + dur ))
while [ "$(date +%s)" -lt "$end" ]; do
  rocm-smi --showmeminfo vram --json 2>/dev/null | python3 -c "import json,sys,time; d=json.load(sys.stdin); print(time.strftime('%H:%M:%S'), ' '.join(str(round(int(d[k]['VRAM Total Used Memory (B)'])/2**30,2)) for k in sorted(d) if k.startswith('card')))" >> "$out"
  sleep 5
done
