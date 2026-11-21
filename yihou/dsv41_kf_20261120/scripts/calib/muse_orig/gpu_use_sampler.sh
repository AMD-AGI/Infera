#!/bin/bash
# Host-side GPU utilisation sampler: rocm-smi --showuse --showmemuse every INTERVAL s until END (ISO UTC).
# Output TSV: utc  gpu  gpu_use_pct  mem_activity_pct.  Usage: gpu_use_sampler.sh <out.tsv> <end_iso> [interval=2]
set -u
OUT=$1; END=$(date -d "$2" +%s); IV=${3:-2}
echo -e "utc\tgpu\tgpu_use_pct\tmem_activity_pct" > "$OUT"
while [ "$(date +%s)" -lt "$END" ]; do
  t=$(date -u +%FT%TZ)
  timeout 20 rocm-smi --showuse --showmemuse --json 2>/dev/null | python3 -c '
import json,sys
t=sys.argv[1]; d=json.load(sys.stdin)
for k,v in sorted(d.items()):
    if not k.startswith("card"): continue
    g=v.get("GPU use (%)","NA"); m=v.get("GPU Memory Allocated (VRAM%)", v.get("GPU memory use (%)","NA"))
    print(f"{t}\t{k[4:]}\t{g}\t{m}")' "$t" >> "$OUT"
  sleep "$IV"
done
