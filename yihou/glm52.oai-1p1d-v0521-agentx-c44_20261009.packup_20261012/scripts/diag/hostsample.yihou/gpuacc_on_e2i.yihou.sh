#!/usr/bin/env bash
# Start gpuacc_sampler at each e2i "profiling started" chain-log line (R1, then R2); 3600 s at 10 s.
set -uo pipefail
D=/shared_nfs/yihou_glm52/v0521dbg_20261010/diag/hostsample.yihou; CHAIN=/shared_nfs/yihou_glm52/v0521dbg_20261010/runs/e2i.chain.log
for tag in e2iR1 e2iR2; do
  for i in $(seq 1 1440); do  # up to 4 h per tag
    l=$(grep -a "$tag: " "$CHAIN" 2>/dev/null | tail -1)
    [[ -n "$l" ]] && break; sleep 10
  done
  echo "$(date -u +%T) $tag: $l"
  case "$l" in *"profiling started"*)
    python3 "$D/gpuacc_sampler.yihou.py" crsuse2-m2m-102 207495 3600 10 "$D/gpuacc.$tag.yihou.tsv" 2> "$D/gpuacc.$tag.yihou.err" ;;
  *) echo "$tag: no profiling start ($l)";; esac
done
