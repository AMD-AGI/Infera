#!/usr/bin/env bash
# probe_mem (task llrepro 2026-10-26, ON a node): read-only. Usage: probe_mem.yihou.sh <outprefix> <seconds>
# Every 10 s: rocm-smi VRAM used/total + use% + power (json), docker stats of our yihou-llr-* containers (CPU%, host mem),
# host free -g; plus the 1 s RDMA/CPU probe (node_probe1s.yihou.sh) in parallel.
o=$1; s=${2:-180}; h=$(hostname -s); D=/shared_nfs/yihou_glm52/llrepro_20261023
bash $D/node_probe1s.yihou.sh $o $s &
for i in $(seq 1 $((s/10))); do
  echo "== $(date -u +%T)" >> $o.$h.vram
  rocm-smi --showmeminfo vram --showuse --showpower --json 2>/dev/null >> $o.$h.vram; echo >> $o.$h.vram
  echo "== $(date -u +%T)" >> $o.$h.dstats
  docker stats --no-stream --format '{{.Name}} cpu={{.CPUPerc}} mem={{.MemUsage}}' 2>/dev/null | grep yihou-llr >> $o.$h.dstats
  free -g | head -2 >> $o.$h.dstats
  sleep 10
done
wait; echo done > $o.$h.memdone
