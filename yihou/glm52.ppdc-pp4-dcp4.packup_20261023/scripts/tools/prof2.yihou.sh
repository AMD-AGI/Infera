#!/usr/bin/env bash
# prof2 (task dcpx7b 2026-10-22, ON 135): read-only transfer-path profile during profiling.
#  - per-thread CPU (top -H) of the 4 prefill schedulers, 3 x 5 s
#  - rdma counters: 135 tx + 136 rx per rail every 30 s (4 samples)
#  - py-spy dump --native of the 4 prefill schedulers, N samples every 5 s
#  - py-spy dump of the decode schedulers on 136, 3 samples
# Usage: prof2.yihou.sh <round> <label> [N=12]
set -uo pipefail
RD=$1; LB=$2; N=${3:-12}
D=/shared_nfs/yihou_glm52/dcpx_20261022; O=$D/prof2.$RD.$LB.yihou; mkdir -p $O
C=glm52-v0521-yihou-135-$RD-prefill-0; CD=glm52-v0521-yihou-135-$RD-decode-0
ctr(){ bash $D/rdmactr.yihou.sh "$1" >> $O/rdma.135.log; ssh -o BatchMode=yes crsuse2-m2m-136 bash $D/rdmactr.yihou.sh "$1" >> $O/rdma.136.log; }
( for i in 0 1 2 3; do ctr s$i; sleep 30; done ) &
docker exec $C bash -c 'top -H -b -n 3 -d 5 -p $(ps -eo pid,comm | awk "/sglang::schedul/{print \$1}" | paste -sd,)' > $O/topH.txt 2>&1 &
for i in $(seq 1 $N); do
  docker exec $C bash -c 'for p in $(ps -eo pid,comm | awk "/sglang::schedul/{print \$1}"); do echo "===== pid $p"; py-spy dump --native --pid $p 2>&1; done' > $O/pyspy.$i.$(date -u +%H%M%S).txt 2>&1
  sleep 5
done
for i in 1 2 3; do
  ssh -o BatchMode=yes crsuse2-m2m-136 "docker exec $CD bash -c 'for p in \$(ps -eo pid,comm | awk \"/sglang::schedul/{print \\\$1}\"); do echo \"===== pid \$p\"; py-spy dump --pid \$p 2>&1; done'" > $O/dec.pyspy.$i.txt 2>&1
  sleep 5
done
wait
echo "$(date -u +%FT%TZ) done" >> $O/done.txt
