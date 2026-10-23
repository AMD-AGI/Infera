#!/usr/bin/env bash
# xferprobe (task dcpx 2026-10-22, ON 135): read-only probe of the KV-transfer path during profiling.
# rdma hw counters on 135 and 136 at t0 and t0+SPAN; N py-spy dumps of the 4 prefill schedulers in between.
# Usage: xferprobe.yihou.sh <round> <label> [SPAN=120] [N=5]
set -uo pipefail
RD=$1; LB=$2; SPAN=${3:-120}; N=${4:-5}
D=/shared_nfs/yihou_glm52/dcpx_20261022; O=$D/probe.$RD.$LB.yihou; mkdir -p $O
C=glm52-v0521-yihou-135-$RD-prefill-0
ctr(){ bash $D/rdmactr.yihou.sh "$1" >> $O/rdma.135.log; ssh -o BatchMode=yes crsuse2-m2m-136 bash $D/rdmactr.yihou.sh "$1" >> $O/rdma.136.log; }
ctr t0
for i in $(seq 1 $N); do
  docker exec $C bash -c 'for p in $(ps -eo pid,comm | awk "/sglang::schedul/{print \$1}"); do echo "===== pid $p"; py-spy dump --pid $p 2>&1; done' > $O/pyspy.$i.$(date -u +%H%M%S).txt 2>&1
  sleep $(( SPAN / N ))
done
ctr t1
echo "$(date -u +%FT%TZ) done" >> $O/done.txt
