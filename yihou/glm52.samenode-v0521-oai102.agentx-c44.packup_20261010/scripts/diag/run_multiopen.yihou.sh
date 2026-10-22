#!/usr/bin/env bash
# usage: run_multiopen.yihou.sh <node> [image]
set -uo pipefail
node=${1:?node}; IMG=${2:-infera-sglang:v0521-yihou-1009}
D=/shared_nfs/yihou_glm52/v0521dbg_20261010
W=/mnt/m2m_nobackup/yihou_v0521dbg/multiopen/$(date -u +%Y%m%d-%H%M%S)
SSH=(ssh -o BatchMode=yes -o ClearAllForwardings=yes "$node")
COMMON="--device /dev/kfd --device /dev/dri --group-add video --network host --ipc host --cap-add SYS_PTRACE --cap-add IPC_LOCK --security-opt seccomp=unconfined -v $D:$D:ro"
echo "workdir $node:$W"
for mode in default legacy; do for nv in 2 8; do
  cell=$W/$mode.nv$nv; L=""; [[ $mode == legacy ]] && L="-e HSA_ENABLE_IPC_MODE_LEGACY=1"
  echo "=== $mode nviews=$nv"
  "${SSH[@]}" "mkdir -p $cell && \
    docker run -d --rm --name yihou-ipc-mo-exp-$mode-$nv $COMMON -v $cell:$cell $L -e HIP_VISIBLE_DEVICES=4,5,6,7 --entrypoint bash $IMG -c 'python3 $D/diag/ipc_multiopen.yihou.py exp $cell $nv > $cell/exp.log 2>&1' >/dev/null; \
    for i in \$(seq 300); do [ -f $cell/handles.json ] && break; sleep 0.2; done; \
    timeout 120 docker run --rm --name yihou-ipc-mo-imp-$mode-$nv $COMMON -v $cell:$cell $L -e HIP_VISIBLE_DEVICES=0,1,2,3 --entrypoint bash $IMG -c 'python3 $D/diag/ipc_multiopen.yihou.py imp $cell' > $cell/imp.log 2>&1; touch $cell/done; \
    for i in \$(seq 100); do grep -q exit $cell/exp.log 2>/dev/null && break; sleep 0.2; done; \
    strings $cell/exp.log | grep -a 'get view' | sed 's/ h16=.*//'; strings $cell/imp.log | grep -a '^IMP' || strings $cell/imp.log | tail -3"
done; done
"${SSH[@]}" "docker ps -a --format '{{.Names}}' | grep yihou-ipc || echo no-yihou-ipc-containers"
