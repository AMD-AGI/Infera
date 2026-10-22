#!/usr/bin/env bash
# Extended cross-container IPC test. usage: run_ipcx2.yihou.sh <node> [image]
set -uo pipefail
node=${1:?node}; IMG=${2:-infera-sglang:v0521-yihou-1009}
D=/shared_nfs/yihou_glm52/v0521dbg_20261010
W=/mnt/m2m_nobackup/yihou_v0521dbg/ipctest2/$(date -u +%Y%m%d-%H%M%S)
SSH=(ssh -o BatchMode=yes -o ClearAllForwardings=yes "$node")
COMMON="--device /dev/kfd --device /dev/dri --group-add video --network host --ipc host --cap-add SYS_PTRACE --cap-add IPC_LOCK --security-opt seccomp=unconfined -v $D:$D:ro"
VARS="${VARS:-M0 M3 W0 W3}"
echo "workdir $node:$W image=$IMG"
for mode in default legacy; do
  cell=$W/$mode; L=""; [[ $mode == legacy ]] && L="-e HSA_ENABLE_IPC_MODE_LEGACY=1"
  "${SSH[@]}" "mkdir -p $cell && \
    docker run -d --rm --name yihou-ipc2-exp-$mode $COMMON -v $cell:$cell $L -e HIP_VISIBLE_DEVICES=4,5,6,7 --entrypoint bash $IMG -c 'python3 $D/diag/ipc_xcontainer2.yihou.py exp $cell 4 > $cell/exp.log 2>&1' >/dev/null; \
    for i in \$(seq 300); do [ -f $cell/handles.json ] && break; sleep 0.2; done; \
    for v in $VARS; do timeout 120 docker run --rm --name yihou-ipc2-imp-$mode-\$v $COMMON -v $cell:$cell $L -e HIP_VISIBLE_DEVICES=0,1,2,3 --entrypoint bash $IMG -c \"python3 $D/diag/ipc_xcontainer2.yihou.py imp $cell \$v\" > $cell/imp.\$v.log 2>&1; [ -f $cell/done.\$v ] || touch $cell/done.\$v; done; \
    for i in \$(seq 100); do grep -q exit $cell/exp.log 2>/dev/null && break; sleep 0.2; done; \
    strings $cell/exp.log | grep -a '^\[exp'; for v in $VARS; do strings $cell/imp.\$v.log | grep -a RESULT || echo \"RESULT $mode \$v (none) \$(strings $cell/imp.\$v.log | tail -2 | tr '\n' ' ')\"; done"
done
"${SSH[@]}" "docker ps -a --format '{{.Names}}' | grep yihou-ipc || echo no-yihou-ipc-containers"
