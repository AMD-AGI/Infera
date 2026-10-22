#!/usr/bin/env bash
# usage: [PIDHOST=1] [DST_DEV=k] [SRC_DEV=j] run_te_pair.yihou.sh <node> [modes="legacy default"] [image]
set -uo pipefail
node=${1:?node}; MODES=${2:-legacy default}; IMG=${3:-infera-sglang:v0521-yihou-1009}
D=/shared_nfs/yihou_glm52/v0521dbg_20261010
W=/mnt/m2m_nobackup/yihou_v0521dbg/tepair/$(date -u +%Y%m%d-%H%M%S)
SSH=(ssh -o BatchMode=yes -o ClearAllForwardings=yes "$node")
IP=$("${SSH[@]}" "ip -4 -o addr show ens3 | awk '{print \$4}' | cut -d/ -f1")
PIDF=""; [[ "${PIDHOST:-0}" == 1 ]] && PIDF="--pid host"   # mirror engines run with --pid host
COMMON="$PIDF --device /dev/kfd --device /dev/dri --device /dev/infiniband --group-add video --network host --ipc host --cap-add SYS_PTRACE --cap-add IPC_LOCK --security-opt seccomp=unconfined --ulimit memlock=-1:-1 -v $D:$D:ro -e MC_TE_FILTERS=yihou_no_such_rdma_device -e MOONCAKE_DISABLE_HIP_DMABUF=0 -e MC_GID_INDEX=1 -e MC_LOG_LEVEL=INFO -e PYTHONUNBUFFERED=1"
echo "workdir $node:$W ip=$IP image=$IMG pidhost=${PIDHOST:-0}"
for mode in $MODES; do
  cell=$W/$mode; L=""; [[ $mode == legacy ]] && L="-e HSA_ENABLE_IPC_MODE_LEGACY=1"
  echo "=== $mode"
  "${SSH[@]}" "mkdir -p $cell && chmod 777 $cell && \
    docker run -d --rm --name yihou-ipc-te-dst-$mode $COMMON -v $cell:$cell $L -e HIP_VISIBLE_DEVICES=4,5,6,7 --entrypoint bash $IMG -c 'python3 $D/diag/te_pair.yihou.py dst $cell $IP ${DST_DEV:-0} > $cell/dst.log 2>&1' >/dev/null; \
    timeout 300 docker run --rm --name yihou-ipc-te-src-$mode $COMMON -v $cell:$cell $L -e HIP_VISIBLE_DEVICES=0,1,2,3 --entrypoint bash $IMG -c 'python3 $D/diag/te_pair.yihou.py src $cell $IP ${SRC_DEV:-0}' > $cell/src.log 2>&1; touch $cell/src.done 2>/dev/null; \
    for i in \$(seq 300); do grep -q 'exit' $cell/dst.log 2>/dev/null && break; sleep 0.2; done; \
    strings $cell/src.log | grep -aE '^\[src|hip_transport|HipTransport|HCAs|HIP transport|Error|error' | head -30; strings $cell/dst.log | grep -aE '^\[dst|HipTransport|HCAs|HIP transport' | head -30"
done
"${SSH[@]}" "docker ps -a --format '{{.Names}}' | grep yihou-ipc || echo no-yihou-ipc-containers"
