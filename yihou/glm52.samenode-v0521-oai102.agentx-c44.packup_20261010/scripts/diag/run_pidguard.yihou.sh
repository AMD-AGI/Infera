#!/usr/bin/env bash
# Confirm CLR hipIpcOpenMemHandle PID guard (hip_memory.cpp:3381 rocm-7.2.0): exporter PID == importer PID
# -> hipErrorInvalidContext (201), even across containers. Both containers run python3 AS PID 1
# (private PID ns) -> same PID; --pid host makes them differ. usage: run_pidguard.yihou.sh <node> [image] [exp_gpu=4] [imp_gpu=0]
set -uo pipefail
node=${1:?node}; IMG=${2:-infera-sglang:v0521-yihou-1010-mc4340}; EG=${3:-4}; IG=${4:-0}
D=/shared_nfs/yihou_glm52/v0521dbg_20261010
W=/mnt/m2m_nobackup/yihou_v0521dbg/pidguard/$(date -u +%Y%m%d-%H%M%S)
SSH=(ssh -o BatchMode=yes -o ClearAllForwardings=yes "$node")
BASE="--device /dev/kfd --device /dev/dri --group-add video --network host --ipc host --cap-add SYS_PTRACE --cap-add IPC_LOCK --security-opt seccomp=unconfined -v $D:$D:ro"
echo "workdir $node:$W image=$IMG"
for mode in legacy default; do for pid in private host; do
  c=$mode.$pid; cell=$W/$c; L=""; [[ $mode == legacy ]] && L="-e HSA_ENABLE_IPC_MODE_LEGACY=1"
  P=""; [[ $pid == host ]] && P="--pid host"
  "${SSH[@]}" "mkdir -p $cell && chmod 777 $cell && \
    docker run -d --rm --name yihou-ipc-pid-exp-$c $BASE $P -v $cell:$cell $L -e HIP_VISIBLE_DEVICES=$EG --entrypoint python3 $IMG $D/diag/ipc_xcontainer.yihou.py exp $cell >/dev/null; \
    sleep 0.5; docker logs -f yihou-ipc-pid-exp-$c > $cell/exp.log 2>&1 & \
    for i in \$(seq 150); do [ -f $cell/handle.bin ] && break; sleep 0.2; done; \
    timeout 120 docker run --rm --name yihou-ipc-pid-imp-$c $BASE $P -v $cell:$cell $L -e HIP_VISIBLE_DEVICES=$IG --entrypoint python3 $IMG $D/diag/ipc_xcontainer.yihou.py imp $cell > $cell/imp.log 2>&1; \
    [ -f $cell/done ] || echo x > $cell/done; sleep 2; wait"
  r=$("${SSH[@]}" "cat $cell/exp.log $cell/imp.log 2>/dev/null" | strings)
  printf "%-8s pid=%-8s | exp: %s | imp: %s | %s\n" $mode $pid \
    "$(echo "$r" | grep -aoE '\[exp pid=[0-9]+' | head -1) $(echo "$r" | grep -a 'RESULT exp' | head -1)" \
    "$(echo "$r" | grep -aoE '\[imp pid=[0-9]+' | head -1)" \
    "$(echo "$r" | grep -aE 'RESULT imp' | head -1) $(echo "$r" | grep -a '^\[imp.*hipIpcOpenMemHandle' | sed -E 's/^\[[^]]*\] //' | head -1)"
done; done
"${SSH[@]}" "docker ps -a --format '{{.Names}}' | grep yihou-ipc || echo no-yihou-ipc-containers"
