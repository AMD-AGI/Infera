#!/usr/bin/env bash
# R022: PID-offset workaround for the CLR hipIpcOpenMemHandle PID guard when --pid host is not allowed.
# Private PID namespaces, default IPC. Cases: control (both python3 as PID 1) and offset (exporter
# side advances its PID counter with K throwaway subshells, then execs python3).
# usage: run_pidoffset.yihou.sh <node> <job|-> [image] [exp_gpu=4] [imp_gpu=0] [K=30000]
set -uo pipefail
node=${1:?node}; job=${2-}; IMG=${3:-infera-sglang:v0521-yihou-1010-mc4340}; EG=${4:-4}; IG=${5:-0}; K=${6:-30000}
D=/shared_nfs/yihou_glm52/v0521dbg_20261010
ts=$(date -u +%Y%m%d-%H%M%S); out=$D/diag/out.pidoffset.yihou/$ts; mkdir -p "$out"
if [[ -z "$job" || "$job" == "-" ]]; then SSH=(ssh -o BatchMode=yes -o ClearAllForwardings=yes); W=/tmp/yihou_v0521dbg/pidoffset/$ts
else export PATH=/shared_nfs/yihou_glm52/kvcap_20261008/scripts/shimbin:$PATH SPUR_SSH_JOBS="$node:$job"; SSH=(ssh); W=/mnt/m2m_nobackup/yihou_v0521dbg/pidoffset/$ts; fi
BASE="--device /dev/kfd --device /dev/dri --group-add video --network host --ipc host --cap-add SYS_PTRACE --cap-add IPC_LOCK --security-opt seccomp=unconfined -v $D:$D:ro"
PY="$D/diag/ipc_xcontainer.yihou.py"
echo "workdir $node:$W image=$IMG K=$K"
for c in control offset; do
  cell=$W/$c
  if [[ $c == offset ]]; then
    EXP="--entrypoint bash $IMG $D/diag/pidoffset_exec.yihou.sh $K python3 $PY exp $cell"
  else
    EXP="--entrypoint python3 $IMG $PY exp $cell"
  fi
  "${SSH[@]}" "$node" "mkdir -p $cell && chmod 777 $cell && \
    docker run -d --rm --name yihou-ipc-pidoff-exp-$c $BASE -v $cell:$cell -e HIP_VISIBLE_DEVICES=$EG $EXP >/dev/null; \
    sleep 0.5; docker logs -f yihou-ipc-pidoff-exp-$c > $cell/exp.log 2>&1 & \
    for i in \$(seq 900); do [ -f $cell/handle.bin ] && break; sleep 0.2; done; \
    timeout 120 docker run --rm --name yihou-ipc-pidoff-imp-$c $BASE -v $cell:$cell -e HIP_VISIBLE_DEVICES=$IG --entrypoint python3 $IMG $PY imp $cell > $cell/imp.log 2>&1; \
    [ -f $cell/done ] || echo x > $cell/done; sleep 2; wait"
  r=$("${SSH[@]}" "$node" "cat $cell/exp.log $cell/imp.log 2>/dev/null" | strings)
  printf "%-8s | %s | exp: %s %s | imp: %s %s | %s\n" $c "$(echo "$r" | grep -aoE 'OFFSET_LOOP_S=[0-9.]+' | head -1)" \
    "$(echo "$r" | grep -aoE '\[exp pid=[0-9]+' | head -1)" "$(echo "$r" | grep -a 'RESULT exp' | head -1)" \
    "$(echo "$r" | grep -aoE '\[imp pid=[0-9]+' | head -1)" "$(echo "$r" | grep -aE 'RESULT imp' | head -1)" \
    "$(echo "$r" | grep -a '^\[imp.*hipIpcOpenMemHandle' | sed -E 's/^\[[^]]*\] //' | head -1)" | tee -a "$out/summary.txt"
  "${SSH[@]}" "$node" "cat $cell/exp.log" > "$out/$c.exp.log" 2>/dev/null; "${SSH[@]}" "$node" "cat $cell/imp.log" > "$out/$c.imp.log" 2>/dev/null
done
"${SSH[@]}" "$node" "docker ps -a --format '{{.Names}}' | grep yihou-ipc || echo no-yihou-ipc-containers"
