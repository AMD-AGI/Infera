#!/usr/bin/env bash
# B' microbench on ONE pair of GPUs in ONE container. usage: run_peer_copy_bench.yihou.sh <node> <job|-> [src_gpu=0] [dst_gpu=4] [image]
set -uo pipefail
node=${1:?node}; job=${2-}; SG=${3:-0}; DG=${4:-4}; IMG=${5:-infera-sglang:v0521-yihou-1010-mc4340}
D=/shared_nfs/yihou_glm52/v0521dbg_20261010
ts=$(date -u +%Y%m%d-%H%M%S); out=$D/diag/bprime/out.$ts.yihou; mkdir -p "$out"
if [[ -z "$job" || "$job" == "-" ]]; then SSH=(ssh -o BatchMode=yes -o ClearAllForwardings=yes)
else export PATH=/shared_nfs/yihou_glm52/kvcap_20261008/scripts/shimbin:$PATH SPUR_SSH_JOBS="$node:$job"; SSH=(ssh); fi
echo "[pcb] $node gpus=$SG->$DG image=$IMG out=$out"
"${SSH[@]}" "$node" "timeout 900 docker run --rm --name yihou-ipc-pcb --device /dev/kfd --device /dev/dri --group-add video --network none --ipc host --cap-add SYS_PTRACE --security-opt seccomp=unconfined -e HIP_VISIBLE_DEVICES=$SG,$DG -e PYTHONUNBUFFERED=1 ${PB_ENV:-} -v $D:$D:ro --entrypoint python3 $IMG $D/diag/bprime/peer_copy_bench.yihou.py" > "$out/bench.log" 2>&1
grep -aE "peer access|blocks=|RESULT|VERDICT|Error|error" "$out/bench.log" | cut -c1-400
echo "[pcb] log: $out/bench.log"
