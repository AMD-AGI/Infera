#!/usr/bin/env bash
# H-X1 microbench, 2 GPUs in one container; runs with HSA_ENABLE_SDMA unset and =0.
# usage: run_copy_interference.yihou.sh <node> <job|-> [src=0] [dst=4] [image]
set -uo pipefail
node=${1:?node}; job=${2-}; SG=${3:-0}; DG=${4:-4}; IMG=${5:-infera-sglang:v0521-yihou-1010-mc4340}
D=/shared_nfs/yihou_glm52/v0521dbg_20261010
ts=$(date -u +%Y%m%d-%H%M%S); out=$D/diag/bprime/out.ci.$ts.yihou; mkdir -p "$out"
if [[ -z "$job" || "$job" == "-" ]]; then SSH=(ssh -o BatchMode=yes -o ClearAllForwardings=yes)
else export PATH=/shared_nfs/yihou_glm52/kvcap_20261008/scripts/shimbin:$PATH SPUR_SSH_JOBS="$node:$job"; SSH=(ssh); fi
for sd in unset 0; do
  E=""; [[ $sd == 0 ]] && E="-e HSA_ENABLE_SDMA=0"
  "${SSH[@]}" "$node" "timeout 600 docker run --rm --name yihou-ipc-ci --device /dev/kfd --device /dev/dri --group-add video --network none --ipc host --cap-add SYS_PTRACE --security-opt seccomp=unconfined -e HIP_VISIBLE_DEVICES=$SG,$DG -e PYTHONUNBUFFERED=1 $E ${CI_ENV:-} -v $D:$D:ro --entrypoint python3 $IMG $D/diag/bprime/copy_interference.yihou.py" > "$out/sdma_$sd.log" 2>&1
  grep -aE "^env|RESULT|Error|error" "$out/sdma_$sd.log" | cut -c1-300
done
echo "[ci] logs: $out"
