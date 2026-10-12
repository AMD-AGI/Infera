#!/usr/bin/env bash
# R038 follow-up: copy starved behind long kernels -- lever arms. 2 GPUs, one container per config.
# usage: run_copy_levers.yihou.sh <node> <job|-> [src=0] [dst=4] [image]
set -uo pipefail
node=${1:?node}; job=${2-}; SG=${3:-0}; DG=${4:-4}; IMG=${5:-infera-sglang:v0521-yihou-1010-mc4340}
D=/shared_nfs/yihou_glm52/v0521dbg_20261010
ts=$(date -u +%Y%m%d-%H%M%S); out=$D/diag/bprime/out.levers.$ts.yihou; mkdir -p "$out"
if [[ -z "$job" || "$job" == "-" ]]; then SSH=(ssh -o BatchMode=yes -o ClearAllForwardings=yes)
else export PATH=/shared_nfs/yihou_glm52/kvcap_20261008/scripts/shimbin:$PATH SPUR_SSH_JOBS="$node:$job"; SSH=(ssh); fi
L="-e CI_CKSUM=0 -e CI_LOADS=idle,gemm_stream,spin_stream"
# name | env
CONFIGS=(
  "a_base_normal64|$L"
  "b_hwq16_normal64|$L -e GPU_MAX_HW_QUEUES=16"
  "c_high64|$L -e CI_STREAM_MODE=high"
  "d_cumask8|$L -e CI_STREAM_MODE=cumask -e CI_STREAMS=8"
  "e_normal1|$L -e CI_STREAMS=1"
  "f_peersdma0_normal64|$L -e HSA_ENABLE_PEER_SDMA=0"
  "h_hwq16_normal8|$L -e GPU_MAX_HW_QUEUES=16 -e CI_STREAMS=8"
  "i_hwq40_normal8|$L -e GPU_MAX_HW_QUEUES=40 -e CI_STREAMS=8"
  "g_log_engine|-e CI_CKSUM=0 -e CI_LOADS=idle -e CI_BLOCKS=2 -e CI_REPS=1 -e CI_STREAMS=2 -e AMD_LOG_LEVEL=4"
)
for c in "${CONFIGS[@]}"; do
  name=${c%%|*}; E=${c#*|}
  "${SSH[@]}" "$node" "timeout 300 docker run --rm --name yihou-ipc-cl --device /dev/kfd --device /dev/dri --group-add video --network none --ipc host --cap-add SYS_PTRACE --security-opt seccomp=unconfined -e HIP_VISIBLE_DEVICES=$SG,$DG -e PYTHONUNBUFFERED=1 $E -v $D:$D:ro --entrypoint python3 $IMG $D/diag/bprime/copy_interference.yihou.py" > "$out/$name.log" 2>&1
  echo "== $name"; grep -aE "^stream_mode|^RESULT|Error|error" "$out/$name.log" | cut -c1-260
done
# engine evidence from the CLR debug log
grep -aE "HSA Copy|shaderCopyBuffer|Created SWq|Selected queue|copy_engine|ShaderCopy|Blit" "$out/g_log_engine.log" | sort | uniq -c | sort -rn | head -20 | cut -c1-260
echo "[levers] logs: $out"
