#!/bin/bash
set -euo pipefail
W=/apps/tas/yaoc/research/topic/infera-with-hyperloom/exp_only/Infera-yx-test/examples/glm52_fake_tp4ep4_goodput/experiments/20260910_1525_n04_c128_c256
MODE=${1:?mix, decode, or sim}
ROUND=${2:?round name}
MEM=${3:-0.85}
CONC=${4:-16}
NAME=glm52-goodput-high-20260910-1525
IMAGE=sha256:416d51effc431e27a4ddeed5cdd8ca6012c68eeabb071f987ce99f1ba9854973
MODEL=/shared_nfs/models/GLM-5.2-MXFP4
PORT=31864
case "$MODE" in mix|decode|sim) ;; *) exit 2;; esac
R="$W/rounds/$ROUND"
mkdir -p "$R" "/data/cyao1002/glm52_profile_20260910_n04_c32/jit-b9a83742f631" "/data/cyao1002/glm52_profile_20260910_n04_c32/triton" "/data/cyao1002/glm52_profile_20260910_n04_c32/torch" "/data/cyao1002/glm52_profile_20260910_n04_c32/hf"
if docker container inspect "$NAME" >/dev/null 2>&1; then
    printf 'Refusing to replace existing container %s; stop explicitly first.\n' "$NAME" >&2
    exit 2
fi
# This probe runs through the HOST daemon: spur exec has no host render nodes.
docker run --rm --name "$NAME-probe" --device=/dev/kfd --device=/dev/dri \
  --entrypoint bash "$IMAGE" -c 'rocm-smi --showmemuse' > "$R/gpu-before.txt" 2>&1
python3 - "$R/gpu-before.txt" <<'PY'
import re,sys
s=open(sys.argv[1]).read()
x=[int(v) for v in re.findall(r'GPU Memory Allocated \(VRAM%\):\s*(\d+)',s)]
if len(x)!=8 or max(x)>2:
    raise SystemExit(f'GPU idle gate failed: {x}; no server launched')
PY
EXTRA=()
ENV=()
PATCH_MOUNTS=(-v "$W/source/dsa_utils_fake_fix.py:/sglang/python/sglang/srt/layers/attention/dsa/utils.py:ro")
if [ "$MODE" != mix ] && [ -f "$W/source/kv_cache_configurator_fake_fix.py" ]; then
    PATCH_MOUNTS+=(-v "$W/source/kv_cache_configurator_fake_fix.py:/sglang/python/sglang/srt/mem_cache/kv_cache_configurator.py:ro")
fi
if [ "$MODE" != mix ] && [ -f "$W/source/eagle_disaggregation_fake_fix.py" ]; then
    PATCH_MOUNTS+=(-v "$W/source/eagle_disaggregation_fake_fix.py:/sglang/python/sglang/srt/speculative/eagle_disaggregation.py:ro")
fi
if [ "$MODE" != mix ]; then
    EXTRA+=(--disaggregation-mode decode --disaggregation-transfer-backend fake)
fi
if [ "$MODE" = sim ]; then
    ENV+=(-e SGLANG_SIMULATE_ACC_LEN=3.61 -e SGLANG_SIMULATE_ACC_METHOD=match-expected -e SGLANG_SIMULATE_ACC_TOKEN_MODE=real-draft-token)
fi
CMD=(python3 -m sglang.launch_server
    --model-path "$MODEL" --served-model-name glm52 --trust-remote-code
    --host 0.0.0.0 --port "$PORT" --tp 4 --ep-size 4
    --kv-cache-dtype fp8_e4m3 --dsa-prefill-backend flydsl --dsa-decode-backend flydsl
    --tool-call-parser glm47 --reasoning-parser glm45
    --chunked-prefill-size 32768 --mem-fraction-static "$MEM"
    --max-running-requests "$CONC" --cuda-graph-max-bs "$CONC" --cuda-graph-bs-decode 1 16 32 64 128 256
    --speculative-algorithm EAGLE --speculative-num-steps 5
    --speculative-eagle-topk 1 --speculative-num-draft-tokens 6
    --dsa-topk-backend aiter --enable-aiter-allreduce-fusion --enable-fused-qk-norm-rope
    --watchdog-timeout 1800 --enable-metrics --decode-log-interval 10 "${EXTRA[@]}")
printf '%q ' "${CMD[@]}" > "$R/server-command.txt"; printf '\n' >> "$R/server-command.txt"
date -u +%FT%TZ > "$R/start-time.txt"
docker run -d --name "$NAME" --label experiment=glm52-faked-1016 -w / \
    --network host --ipc=host --shm-size 32g \
    --device=/dev/kfd --device=/dev/dri --group-add video --group-add render \
    --security-opt seccomp=unconfined --ulimit memlock=-1:-1 \
    -v "$W:$W" -v "/data/models/GLM-5.2-MXFP4:$MODEL:ro" \
    -v "/data/cyao1002/glm52_profile_20260910_n04_c32/jit-b9a83742f631:/jit-cache" -e AITER_JIT_DIR=/jit-cache \
    -v "/data/cyao1002/glm52_profile_20260910_n04_c32/triton:/triton-cache" -e TRITON_CACHE_DIR=/triton-cache \
    -v "/data/cyao1002/glm52_profile_20260910_n04_c32/torch:/torch-cache" -e TORCHINDUCTOR_CACHE_DIR=/torch-cache \
    -v "/data/cyao1002/glm52_profile_20260910_n04_c32/hf:/hf_cache" -e HF_HOME=/hf_cache \
    -e SGLANG_OPT_USE_TOPK_V2=false -e SGLANG_TIMEOUT_KEEP_ALIVE=900 \
    -e PYTHONPATH=/sglang/python -e HIP_VISIBLE_DEVICES=0,1,2,3 -e PYTHONNOUSERSITE=1 -e PYTHONDONTWRITEBYTECODE=1 \
    -e AITER_USE_FLYDSL_MOE_SORTING=1 -e SGLANG_OPT_USE_JIT_KERNEL_GROUPED_TOPK=1 \
    "${ENV[@]}" "${PATCH_MOUNTS[@]}" "$IMAGE" "${CMD[@]}" > "$R/container-id.txt"
docker inspect "$NAME" > "$R/container-inspect.json"
printf 'Started %s mode=%s mem=%s round=%s\n' "$NAME" "$MODE" "$MEM" "$ROUND"
