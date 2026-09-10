#!/bin/bash
set -euo pipefail
W=/shared_nfs/yihou/playground/glm52_decode_fake_20260909_1016
MODE=${1:?mix, decode, or sim}
ROUND=${2:?round name}
MEM=${3:-0.85}
NAME=g52-faked-1016
IMAGE=sha256:b9a83742f631d7321c6b75aceb031a8b347c70f97c62ef8c62370c686853de7d
MODEL=/shared_nfs/models/GLM-5.2-MXFP4
PORT=31816
case "$MODE" in mix|decode|sim) ;; *) exit 2;; esac
R="$W/rounds/$ROUND"
mkdir -p "$R" "$W/cache/jit-b9a83742f631" "$W/cache/triton" "$W/cache/torch" "$W/cache/hf"
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
    --host 0.0.0.0 --port "$PORT" --tp 8 --ep-size 1
    --kv-cache-dtype fp8_e4m3 --dsa-prefill-backend flydsl --dsa-decode-backend flydsl
    --tool-call-parser glm47 --reasoning-parser glm45
    --chunked-prefill-size 32768 --mem-fraction-static "$MEM"
    --max-running-requests 16 --cuda-graph-max-bs 16
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
    -v "$W:$W" -v "$MODEL:$MODEL:ro" \
    -v "$W/cache/jit-b9a83742f631:/jit-cache" -e AITER_JIT_DIR=/jit-cache \
    -v "$W/cache/triton:/triton-cache" -e TRITON_CACHE_DIR=/triton-cache \
    -v "$W/cache/torch:/torch-cache" -e TORCHINDUCTOR_CACHE_DIR=/torch-cache \
    -v "$W/cache/hf:/hf_cache" -e HF_HOME=/hf_cache \
    -e SGLANG_OPT_USE_TOPK_V2=false -e SGLANG_TIMEOUT_KEEP_ALIVE=900 \
    -e PYTHONNOUSERSITE=1 -e PYTHONDONTWRITEBYTECODE=1 \
    -e AITER_USE_FLYDSL_MOE_SORTING=1 -e SGLANG_OPT_USE_JIT_KERNEL_GROUPED_TOPK=1 \
    "${ENV[@]}" "${PATCH_MOUNTS[@]}" "$IMAGE" "${CMD[@]}" > "$R/container-id.txt"
docker inspect "$NAME" > "$R/container-inspect.json"
printf 'Started %s mode=%s mem=%s round=%s\n' "$NAME" "$MODE" "$MEM" "$ROUND"
