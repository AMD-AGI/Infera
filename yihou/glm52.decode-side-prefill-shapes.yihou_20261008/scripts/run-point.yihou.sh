#!/usr/bin/env bash
# One decode-leg point through the forked internal bench, inside the container.
# Usage (on the node): NAME=<iter> BS=16 ISL=110000 GRAPH=off|on SHAPE_STEP=12 \
#                      PROFILE=1 bash run-point.yihou.sh [extra bench args...]
# Detached: the in-container process owns its log; the caller returns at once.
set -euo pipefail
WS=/home/yihou/dev/git/infera.glm52.pd/yihou/glm52.decode-side-prefill-shapes.yihou_20261008
BENCH=$WS/scripts/decode-bench-yihou/bench/profile_decode.py
CONTAINER=${CONTAINER:-glm52-dsp-yihou}
MODEL=/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4
SCRATCH=/mnt/m2m_nobackup/yihou/dsp-yihou
: "${NAME:?}"; BS=${BS:-16}; ISL=${ISL:-110000}; OSL=${OSL:-1000}
GRAPH=${GRAPH:-off}; MAX_STEPS=${MAX_STEPS:-30}; WARMUP=${WARMUP:-3}
SHAPE_STEP=${SHAPE_STEP:--1}; PROFILE=${PROFILE:-0}
PSTART=${PSTART:-10}; PNUM=${PNUM:-3}
OUT=$SCRATCH/out/$NAME
[[ ! -e "$OUT" ]] || { echo "refusing existing $OUT" >&2; exit 1; }
mkdir -p "$OUT"

# Deployment decode-leg flags (server-info.29257.json / bench-harness engine.sh).
ARGS=(--model-path "$MODEL" --result-dir "$OUT"
      --tp-size 4 --ep-size 1 --enable-dp-attention
      --batch-size "$BS" --max-running-requests 64
      --input-len "$ISL" --output-len "$OSL" --accept-length 3.61
      --accept-method match-expected --accept-token-mode real-draft-token
      --warmup-steps "$WARMUP" --max-steps "$MAX_STEPS" --log-interval 5
      --kv-cache-dtype fp8_e4m3 --mem-fraction-static 0.85 --chunked-prefill-size 32768
      --dsa-prefill-backend triton --dsa-decode-backend triton --dsa-topk-backend sgl-kernel
      --json-model-override-args '{"index_share_for_mtp_iteration":true}'
      --enable-fused-qk-norm-rope --enable-aiter-allreduce-fusion --disable-custom-all-reduce
      --shape-capture-step "$SHAPE_STEP")
[[ "$GRAPH" == off ]] && ARGS+=(--disable-cuda-graph)
[[ "$PROFILE" == 1 ]] && ARGS+=(--profile --profile-start-step "$PSTART" --profile-num-steps "$PNUM"
                                 --profile-ranks 0 --profile-top-n 40)
[[ -n "${EAGER:-}" ]] && ARGS+=(--eager-window $EAGER)   # "START NUM"
ARGS+=("$@")
printf '%q ' python3 "$BENCH" "${ARGS[@]}" > "$OUT/command.txt"; echo >> "$OUT/command.txt"
printf -v INNER '%q ' python3 "$BENCH" "${ARGS[@]}"
docker exec -d -w "$OUT" "$CONTAINER" bash -c "cd $(printf %q "$WS/scripts/decode-bench-yihou/bench") && $INNER > $(printf %q "$OUT/runtime.log") 2>&1; echo \$? > $(printf %q "$OUT/exit_code")"
echo "launched $NAME -> $OUT"
