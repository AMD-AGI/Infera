#!/bin/bash
# One rung of the Track-F ladder (PLAN.md §3). Bare sglang, no infera, no etcd,
# no router. Exactly one variable differs between adjacent rungs.
#
#   RUNG=0  vendor image (rocm724) + bind-mounted c821c425 + bare launch_server
#   RUNG=1  our base image (rocm720) + bind-mounted c821c425 + bare launch_server
#   RUNG=2  our built image (overlay baked in)               + bare launch_server
#
# Flags below are the OneNexus GLM-5.3-Flash-MXFP4 model card, verbatim. Nothing
# here is tuned; a rung that passes is a reference point, not a recipe.
set -u
RUNG="${RUNG:-0}"
MY_IP="${MY_IP:-127.0.0.1}"
PORT="${PORT:-30010}"
TP="${TP:-4}"
GPUS="${GPUS:-0,1,2,3}"
MODEL="${MODEL:-/apps/data/models/GLM-5.3-Flash-MXFP4}"
SRC="${SRC:-/apps/yihou/glm53.series.workspace_20260901/probe/sglang/python/sglang}"
TAG="${TAG:-}"
CTR="${CTR:-glm53_rung${RUNG}${TAG:+_$TAG}}"
# Flag axes. Empty QUANT omits --quantization entirely (auto-detect from
# config.json), which is what the BIG GLM-5.3-MXFP4 card says is correct;
# the Flash card passes it explicitly. One axis per run.
QUANT="${QUANT-quark}"
MOE_RUNNER="${MOE_RUNNER:-aiter}"
# Extra launch args, appended verbatim. Used to test one flag at a time.
EXTRA_ARGS="${EXTRA_ARGS:-}"
LOGDIR="${LOGDIR:-/apps/yihou/glm53.series.workspace_20260901/logs}"
LOG="$LOGDIR/rung${RUNG}${TAG:+_$TAG}.log"
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

VENDOR_IMAGE=lmsysorg/sglang-rocm:v0.5.18-rocm724-mi35x-20260822
OURBASE_IMAGE=lmsysorg/sglang:v0.5.18-rocm720-mi35x
BUILT_IMAGE=infera/engine-sglang:glm53-c821c425

case "$RUNG" in
  0) IMAGE=$VENDOR_IMAGE;  MOUNT_SRC=1 ;;
  1) IMAGE=$OURBASE_IMAGE; MOUNT_SRC=1 ;;
  2) IMAGE=$BUILT_IMAGE;   MOUNT_SRC=0 ;;
  *) echo "RUNG must be 0, 1 or 2"; exit 2 ;;
esac

echo "===== rung $RUNG${TAG:+/$TAG}: image=$IMAGE mount_src=$MOUNT_SRC tp=$TP gpus=$GPUS quant=${QUANT:-<auto>} moe=$MOE_RUNNER ====="
mkdir -p "$LOGDIR"

docker rm -f "$CTR" >/dev/null 2>&1
# Only ever reclaims OUR containers; a foreign KFD process aborts instead.
OWN_CTR_RE="^(${CTR})$" GPUS="$GPUS" bash "$SELF/reset_gpus.sh" \
  || { echo "  ABORT: GPUs $GPUS not available"; exit 1; }

# /apps/data/models is its own mount on this host and does NOT come along when
# you bind its parent, so it gets an explicit bind of its own.
SRC_MOUNT=()
[ "$MOUNT_SRC" = "1" ] && SRC_MOUNT=(-v "$SRC":/sgl-workspace/sglang/python/sglang:ro)

docker run -d --name "$CTR" --network=host --ipc=host --shm-size=64G \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add render --cap-add=SYS_PTRACE --cap-add=IPC_LOCK \
  --security-opt seccomp=unconfined --ulimit memlock=-1:-1 \
  -v /apps/data/models:/apps/data/models:ro \
  -v /apps/yihou/glm53.series.workspace_20260901:/apps/yihou/glm53.series.workspace_20260901:ro \
  "${SRC_MOUNT[@]}" \
  "$IMAGE" sleep infinity >/dev/null || { echo "  container failed to start"; exit 1; }
sleep 5

docker exec "$CTR" bash -lc "python3 -c \"import importlib.util as u; s=u.find_spec('sglang.srt.models.glm5_next'); print('glm5_next:', s.origin if s else 'MISSING')\"" 2>&1 | grep -v libtinfo

# Vendor recipe, verbatim. HIP_VISIBLE_DEVICES rather than ROCR_ so the ranks
# see contiguous device ids 0..TP-1.
docker exec -d "$CTR" env \
  HIP_VISIBLE_DEVICES="$GPUS" \
  SGLANG_USE_AITER=1 \
  SGLANG_OPT_DEEPGEMM_HC_PRENORM=0 \
  bash -lc "python3 -m sglang.launch_server \
    --model-path '$MODEL' \
    --served-model-name glm5.3-flash-mxfp4 \
    --tp-size $TP \
    ${QUANT:+--quantization $QUANT} \
    --trust-remote-code \
    --disable-cuda-graph \
    --context-length 65536 \
    --mem-fraction-static 0.80 \
    --max-running-requests 32 \
    --chunked-prefill-size 4096 \
    --max-prefill-tokens 16384 \
    --dsa-prefill-backend tilelang \
    --dsa-decode-backend tilelang \
    --kv-cache-dtype bfloat16 \
    --moe-runner-backend $MOE_RUNNER \
    --reasoning-parser glm45 \
    --tool-call-parser glm47 \
    --mm-feature-transport cpu \
    $EXTRA_ARGS \
    --host 0.0.0.0 --port $PORT > /tmp/rung.log 2>&1"

echo "  launched -> /tmp/rung.log in $CTR (mirrored to $LOG at the end)"
for i in $(seq 1 180); do
  if docker exec "$CTR" curl -sf -m3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "  SERVING after $((i*10))s"; break
  fi
  if ! docker exec "$CTR" pgrep -f launch_server >/dev/null 2>&1; then
    echo "  DIED after $((i*10))s"; break
  fi
  sleep 10
done
docker exec "$CTR" cat /tmp/rung.log > "$LOG" 2>/dev/null
echo "  log -> $LOG"
