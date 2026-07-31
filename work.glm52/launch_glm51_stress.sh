#!/usr/bin/env bash
# GLM-5.1-FP8 on the VERIFIED e2e recipe, but with a production-sized cuda-graph
# batch so conc=64 is actually captured (e2e uses --cuda-graph-max-bs 8, which
# would fall back to eager above bs=8 and tell us nothing about steady state).
#
# Everything else mirrors the e2e case that just PASSED:
#   model  = zai-org/GLM-5.1-FP8 (bind-mounted e2e path)
#   tp     = 4
#   args   = --reasoning-parser glm45 --mem-fraction-static 0.85 --trust-remote-code
#   env    = SGLANG_USE_AITER=1  (+ SGLANG_OPT_USE_TOPK_V2=0, required on the
#            v0.5.15.post1 base; without it JIT dies on cooperative_groups.h)
#   DSA    = NOT specified -> ROCm auto -> tilelang, kv-cache-dtype -> bfloat16
# Launched directly via sglang.launch_server (no etcd/router) so the bench can
# hit it straight; the engine config is identical.
set -uo pipefail
IMAGE=${IMAGE:-infera/engine-sglang:test-local}
MODEL=${MODEL:-/mnt/vast/c_huggingface/e2e_models/zai-org/GLM-5.1-FP8}
REAL=${REAL:-/mnt/vast/xiaobo/models/GLM-5.1-FP8}
PORT="${PORT:-30000}"
TP="${TP:-4}"
CGBS="${CGBS:-64}"
MAXRUN="${MAXRUN:-64}"
MEMFRAC="${MEMFRAC:-0.85}"
NAME="${NAME:-glm51-stress}"

docker rm -f "$NAME" >/dev/null 2>&1 || true
sleep 3

docker run -d --name "$NAME" --network host --ipc host --shm-size 64g \
  --device /dev/kfd --device /dev/dri --group-add video --group-add render \
  --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
  -v /mnt/vast:/mnt/vast \
  -v "$REAL":"$MODEL":ro \
  -e SGLANG_USE_AITER=1 \
  -e SGLANG_OPT_USE_TOPK_V2=0 \
  --entrypoint python3 "$IMAGE" \
  -m sglang.launch_server \
    --model-path "$MODEL" --served-model-name glm5.1-fp8 \
    --host 0.0.0.0 --port "$PORT" --tp-size "$TP" --trust-remote-code \
    --reasoning-parser glm45 --mem-fraction-static "$MEMFRAC" \
    --cuda-graph-max-bs "$CGBS" --max-running-requests "$MAXRUN" \
  >/dev/null 2>&1

echo "started $NAME  tp=$TP cgbs=$CGBS maxrun=$MAXRUN memfrac=$MEMFRAC"
echo "  (DSA backend + kv dtype left to ROCm auto = tilelang + bfloat16, as in e2e)"
docker ps --filter name="$NAME" --format '{{.Names}} {{.Status}}'
