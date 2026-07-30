#!/bin/bash
# GLM-5.1-FP8 on ATOM — single-node (one server, prefill+decode together).
# Verified 2026-07-23 on chi2866 (MI355X gfx950), card4-7, temp=0 probes ALL PASS.
#
# Single-node => NO RDMA / ionic / MoRIIO / Mooncake. Just one ATOM server.
# This starts the container AND the server. Run from the HOST (it does docker run).
set -u

MODEL=${MODEL:-/mnt/vast/xiaobo/models/GLM-5.1-FP8}
IMG=${IMG:-infera/engine-atom:kimi}      # already loaded on the node; do NOT re-load (45GB, nearly filled /)
CTR=${CTR:-glm_atom_c_hf}
PORT=${PORT:-8000}
LOG=${LOG:-/mnt/vast/c_huggingface/glm_atom_mix.log}

docker rm -f "$CTR" 2>/dev/null || true

# card4-7 ONLY (card0-3 held foreign titan training). HSA_NO_SCRATCH_RECLAIM=1 is
# harmless on gfx950 (mandatory on gfx942). Model is RO-mounted from /mnt/vast so
# the container writable layer stays tiny (the node / is disk-tight).
docker run -d --name "$CTR" \
  --device=/dev/kfd --device=/dev/dri --ipc=host --shm-size=32g \
  --group-add video --group-add render --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -e HIP_VISIBLE_DEVICES=4,5,6,7 -e ROCM_VISIBLE_DEVICES=4,5,6,7 \
  -e HSA_NO_SCRATCH_RECLAIM=1 -e AITER_LOG_LEVEL=WARNING \
  -p ${PORT}:${PORT} \
  -v "$MODEL":"$MODEL" -v /mnt/vast:/mnt/vast \
  --entrypoint bash "$IMG" -lc "python -m atom.entrypoints.openai_server \
    --model $MODEL \
    --kv_cache_dtype fp8 -tp 4 \
    --host 0.0.0.0 --server-port ${PORT} 2>&1 | tee $LOG"

echo "[launch] container $CTR started; follow: docker logs -f $CTR"
echo "[launch] wait for 'Server started successfully and ready to accept requests'"
echo "[launch] first start compiles aiter MoE + cudagraph capture [1..512] -> a few min"
echo "[note] NO --method mtp: GLM ships no MTP/nextn draft weights, and gfx950 plain"
echo "       decode is correct (the gfx942 broken-plain-decode bug does not reproduce)."
