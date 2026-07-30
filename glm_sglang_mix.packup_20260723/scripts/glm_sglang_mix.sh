#!/bin/bash
# GLM-5.1-FP8 on SGLang — single-node PD-mix (one server, prefill+decode together).
# Verified 2026-07-23 on chi2866 (MI355X gfx950), card4-7, temp=0 probes ALL PASS.
#
# Single-node mix => NO RDMA / ionic / MoRIIO / Mooncake needed. Just one server.
# Run this INSIDE the sglang container (see REPRODUCE.md §2 for the docker run).
set -u

MODEL=${MODEL:-/mnt/vast/xiaobo/models/GLM-5.1-FP8}
PORT=${PORT:-30000}
LOG=${LOG:-/mnt/vast/c_huggingface/glm_sglang_mix.log}

# card4-7 ONLY (card0-3 held foreign titan training on the shared node). sglang
# then sees them as local 0-3.
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-4,5,6,7}
export ROCM_VISIBLE_DEVICES=${ROCM_VISIBLE_DEVICES:-4,5,6,7}
# aiter fused kernels (paged-MQA preshuffle, GEMM). GLM's DSA path uses them.
export SGLANG_USE_AITER=1

: > "$LOG"
# Minimal flag set ON PURPOSE: GlmMoeDsaForCausalLM auto-selects the DSA attention
# path (attention_backend=dsa, page_size=64, tilelang prefill/decode, kv bf16). Do
# NOT force the DSv4 flags (--attention-backend dsv4 / --page-size 256) — they fight
# the auto-config. --reasoning-parser glm45 splits GLM's reasoning_content cleanly.
nohup python3 -m sglang.launch_server \
  --model-path "$MODEL" \
  --tp-size 4 --trust-remote-code \
  --host 0.0.0.0 --port "$PORT" \
  --mem-fraction-static 0.85 \
  --reasoning-parser glm45 \
  > "$LOG" 2>&1 &
echo "sglang_pid=$!  log=$LOG"
echo "poll readiness: curl -s http://127.0.0.1:$PORT/health  (CG capture ~15 min total;"
echo "  an ~8-10 min SILENT window during tilelang JIT + aiter GEMM tuning is NORMAL — do not kill)"
