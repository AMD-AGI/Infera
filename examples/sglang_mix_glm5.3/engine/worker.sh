#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# The real launcher, and the file that carries the tuned recipe. Runs INSIDE
# the container, staged there by up.sh. Every site-specific value arrives as an
# env var from env.sh; there are no addresses or paths in here.
#
# Launches through `python3 -m infera.engine.sglang` rather than
# sglang.launch_server so infera's etcd discovery and kv-aware routing work.
set -u
MY_IP="${MY_IP:?MY_IP=this node data-plane IP}"
MODEL="${MODEL:?MODEL=weights dir inside the container}"
VARIANT="${VARIANT:?VARIANT=flash-mxfp4|flash-fp8|big-mxfp4|big-fp8}"
ETCD_IP="${ETCD_IP:-$MY_IP}"
ETCD_PORT="${ETCD_PORT:-12379}"
PORT="${PORT:-30000}"
TP="${TP:-4}"
GPUS="${GPUS:-$(seq -s, 0 $((TP - 1)))}"
SERVED="${SERVED:-glm5.3-$VARIANT}"
LOG="${LOG:-/tmp/glm53_mix.log}"

# Decode CUDA graphs. ON by default: measured 12.5 -> 117.9 output tok/s at
# concurrency 1 on flash-mxfp4 and 12-15 -> ~110 on flash-fp8, both ~7.5x, for
# ~33-82 s of capture. The bs list is graph COVERAGE, not a concurrency cap -- a
# decode batch is padded up to the next captured size and anything larger runs
# eager. Sizes above --max-running-requests are dropped at capture time.
#
# BUDGET THE VRAM AT TP4. Capture cost was measured at 15.4-17.4 GB per rank at
# TP4, against ~1.4 GB in an earlier TP8 bring-up of the same model -- fewer
# ranks means each one's graphs cover a larger shard. Do not carry a TP8 figure
# into a TP4 plan.
# Prefill graphs stay disabled: that is what upstream validated on gfx950, and
# prefill is where the DSA/KDA shape variance lives.
CUDA_GRAPH="${CUDA_GRAPH:-1}"
GRAPH_BS="${GRAPH_BS:-1 2 4 8 16 24 32 48 64 96 128}"
KVAWARE="${KVAWARE:-1}"

# Cache-hit accounting. OFF by default because it is not free, but you must turn
# it ON for any workload whose result depends on prefix reuse -- an agentic
# replay, or anything passing bench_serving's --cache-report.
#
# Without it the server still answers normally and simply reports nothing:
# `usage.prompt_tokens_details` comes back **null** through the router and the
# engine logs `#cached-token: 0`. That is indistinguishable from a genuine 0 %
# hit rate, so a cache-sensitive benchmark reads as "the cache never worked"
# rather than "the counter was never enabled". Verified on this stack.
CACHE_REPORT="${CACHE_REPORT:-0}"

# --- common ROCm env --------------------------------------------------------
# SGLANG_USE_AITER gates the AITER fast paths on gfx950. On the flash variants
# it gates PR #36607's mHC pre/post dispatch specifically, and its absence is
# SILENT: the server starts, answers correctly, and is 4.3-5.4x slower with
# nothing in any log saying so. smoke.sh greps for the mHC lines for that reason.
export SGLANG_USE_AITER=1
export SAFETENSORS_FAST_GPU=1 HIP_FORCE_DEV_KERNARG=1 HSA_NO_SCRATCH_RECLAIM=1
export NCCL_IGNORE_CPU_AFFINITY=1
# Stable block hashes -> stable kv-aware keys across restarts.
export PYTHONHASHSEED=0
export SGLANG_HOST_IP="$MY_IP" HOST_IP="$MY_IP"
export INFERA_SGLANG_READY_TIMEOUT="${READY_TIMEOUT:-3600}"
NIC=$(ip -o -4 addr show | awk -v ip="$MY_IP" '$4 ~ ("^" ip "/") {print $2; exit}')
[ -n "$NIC" ] && export SGLANG_LOCAL_IP_NIC="$NIC" GLOO_SOCKET_IFNAME="$NIC"

ARGS=()
case "$VARIANT" in
  flash-*)
    # --- glm5_next family ---------------------------------------------------
    # DSA flags are --dsa-*-backend here. GLM-5.2 used --nsa-*; carrying that
    # spelling forward gets unknown-flag errors.
    # THE KDA-POOL CLAMP IS REAL AND IT FIRES. This family keeps a second
    # memory pool for the linear-attention state, and the scheduler will cap
    # concurrency against it regardless of what you ask for here. Observed on
    # every rank at TP4:
    #   max_running_requests is capped to 200 by the mamba state cache
    #   (max_mamba_cache_size=1000, 5 state slots per request). To raise it:
    #   increase --mamba-full-memory-ratio or --max-mamba-cache-size, or halve
    #   the state size with --mamba-ssm-dtype bfloat16.
    # Read the RESOLVED value out of the worker log, not out of server_args --
    # server_args records what was requested, and reading it instead is how one
    # bring-up concluded the clamp had not fired when it had.
    ARGS+=(--dsa-prefill-backend tilelang --dsa-decode-backend tilelang
           --kv-cache-dtype "${KV_DTYPE:-bfloat16}"
           --context-length "${CTX:-65536}"
           --mem-fraction-static "${GMU:-0.80}"
           --max-running-requests "${MAX_RUNNING:-32}"
           --chunked-prefill-size "${CHUNK:-4096}"
           --max-prefill-tokens "${MAX_PREFILL:-16384}"
           --mm-feature-transport cpu)

    # --disable-shared-experts-fusion is LOAD-BEARING on mxfp4, not tuning.
    # sglang PR #36607 opened the gfx950 branch of glm5_next's fusion gate
    # (glm5_next.py:1414) without carrying the
    # quant_blocks_shared_experts_fusion(quant_config) guard that
    # deepseek_v2.py:3069 has, and QuarkConfig.can_fuse_shared_expert() -- which
    # computes the right answer -- is never consulted. The checkpoint's BF16
    # shared expert is then renamed into routed slot 288 of an MXFP4-packed
    # FusedMoE and weight load dies with
    #   RuntimeError: The size of tensor a (256) must match the size of tensor b
    #   (512) at non-singleton dimension 1
    # in fused_moe_triton/layer.py::_load_w2. Upstream issue #37268 is the same
    # bug on NVFP4/NVIDIA and documents the same workaround.
    # Set SHARED_EXPERT_FUSION=1 to re-enable, e.g. on a uniformly-quantized
    # checkpoint where fusion is both correct and profitable.
    SHARED_EXPERT_FUSION="${SHARED_EXPERT_FUSION:-0}"
    [ "$SHARED_EXPERT_FUSION" = "0" ] && ARGS+=(--disable-shared-experts-fusion)

    if [ "$VARIANT" = "flash-mxfp4" ]; then
      # Quark MXFP4 (fp4 E2M1, 1x32 block scales). --quantization is explicit
      # per the vendor model card; the aiter runner dispatches native FP4 MoE
      # kernels (torch.float4_e2m1fn_x2). With `triton` the checkpoint is
      # dequantized to BF16 GEMMs -- it still serves, and it is much slower.
      ARGS+=(--quantization "${QUANT:-quark}" --moe-runner-backend "${MOE_RUNNER:-aiter}")
      # Vendor-set for this checkpoint and absent from the FP8 recipe. Not noise.
      export SGLANG_OPT_DEEPGEMM_HC_PRENORM=0
    else
      # FP8 original: config.json already carries the quantization, so no
      # --quantization flag.
      ARGS+=(--moe-runner-backend "${MOE_RUNNER:-triton}")
    fi
    ;;

  big-*)
    # --- glm_moe_dsa family: the GLM-5.2 code path --------------------------
    # MANDATORY on gfx950. Without this env block the model serves, returns
    # 200s, and returns GARBAGE, because the sparse-attention indexer takes a
    # path not ported to this architecture. infera.engine.sglang already
    # defaults SGLANG_OPT_USE_TOPK_V2 off on ROCm
    # (infera/engine/rocm_dsa_env.py); it is repeated so that a bare
    # launch_server run of this same recipe behaves identically.
    export SGLANG_ROCM_FUSED_DECODE_MLA=0 SGLANG_OPT_USE_TILELANG_INDEXER=1
    export SGLANG_OPT_USE_TOPK_V2=0 SGLANG_OPT_USE_JIT_NORM=0

    # --ep-size is emitted unconditionally and OUTSIDE any DP-attention branch:
    # expert parallelism and attention parallelism are different axes, and
    # gating both on one condition silently collapses the MoE whenever DPA is
    # off, after which no latency delta is attributable to either.
    # --max-running-requests is passed EXPLICITLY rather than left to the
    # engine's memory-derived default. The default is not wrong, but it is
    # derived from whatever VRAM happens to be free, so two runs of the same
    # recipe on differently-loaded nodes silently get different admission
    # limits -- and a benchmark then measures the limit, not the engine.
    ARGS+=(--ep-size "${EP_SIZE:-$TP}"
           --dsa-prefill-backend tilelang --dsa-decode-backend tilelang
           --kv-cache-dtype "${KV_DTYPE:-fp8_e4m3}"
           --context-length "${CTX:-262144}"
           --max-running-requests "${MAX_RUNNING:-32}"
           --mem-fraction-static "${GMU:-0.80}"
           --chunked-prefill-size "${CHUNK:-65536}")

    # The aiter custom all-reduce kernel deadlocks on this architecture under
    # speculative verify. Disabled independently of MTP so that any "MTP on vs
    # off" comparison stays a one-variable comparison.
    ARGS+=(--disable-custom-all-reduce)

    if [ "$VARIANT" = "big-mxfp4" ]; then
      # Quantization is AUTO-DETECTED from config.json; the vendor card states
      # no --quantization flag is required.
      ARGS+=(--moe-runner-backend "${MOE_RUNNER:-aiter}")
      # Insurance rather than a fix: glm4_moe.py:1174's fusion gate only
      # special-cases w4afp8 and would fuse under quark, but this checkpoint's
      # shared experts are themselves MXFP4, so the precondition is absent.
      # Kept on because upstream #25261 shows this class of mismatch failing
      # SILENTLY with wrong output rather than crashing. Set
      # SHARED_EXPERT_FUSION=1 for a clean single-variable performance round.
      [ "${SHARED_EXPERT_FUSION:-0}" = "0" ] && ARGS+=(--disable-shared-experts-fusion)
    fi
    ;;
  *) echo "unknown VARIANT: $VARIANT" >&2; exit 2 ;;
esac

# MTP/EAGLE is deliberately NOT enabled for either family. Upstream's GLM-5.3
# cookbook disables speculative decoding on AMD because the gfx950 draft kernel
# is unvalidated, while the OneNexus big-MXFP4 card runs EAGLE at 3 steps. That
# contradiction is recorded rather than resolved; do not add --speculative-*
# without re-deriving it.

# kvd / hierarchical cache is OFF. On gfx950 (xnack-) hicache stores raw host
# data_ptr()s that a GPU kernel dereferences while hipHostRegister maps those
# pages at a different device VA, and the process aborts with
# "Memory access fault by GPU node-N on address <host VA>". The fix lives in
# patches/sglang_rocm/; confirm it is in your image before turning either on.

[ "$CACHE_REPORT" = "1" ] && ARGS+=(--enable-cache-report)

if [ "$CUDA_GRAPH" = "1" ]; then
  ARGS+=(--cuda-graph-backend-decode full --cuda-graph-backend-prefill disabled
         --cuda-graph-bs-decode $GRAPH_BS)
else
  ARGS+=(--cuda-graph-backend-decode disabled --cuda-graph-backend-prefill disabled)
fi

INFERA_ARGS=(--advertise-host "$MY_IP" --etcd-endpoint "$ETCD_IP:$ETCD_PORT"
             --discovery-backend etcd --request-transport http --kv-event-transport zmq)
if [ "$KVAWARE" = "1" ]; then
  INFERA_ARGS+=(--kv-events-bind "tcp://0.0.0.0:${KV_PUB_PORT:-5557}"
                --kv-snapshot-port "${KV_SNAP_PORT:-8801}")
else
  INFERA_ARGS+=(--no-enable-kv-events)
fi

echo "[glm53-mix] variant=$VARIANT ip=$MY_IP:$PORT tp=$TP gpus=$GPUS graph=$CUDA_GRAPH -> $LOG"
HIP_VISIBLE_DEVICES="$GPUS" python3 -m infera.engine.sglang \
  --model-path "$MODEL" --served-model-name "$SERVED" --tp-size "$TP" --trust-remote-code \
  --host "$MY_IP" --port "$PORT" \
  --watchdog-timeout 3600 \
  --reasoning-parser glm45 --tool-call-parser glm47 \
  "${ARGS[@]}" "${INFERA_ARGS[@]}" > "$LOG" 2>&1
