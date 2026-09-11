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
VARIANT="${VARIANT:?VARIANT=big-mxfp4|big-fp8}"
ETCD_IP="${ETCD_IP:-$MY_IP}"
ETCD_PORT="${ETCD_PORT:-12379}"
PORT="${PORT:-30000}"
TP="${TP:-4}"
GPUS="${GPUS:-$(seq -s, 0 $((TP - 1)))}"
SERVED="${SERVED:-glm5.3-$VARIANT}"
LOG="${LOG:-/tmp/glm53_mix.log}"

# Decode CUDA graphs. ON by default: worth roughly 7.5x output throughput at
# concurrency 1, for 33-82 s of capture. Prefill graphs stay disabled, which is
# what upstream validated on gfx950. Pass --cuda-graph-*-decode, not the
# deprecated --cuda-graph-max-bs: it still resolves but breaks on a base bump.
CUDA_GRAPH="${CUDA_GRAPH:-1}"
# The bs list is graph COVERAGE, not a concurrency cap: a decode batch is padded
# up to the next captured size, anything larger runs eager, and sizes above
# --max-running-requests are dropped at capture time. Budget VRAM per shape --
# capture costs 15.4-17.4 GB per rank at TP4 against ~1.4 GB at TP8.
GRAPH_BS="${GRAPH_BS:-1 2 4 8 16 24 32 48 64 96 128}"
KVAWARE="${KVAWARE:-1}"

# Cache-hit accounting. OFF by default because it is not free; turn it ON for any
# workload whose result depends on prefix reuse. Without it the server answers
# normally and reports nothing -- usage.prompt_tokens_details comes back null and
# the engine logs `#cached-token: 0`, indistinguishable from a genuine 0 % rate.
CACHE_REPORT="${CACHE_REPORT:-0}"

# --- common ROCm env --------------------------------------------------------
# SGLANG_USE_AITER gates the AITER fast paths on gfx950. Its absence is SILENT:
# the server starts and answers correctly, only slower, with nothing in any log
# saying so.
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
  big-*)
    # --- glm_moe_dsa family: the GLM-5.2 code path. This env block is MANDATORY
    # on gfx950. Without it the model serves, returns 200s, and returns GARBAGE,
    # because the sparse-attention indexer takes a path not ported to this
    # architecture. Mirrors infera/engine/rocm_dsa_env.py's ROCm defaults.
    export SGLANG_ROCM_FUSED_DECODE_MLA=0 SGLANG_OPT_USE_TILELANG_INDEXER=1
    export SGLANG_OPT_USE_TOPK_V2=0 SGLANG_OPT_USE_JIT_NORM=0

    # Under DP-attention the engine prints PER-RANK values while
    # /get_server_info reports the GLOBAL ones -- 256/8 and 65536/8 read exactly
    # like a clamp. --ep-size stays outside any DPA branch (a different axis),
    # and --max-running-requests is explicit so free VRAM cannot set the limit.
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
      # Insurance rather than a fix: glm4_moe.py's fusion gate only special-cases
      # w4afp8, and this checkpoint's shared experts are themselves MXFP4, so the
      # mis-load precondition is absent. Kept on because upstream #25261 shows
      # this class failing SILENTLY with wrong output. SHARED_EXPERT_FUSION=1 lifts it.
      [ "${SHARED_EXPERT_FUSION:-0}" = "0" ] && ARGS+=(--disable-shared-experts-fusion)
    fi
    ;;
  flash-*)
    echo "VARIANT=$VARIANT is not served by this kit, which covers the big family only." >&2
    exit 2 ;;
  *) echo "unknown VARIANT: $VARIANT (want big-mxfp4|big-fp8)" >&2; exit 2 ;;
esac

# MTP/EAGLE is NOT enabled here. Why: it is validated on this checkpoint only in
# the 1P1D PD kit, and this is an aggregated shape -- untested, not ruled out.
# How to try it: MTP(3,1,4) as the PD kit passes it, plus --disable-custom-all-reduce,
# which is already on below. If missing: decode emits one token per step.

# kvd / hierarchical cache is OFF. On gfx950 (xnack-) hicache stores raw host
# data_ptr()s that a GPU kernel dereferences while hipHostRegister maps those
# pages at a different device VA, and the process aborts with "Memory access
# fault by GPU node-N". The fix is in patches/sglang_rocm/; confirm it is in your image.

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
