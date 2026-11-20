#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# what: launch ONE DeepSeek-V4.1-Flash PD leg (prefill|decode) through the infera vLLM wrapper.
# why : this is the real launcher. It carries the recipe — InferenceX's top MI355X vLLM point
#       (dsv41flash/vllm/mi355x-fp4-mtp/agentic.yaml, override_dp2_umbp_c384) split into a prefill
#       and a decode engine — and NOTHING site-specific: addresses, GPUs, NICs and the RDMA
#       registration mode come in as env vars from cluster/<your-cluster>.sh.
# how : runs ON the node, docker-execs into $CTR. Called by engine/up.sh and engine/golden.sh.
#
# Switches:
#   ROLE=prefill|decode
#   SPEC_MODE=block|golden  DSpark verification. block = real rejection (correctness runs; always on
#                           prefill). golden = vLLM synthetic acceptance at GOLDEN_AL (3.51, InferenceX's
#                           thinking_on K=5 value) — the AgentX scoring contract, decode leg only.
#   UMBP=0|1                prefill: add UMBPStoreConnector (host-DRAM KV tier) next to Mooncake.
#   MR_MODE=gem|peermem     how Mooncake registers VRAM; set by the wrapper, see cluster/README.md.
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$DIR/../common.sh"

ROLE="${ROLE:?ROLE=prefill|decode}"
require_env MY_IP "this node's data-plane IP (the one peers and the router can reach)"
require_env ETCD_IP "the node running etcd (by convention the prefill node)"
require_env MODEL "model directory, inside \$MODEL_MOUNT"
require_env GPUS "comma-separated GPU indices for this leg (DP ranks = number of GPUs)"
require_env MR_MODE "gem | peermem — read it off preflight_rdma.sh"
require_env RDMA_IB_DEVICES "comma-separated RDMA devices Mooncake may use, e.g. ionic_0,...,ionic_7"
require_env MC_GID_INDEX "the routable RoCE v2 GID index"

if [ "$ROLE" = "prefill" ]; then
  PORT="${PORT:-$PREFILL_PORT}"; LEG=0; KV_ROLE=kv_producer; SPEC_MODE=block
else
  PORT="${PORT:-$DECODE_PORT}"; LEG=1; KV_ROLE=kv_consumer; SPEC_MODE="${SPEC_MODE:-block}"
fi
DP=$(( $(tr -cd , <<<"$GPUS" | wc -c) + 1 ))
LOG="$LOG_DIR/$ROLE.log"

# ---- recipe: shared by both legs, verbatim from the c384 point ----------------------------
# DP2 x TP1 per leg: attention data-parallel, the MoE tensor-parallel across the DP ranks (no EP).
# fp8_e4m3 = per-tensor fp8 KV rows read by the AITER sparse MLA kernel (vllm#66036) instead of the
# block-scaled fp8_ds_mla record; it is ALSO the KV transfer format, so P and D must agree.
# Engram tables live in host memory, as at the c384 point; no --monokernel (the DP2 points were
# tuned without it). Block size 128 is what the indexer and SparseMLA prefer.
if [ "$SPEC_MODE" = "golden" ]; then
  SPEC="{\"method\":\"dspark\",\"num_speculative_tokens\":5,\"draft_sample_method\":\"probabilistic\",\"rejection_sample_method\":\"synthetic\",\"synthetic_acceptance_length\":${GOLDEN_AL:-3.51},\"enable_adaptive_verification\":false}"
else
  SPEC='{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","rejection_sample_method":"block","enable_adaptive_verification":false}'
fi
ARGS=(--model "$MODEL" --served-model-name "$SERVED" --language-model-only
      --tokenizer-mode deepseek_v41 --tool-call-parser deepseek_v41 --enable-auto-tool-choice
      --reasoning-parser deepseek_v41 --moe-backend aiter
      --gpu-memory-utilization "${GMU:-0.9}"
      --attention-config '{"indexer_kv_dtype":"mxfp4","indexer_sparse_logits":true}'
      --block-size 128 --speculative-config "$SPEC" --max-model-len 1048576
      --disable-uvicorn-access-log
      --tensor-parallel-size 1 --data-parallel-size "$DP" --data-parallel-rpc-port $((29610 + 2 * LEG))
      --engram-config '{"cpu_offload":true}'
      --max-cudagraph-capture-size 1024 --max-num-batched-tokens 16384
      --kv-cache-dtype fp8_e4m3)

# ---- KV transfer ---------------------------------------------------------------------------
# GEM mode: the KV cache is allocated from amdgpu GEM BOs (charged once by KFD) and registered by
# dma-buf; peer memory must never touch those BOs. peermem mode: bare ibv_reg_mr through the
# peer-memory module, dma-buf off. See cluster/README.md for which one your nodes need.
MC_ENV=(MC_GID_INDEX="$MC_GID_INDEX" MC_DISABLE_HIP=1 MC_MAX_MR_SIZE=1073741824 MC_RDMA_NOTIFY_ENABLED=0
        VLLM_MOONCAKE_BOOTSTRAP_PORT=$((28390 + LEG)) VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT=1800)
mc_mod=""
case "$MR_MODE" in
  gem)
    GEM_SO=/usr/local/lib/libhip_gem_alloc.so
    MC_ENV+=(INFERA_AMD_GEM_ALLOCATOR="$GEM_SO" LD_PRELOAD="$GEM_SO" MOONCAKE_DISABLE_HIP_DMABUF=0 MC_HIP_KFD_PIN_FIX=0)
    mc_mod=',"kv_connector_module_path":"infera.engine.vllm.gem_mooncake"' ;;
  peermem) MC_ENV+=(MOONCAKE_DISABLE_HIP_DMABUF=1) ;;
  *) die "MR_MODE must be gem or peermem, got '$MR_MODE'" ;;
esac
# One NIC per GPU (its PCIe neighbour) when the site names them. Left unset, infera pins every GPU
# to ONE NIC whenever the NICs span several subnets, and all KV then crosses that single NIC.
if [ -n "${RDMA_GPU_NICS:-}" ]; then
  topo="$LOG_DIR/mc-topo-$ROLE.json"
  python3 - "$GPUS" "$RDMA_GPU_NICS" "$topo" <<'PY'
import json, sys
gpus = [int(g) for g in sys.argv[1].split(",")]
nics = sys.argv[2].split(",")
half = len(nics) // 2
t = {f"cuda:{i}": [[nics[g]], []] for i, g in enumerate(gpus)}
t.update({"cpu:0": [nics[:half], []], "cpu:1": [nics[half:], []]})
json.dump(t, open(sys.argv[3], "w"))
PY
  MC_ENV+=(MC_CUSTOM_TOPO_JSON="$topo")
fi
MOONCAKE="{\"kv_connector\":\"MooncakeConnector\"$mc_mod,\"kv_role\":\"$KV_ROLE\",\"kv_connector_extra_config\":{\"num_workers\":10,\"device_name\":\"$RDMA_IB_DEVICES\"}}"

ROLE_ENV=()
INFERA_ARGS=(--discovery-backend etcd --etcd-endpoint "$ETCD_IP:$ETCD_PORT"
             --request-transport http --kv-event-transport zmq --advertise-host "$MY_IP")
if [ "$ROLE" = "prefill" ]; then
  # Prefix caching on (vLLM default) with KV events, so the router steers each session to the
  # rank holding its prefix. No --prefill-schedule-interval: it admits prefills only every N steps,
  # which on a prefill-only engine idles N-1 of every N steps.
  if [ "${UMBP:-1}" = "1" ]; then
    UMBPC="{\"kv_connector\":\"UMBPStoreConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"mode\":\"standalone\",\"endpoint\":\"$UMBP_SOCK\"}}"
    KVT="{\"kv_connector\":\"MultiConnector\",\"kv_role\":\"kv_producer\",\"kv_load_failure_policy\":\"recompute\",\"kv_connector_extra_config\":{\"connectors\":[$MOONCAKE,$UMBPC]}}"
    ROLE_ENV+=(UMBP_STANDALONE_ADDRESS="$UMBP_SOCK")
  else
    KVT="$MOONCAKE"
  fi
  ARGS+=(--kv-transfer-config "$KVT")
else
  # No prefix cache on decode (every turn's KV arrives from prefill), so no KV events, and vLLM's
  # own DP balancer picks the rank by live load (INFERA_VLLM_DP_ENGINE_LB=1). The prefill cadence of
  # the c384 point is kept: on decode it times the admission of requests whose KV has landed.
  ARGS+=(--kv-transfer-config "$MOONCAKE" --no-enable-prefix-caching
         --prefill-schedule-interval "${D_PSI:-11}")
  INFERA_ARGS+=(--no-enable-kv-events --kv-events off)
  ROLE_ENV+=(INFERA_VLLM_DP_ENGINE_LB=1)
fi
# EXTRA_ENGINE_ARGS: the wrapper's escape hatch for vLLM flags this script does not model.
# Unquoted on purpose so several flags in one string word-split.
# shellcheck disable=SC2206
[ -n "${EXTRA_ENGINE_ARGS:-}" ] && ARGS+=($EXTRA_ENGINE_ARGS)

# ---- env -----------------------------------------------------------------------------------
# AITER and graph settings are the c384 point's. The last block is stability, not performance:
# uvicorn's 5 s keep-alive races the Rust router's idle pool (502 on a reused connection); long
# agentic prompts overrun the 30 s chat-template render timeout; VLLM_SERVER_DEV_MODE exposes the
# reset_prefix_cache endpoint the router's kv-aware policy calls at start.
ENGINE_ENV=(HIP_VISIBLE_DEVICES="$GPUS" CUDA_VISIBLE_DEVICES="$GPUS"
            VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_MOE=1 VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=NONE
            AITER_TRITON_LOG_LEVEL=ERROR VLLM_ROCM_USE_AITER_TRITON_SPARSE_MLA=True
            VLLM_USE_BREAKABLE_CUDAGRAPH=1 OMP_NUM_THREADS=1 VLLM_USE_RUST_FRONTEND=1 PYTHONUNBUFFERED=1
            VLLM_HOST_IP="$MY_IP" VLLM_PORT=$((40000 + 2000 * LEG)) VLLM_DP_MASTER_PORT=$((29511 + 2 * LEG))
            VLLM_ENGINE_READY_TIMEOUT_S="${READY_TIMEOUT:-3600}" INFERA_ENGINE_READY_TIMEOUT="${READY_TIMEOUT:-3600}"
            PYTHONHASHSEED=0
            VLLM_HTTP_TIMEOUT_KEEP_ALIVE=120 VLLM_CHAT_TEMPLATE_RENDER_TIMEOUT=600 PYTHONFAULTHANDLER=1
            VLLM_SERVER_DEV_MODE=1)

{ echo "# $(date -u +%FT%TZ) $ROLE leg: GPUs $GPUS (DP$DP) port $PORT spec=$SPEC_MODE umbp=${UMBP:-1} mr=$MR_MODE"
  printf '%q ' env "${ENGINE_ENV[@]}" "${MC_ENV[@]}" "${ROLE_ENV[@]}" python3 -m infera.engine.vllm \
    "${INFERA_ARGS[@]}" --host 0.0.0.0 --port "$PORT" "${ARGS[@]}"; echo; } > "$LOG_DIR/$ROLE.cmd"
log "$ROLE on $MY_IP:$PORT — GPUs $GPUS (DP$DP) spec=$SPEC_MODE umbp=${UMBP:-1} mr=$MR_MODE -> $LOG"
docker exec -d "$CTR" bash -c "$(tail -1 "$LOG_DIR/$ROLE.cmd") > '$LOG' 2>&1"
