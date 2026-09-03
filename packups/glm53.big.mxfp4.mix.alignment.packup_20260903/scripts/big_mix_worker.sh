#!/bin/bash
# GLM-5.3 "big" MIX (aggregated) worker — prefill + decode colocated on the SAME
# GPUs. No PD, no mooncake, no RDMA. Runs through the infera wrapper
# (`python3 -m infera.engine.sglang`) so infera discovery + kv-aware routing work.
#
# Serves BOTH big-family checkpoints, selected by VARIANT:
#   VARIANT=fp8    -> /apps/data/models/GLM-5.3        (704 GB, fp8 e4m3 blockwise)
#   VARIANT=mxfp4  -> /apps/data/models/GLM-5.3-MXFP4  (408 GB, Quark MXFP4)
# Both are model_type glm_moe_dsa / GlmMoeDsaForCausalLM, field-for-field identical
# to GLM-5.2 except transformers_version, so stock sglang serves them through
# glm4_moe.py. No engine source overlay, no new Dockerfile.
#
# NOTHING HERE HAS BEEN RUN. Every value below is sourced, and RECIPE.md says from
# where. Where the vendor card and our GLM-5.2 baseline disagree, the disagreement
# is recorded rather than resolved -- see OPEN_QUESTIONS.md.
#
# DEFAULTS ARE GPUs 4-7 / TP4 so this arm coexists with the leader's arm on 0-3.
#
# Runs INSIDE the container. Safe under `set -u`.
set -u

# No apostrophe in any ${VAR:?...} message below: a `'` there opens a quote that
# silently swallows the rest of the script, and the failure surfaces dozens of
# lines later as an unrelated "unbound variable".
MY_IP="${MY_IP:?MY_IP=node IP (router and clients reach this)}"
ETCD_IP="${ETCD_IP:-$MY_IP}"

VARIANT="${VARIANT:-mxfp4}"
case "$VARIANT" in
  fp8)   DEF_MODEL=/apps/data/models/GLM-5.3;       DEF_SERVED=glm-5.3-fp8;   DEF_QUANT=none;  DEF_MOE=auto;  DEF_SHFUSE=1 ;;
  mxfp4) DEF_MODEL=/apps/data/models/GLM-5.3-MXFP4; DEF_SERVED=glm-5.3-mxfp4; DEF_QUANT=quark; DEF_MOE=aiter; DEF_SHFUSE=0 ;;
  *) echo "VARIANT must be fp8 or mxfp4, got: $VARIANT" >&2; exit 2 ;;
esac

MODEL="${MODEL:-$DEF_MODEL}"
SERVED="${SERVED:-$DEF_SERVED}"
PORT="${PORT:-30010}"          # NOT 30000 -- that is the leader arm.
ETCD_PORT="${ETCD_PORT:-22379}" # NOT 2379 (foreign etcd), NOT 12379 (leader).

# TP4 is what the vendor card validated for the MXFP4 checkpoint (4x MI350). It
# also leaves four GPUs for the other arm. Default to the UPPER half so this arm
# does not collide with the leader on 0-3.
TP="${TP:-4}"
GPUS="${GPUS:-4,5,6,7}"

# --- weights / quantisation --------------------------------------------------
# fp8: config.json already carries quantization_config.quant_method=fp8, so an
# explicit flag is not needed and passing one risks overriding the blockwise
# [128,128] scheme. mxfp4: the card says the loader auto-detects quark, but the
# Flash recipe passes it explicitly and an explicit value cannot be misdetected.
# QUANT=none means: emit no --quantization flag, let the loader read config.json.
QUANT="${QUANT:-$DEF_QUANT}"
# aiter = native AITER MXFP4 MoE kernels (torch.float4_e2m1fn_x2). With `triton`
# the checkpoint is dequantised to BF16 GEMMs -- still serves, much slower.
# The vendor card for THIS checkpoint says `auto`; we pin aiter for mxfp4 because
# the sibling Flash card pins it and `auto` is not a value you can diff against.
MOE_RUNNER="${MOE_RUNNER:-$DEF_MOE}"
KV_DTYPE="${KV_DTYPE:-fp8_e4m3}"   # vendor card AND GLM-5.2 mix baseline agree.

# --- shape -------------------------------------------------------------------
# CTX 262144 matches the GLM-5.2 MIX fixlen baseline we must land near. The vendor
# card uses 1048576, but that is coupled to its concurrency-2 configuration.
CTX="${CTX:-262144}"
GMU="${GMU:-0.80}"                 # vendor card and GLM-5.2 mix baseline agree.
MAX_RUNNING="${MAX_RUNNING:-32}"   # NOT the vendor card 2 -- see RECIPE.md.
# GLOBAL budget that SGLang divides by dp_size ONLY when DP-attention is on. One
# value serves both modes; do NOT hardcode a per-rank number in a DPA-off branch.
CHUNK="${CHUNK:-65536}"
MAX_PREFILL="${MAX_PREFILL:-16384}"
CUDA_GRAPH_BS="${CUDA_GRAPH_BS:-32}"  # NOT the vendor card 2 -- see RECIPE.md.

# --- switches ----------------------------------------------------------------
DPA="${DPA:-0}"          # DP-attention. Off in round 1: the vendor card does not use it.
MTP="${MTP:-0}"          # EAGLE speculative decoding. OFF -- see RECIPE.md / OPEN_QUESTIONS.md.
HICACHE="${HICACHE:-0}"  # Hierarchical cache. OFF -- unverified on this image build.
KVAWARE="${KVAWARE:-1}"
CUSTOM_AR="${CUSTOM_AR:-0}"
# Shared-experts fusion. 1 = leave the engine default (fused). 0 = pass
# --disable-shared-experts-fusion. Default 0 on the mxfp4 arm; see below.
SHARED_FUSION="${SHARED_FUSION:-$DEF_SHFUSE}"
# Vendor-card-only env block (MXFP4 arm). Separable so it can be A/B-ed as ONE
# variable; it conflicts with the GLM-5.2 block in two places (see below).
VENDOR_ENV="${VENDOR_ENV:-0}"

KV_PUB_PORT="${KV_PUB_PORT:-5567}"   # NOT 5557 -- leader arm.
KV_SNAP_PORT="${KV_SNAP_PORT:-8811}" # NOT 8801 -- leader arm.
LOG="${LOG:-/tmp/glm53_big_mix.log}"
mkdir -p "$(dirname "$LOG")"

# ---- GLM-5.2 DSA-on-ROCm recipe (MANDATORY on gfx950) -----------------------
# Carried VERBATIM from fixlen.glm52.mix.packup_20260806/scripts/mix_engine.sh.
# Without these the model still serves and still returns 200s -- it just returns
# garbage, because the sparse-attention indexer takes a path not ported to ROCm.
# Garbage or repeated tokens from a completion is therefore NOT a sampling
# problem; it is the signature of this block not taking effect.
export SGLANG_USE_AITER=1
export SGLANG_ROCM_FUSED_DECODE_MLA="${FUSED_DECODE_MLA:-0}"
export SGLANG_OPT_USE_TILELANG_INDEXER=1
export SGLANG_OPT_USE_TOPK_V2=0
export SGLANG_OPT_USE_JIT_NORM=0
export SAFETENSORS_FAST_GPU=1
export HIP_FORCE_DEV_KERNARG=1
# Stable block hashes -> stable kv-aware keys across restarts.
export PYTHONHASHSEED=0
export NCCL_IB_DISABLE=1 NCCL_IGNORE_CPU_AFFINITY=1 HSA_NO_SCRATCH_RECLAIM=1
[ "$DPA" = "1" ] && export SGLANG_DP_USE_GATHERV=1

# ---- vendor-card env, opt-in ------------------------------------------------
# TWO of these CONFLICT with the block above and are the reason this is opt-in:
#   * SGLANG_ROCM_FUSED_DECODE_MLA: card=1, GLM-5.2 block=0. Set FUSED_DECODE_MLA
#     explicitly rather than letting VENDOR_ENV decide it for you.
#   * SGLANG_SET_CPU_AFFINITY=1 vs NCCL_IGNORE_CPU_AFFINITY=1 above -- opposite
#     intents about who pins threads.
if [ "$VENDOR_ENV" = "1" ]; then
  export SGLANG_SET_CPU_AFFINITY=1 SGLANG_USE_ROCM700A=1 SGLANG_MOE_PADDING=1
  export SGLANG_ROCM_DISABLE_LINEARQUANT=0
  export NCCL_MIN_NCHANNELS=112 ROCM_QUICK_REDUCE_QUANTIZATION=INT8
fi

NIC=$(ip -o -4 addr show | awk -v ip="$MY_IP" '$4 ~ ("^" ip "/") {print $2; exit}')
[ -n "$NIC" ] && export SGLANG_LOCAL_IP_NIC="$NIC" GLOO_SOCKET_IFNAME="$NIC"
export SGLANG_HOST_IP="$MY_IP" HOST_IP="$MY_IP"
export INFERA_SGLANG_READY_TIMEOUT="${READY_TIMEOUT:-3600}"

# ---- args -------------------------------------------------------------------
# --ep-size is emitted UNCONDITIONALLY, OUTSIDE the DPA branch. Expert
# parallelism and attention parallelism are different axes; gating both on one
# condition means turning DPA off also collapses the MoE from ep4 to the TP
# default, and then no latency delta is attributable to either.
DP_ARGS=(--ep-size "${EP_SIZE:-$TP}")
if [ "$DPA" = "1" ]; then
  DP_ARGS+=(--dp-size "$TP" --enable-dp-attention)
fi

QUANT_ARGS=(); [ "$QUANT" = "none" ] || QUANT_ARGS=(--quantization "$QUANT")

# EAGLE MTP. Default OFF. The SGLang GLM-5.3 cookbook disables MTP/EAGLE on AMD
# because the gfx950 draft kernel is unvalidated -- yet the vendor card for THIS
# checkpoint runs EAGLE at 3 steps. Unresolved; see OPEN_QUESTIONS.md. Turning it
# on is one env var, so it stays a single-variable round.
MTP_ARGS=()
if [ "$MTP" = "1" ]; then
  MTP_ARGS=(--speculative-algorithm EAGLE
            --speculative-num-steps "${SPEC_STEPS:-3}"
            --speculative-eagle-topk "${SPEC_TOPK:-1}"
            --speculative-num-draft-tokens "${SPEC_DRAFT:-4}")
fi

# Hierarchical cache. The vendor card enables it; --page-size 64 is a REQUIREMENT
# of its page_first_direct layout, not an independent knob, so the two move
# together. Off by default: only patch_hicache_rocm_host_alloc.py is in this image
# build and the staged-write-back gate was deliberately excluded, so hicache is
# unverified here. Re-derive that gate before turning this on.
HICACHE_ARGS=()
if [ "$HICACHE" = "1" ]; then
  HICACHE_ARGS=(--enable-hierarchical-cache --hicache-ratio "${HICACHE_RATIO:-1.5}"
                --hicache-write-policy write_through --hicache-io-backend direct
                --hicache-mem-layout page_first_direct --page-size "${PAGE_SIZE:-64}")
fi

# The aiter custom all-reduce kernel deadlocks on this architecture during
# speculative verify. Disabled INDEPENDENTLY of MTP: letting the switch follow MTP
# would make any MTP on/off comparison a two-variable one.
CAR_ARGS=(); [ "$CUSTOM_AR" = "1" ] || CAR_ARGS+=(--disable-custom-all-reduce)

# Shared-experts fusion. The sibling Flash MXFP4 arm faults here: a BF16 SHARED
# expert gets renamed into routed slot n_routed_experts and then copied into an
# MXFP4-packed routed param, dying in
# fused_moe_triton/layer.py::_load_w2 -> expert_data.copy_(loaded_weight).
# --disable-shared-experts-fusion is the confirmed fix there.
#
# READ BEFORE CHANGING THIS DEFAULT -- the two halves point opposite ways:
#  * The GUARD IS ABSENT here. glm4_moe.py:1174 overrides
#    shared_experts_fusion_disable_reason WITHOUT calling super() and WITHOUT
#    calling quant_blocks_shared_experts_fusion(). It special-cases exactly one
#    quant method (w4afp8) and returns None -- fuse -- for quark. deepseek_v2.py
#    :3060 does consult it. So glm4_moe is on the same unguarded side as the
#    glm5_next override that produced the Flash fault.
#  * The PRECONDITION IS ABSENT. In this checkpoint the shared experts are
#    themselves MXFP4: mlp.shared_experts.{down,gate,up}_proj has 76 .weight and
#    75 .weight_scale, and the single missing scale is layer 78 (the BF16 MTP
#    layer, not loaded when MTP=0). Layers 3-77 fuse a quantized shared expert
#    into quantized routed slots -- same precision, legitimate.
# So: no landmine is expected, but nothing in the engine would catch one. Round 1
# takes the insurance; SHARED_FUSION=1 turns fusion back on as its own single
# variable, which is also the arm that matches the vendor card (it passes no such
# flag). Expect a perf delta between the two -- fusion is an optimisation.
SHFUSE_ARGS=(); [ "$SHARED_FUSION" = "1" ] || SHFUSE_ARGS+=(--disable-shared-experts-fusion)

INFERA_ARGS=(--advertise-host "$MY_IP" --etcd-endpoint "$ETCD_IP:$ETCD_PORT"
             --discovery-backend etcd --request-transport http --kv-event-transport zmq)
if [ "$KVAWARE" = "1" ]; then
  INFERA_ARGS+=(--kv-events-bind "tcp://0.0.0.0:$KV_PUB_PORT" --kv-snapshot-port "$KV_SNAP_PORT")
else
  INFERA_ARGS+=(--no-enable-kv-events)
fi

# The chat template ships inside both checkpoints; the vendor card passes it
# explicitly. Pass it only if it is actually there, so a checkpoint without one
# falls back to the tokenizer config instead of dying on a missing file.
TMPL_ARGS=()
[ -f "$MODEL/chat_template.jinja" ] && TMPL_ARGS=(--chat-template "$MODEL/chat_template.jinja")

echo "[glm53-big-mix] variant=$VARIANT ip=$MY_IP:$PORT nic=${NIC:-?} tp=$TP gpus=$GPUS quant=$QUANT moe=$MOE_RUNNER kv=$KV_DTYPE dpa=$DPA mtp=$MTP shfuse=$SHARED_FUSION hicache=$HICACHE vendorenv=$VENDOR_ENV gmu=$GMU ctx=$CTX chunk=$CHUNK maxrun=$MAX_RUNNING graphbs=$CUDA_GRAPH_BS kvaware=$KVAWARE -> $LOG"

HIP_VISIBLE_DEVICES="$GPUS" python3 -m infera.engine.sglang \
  --model-path "$MODEL" --served-model-name "$SERVED" --tp-size "$TP" --trust-remote-code \
  --host "$MY_IP" --port "$PORT" \
  --dsa-prefill-backend tilelang --dsa-decode-backend tilelang \
  --kv-cache-dtype "$KV_DTYPE" --moe-runner-backend "$MOE_RUNNER" \
  --mem-fraction-static "$GMU" --context-length "$CTX" \
  --max-running-requests "$MAX_RUNNING" --cuda-graph-max-bs "$CUDA_GRAPH_BS" \
  --chunked-prefill-size "$CHUNK" --max-prefill-tokens "$MAX_PREFILL" \
  --watchdog-timeout "${WATCHDOG:-3600}" \
  --reasoning-parser glm45 --tool-call-parser glm47 \
  --mm-feature-transport cpu --enable-cache-report \
  "${QUANT_ARGS[@]}" "${DP_ARGS[@]}" "${MTP_ARGS[@]}" "${HICACHE_ARGS[@]}" \
  "${CAR_ARGS[@]}" "${SHFUSE_ARGS[@]}" "${TMPL_ARGS[@]}" "${INFERA_ARGS[@]}" > "$LOG" 2>&1
