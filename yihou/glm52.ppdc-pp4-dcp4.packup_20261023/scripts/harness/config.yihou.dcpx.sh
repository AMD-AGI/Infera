#!/usr/bin/env bash
# config.yihou.dcpx.sh (installed 2026-10-22 in $H)
# Overlay (dcpx, 2026-10-22): our best g3x1 (cross-node 1P1D P4D4, prefill 135 GPU0-3 -> decode 136 GPU0-3, Mooncake
# RDMA rail i <-> rail i) with ONLY the decode leg replaced by the DCP4 decode of dcp_pd_20261021 round pdBeperf-r5b:
#   decode TP4 EP4 DP1 (no DP attention) + --dcp-size 4 + KEEP_KV_B_BF16 + split(-triton) decode indexer.
# Unavoidable side effect on the prefill leg: BOTH legs run image infera-sglang:v0522-1020-yihou-dcp-mc4340 with the
# sglang overlay 93034eba91 mounted over /sgl-workspace/sglang/python/sglang (the overlay carries the PD/DCP fixes the
# prefill side needs: patches 0010-0013, 0017, 0019). Prefill ARGS stay g3x1 (TP4/DP4 DPA, OffloaderV2 G3K1 step1,
# GTT HiCache 260 GB, mem 0.90, chunk 32768). Our v0521 single-file mounts are DROPPED (each is in the overlay or v0.5.22:
# noblock = patch 0008, noabsorbcfg = 0009, nonfinal event = 0004, offloader is_shuffled = 0021, PR42553 GTT = overlay
# pool_host/common.py:250-289, idlefix-equivalent = overlay eager_runner.py:445-461; xferfix2 conn = log-only, lost).
# Chain: DCPX_BASE_CONFIG (default config.yihou.xnode.sh -> XNODE_BASE_CONFIG = the g3x1 idlefix chain).
# Placed OUTSIDE xnode on purpose: xnode.sh:34/37 hard-check RDMA_DEVICE == per-GPU map and PD_DP_RANK_AFFINITY == 1,
# both of which DCP4 must change; they are changed here, after xnode's checks ran on the unchanged chain.
_dx_base="${DCPX_BASE_CONFIG:-$(dirname "${BASH_SOURCE[0]}")/config.yihou.xnode.sh}"
[[ -r "$_dx_base" ]] || { echo "config.yihou.dcpx.sh: base not readable: $_dx_base" >&2; return 2 2>/dev/null || exit 2; }
# An exported PD_DP_RANK_AFFINITY=0 from a parent that already sourced this file (run.v3x dryrun runs engine.sh locally
# with the parent's exported env) would trip xnode's "must be 1" check: clear it, xnode sees its default, set 0 below.
[[ "${PD_DP_RANK_AFFINITY:-}" == 0 ]] && unset PD_DP_RANK_AFFINITY
source "$_dx_base"

: "${DCPX_DCP:=4}"
: "${DCPX_OVERLAY:=/shared_nfs/yihou_glm52/dcp_pd_20261021/src/sglang-93034eba91/python/sglang}"
: "${DCPX_DECODE_RDMA:=list}"      # list = shared "ionic_0,...,ionic_3" | map4 = per-GPU JSON, all four per GPU
: "${DCPX_SPLIT:=1}"                # 1 = SGLANG_DSA_DCP_SPLIT_DECODE_INDEXER=1 + _TRITON=1 (r5b); 0 = A/B off
: "${DCPX_SEQLOG:=0}"               # 1 = SGLANG_LOG_DECODE_SEQ_LENS=1 (overlay diagnostic, patch 0025)
: "${DCPX_SDMA0:=0}"                # 1 = HSA_ENABLE_SDMA=0 both legs (needed only on amdgpu 6.16.x nodes)
: "${DCPX_RAIL:=affinity}"           # affinity (dcpx1/2) | destlocal (needs the -destpin image) | hip (same-node: base HIP IPC transport)
[[ "$DCPX_DCP" =~ ^[02-9]$ ]] || { echo "config.yihou.dcpx.sh: DCPX_DCP must be 0 or 2..9" >&2; return 2 2>/dev/null || exit 2; }
[[ -f "$DCPX_OVERLAY/__init__.py" && "$DCPX_OVERLAY" == */python/sglang ]] ||
    { echo "config.yihou.dcpx.sh: bad DCPX_OVERLAY=$DCPX_OVERLAY" >&2; return 2 2>/dev/null || exit 2; }

# --- image: both legs (engine.sh uses one $IMAGE). IDs must be read on EACH node after `docker load` of
# /shared_nfs/yihou_glm52/dcp_pd_20261021/pd/images/infera-sglang_v0522-1020-yihou-dcp-mc4340.tar:
# the kit's nodes show sha256:5b6565ef... (image_ids.dcp.tsv), but on 135/136 a load re-IDed the v0521 tar
# (mc4340n19:2,47), so do not assume. The AITER JIT cache dir is keyed by this ID (engine.sh: $AITER_JIT_CACHE_ROOT/<id>).
IMAGE="${DCPX_IMAGE:-infera-sglang:v0522-1020-yihou-dcp-mc4340}"
[[ "$DCPX_RAIL" != destlocal || "$IMAGE" == *-destpin ]] || { echo "config.yihou.dcpx.sh: DCPX_RAIL=destlocal needs the -destpin image" >&2; return 2 2>/dev/null || exit 2; }
case "$(hostname -s)" in
    "${XNODE_DECODE_NODE:-yihou-no-decode-node}") EXPECTED_IMAGE_ID="${DCPX_DECODE_IMAGE_ID:-}" ;;
    *)                    EXPECTED_IMAGE_ID="${DCPX_PREFILL_IMAGE_ID:-}" ;;
esac
export IMAGE EXPECTED_IMAGE_ID

# --- mounts: overlay only (one row; file created once in $WS/harness, same format as the kit's tsv) ---------------
YIHOU_MOUNTS="$WS/harness/mounts.yihou.dcpx-overlay.$(basename "$(dirname "$(dirname "$DCPX_OVERLAY")")").tsv"
[[ "$(cat "$YIHOU_MOUNTS" 2>/dev/null)" == "$(printf '%s\t%s' "$DCPX_OVERLAY" /sgl-workspace/sglang/python/sglang)" ]] ||
    { echo "config.yihou.dcpx.sh: $YIHOU_MOUNTS must hold exactly the overlay row" >&2; return 2 2>/dev/null || exit 2; }
export PREFILL_MOUNTS_ADD=""        # offloader.yihou-attrs.py (v0521) must not shadow the overlay's offloader.py
export DECODE_MOUNTS_ADD=""
# DCPX_SHARD=room (dcpx7): prefill-only mount of mooncake/conn.py that shards the Mooncake transfer queue by
# bootstrap_room for DCP destinations (overlay shards by sum(dst ports) -> constant for a DP1 DCP decode -> one worker).
: "${DCPX_SHARD:=0}"
case "$DCPX_SHARD" in
    0) ;;
    room) export PREFILL_MOUNTS_ADD="$WS/harness/mounts.yihou.dcpx-shard.tsv" ;;
    hash) export PREFILL_MOUNTS_ADD="$WS/harness/mounts.yihou.dcpx-shard2.tsv" ;;   # dcpx7b: hashed room (room % 4 = prefill DP rank)
    xstats) export PREFILL_MOUNTS_ADD="$WS/harness/mounts.yihou.dcpx-xstats.tsv" ;;   # dcpx8: hash shard + log-only [yihou-xfer] stats
    *) echo "config.yihou.dcpx.sh: DCPX_SHARD must be 0|room|hash|xstats" >&2; return 2 2>/dev/null || exit 2 ;;
esac

# --- prefill: args unchanged; drop the xferfix2 env (read only by the dropped conn.yihou-xferfix2.py) --------------
_dx_pe=""; for _i in $PREFILL_EXTRA_ENV; do [[ "$_i" == YIHOU_XFER_* || "$_i" == YIHOU_NONFINAL_EVENT=* ]] || _dx_pe+=" $_i"; done
PREFILL_EXTRA_ENV="${_dx_pe# }"
[[ "$DCPX_SHARD" == room || "$DCPX_SHARD" == hash || "$DCPX_SHARD" == xstats ]] && PREFILL_EXTRA_ENV+=" YIHOU_DCP_SHARD=room"
[[ "$DCPX_SHARD" == xstats ]] && PREFILL_EXTRA_ENV+=" YIHOU_XFER_STATS=1"
# DCPX_PACKNOWAIT=1 (dcpx9): the DCP pack gather no longer waits on / copies indices through the device default stream
# (dcp_pack.yihou-nowait.py) + pack sub-timers in the [yihou-xfer] stats. Needs DCPX_SHARD=xstats.
: "${DCPX_PACKNOWAIT:=0}"
if [[ "$DCPX_PACKNOWAIT" == 1 ]]; then
    [[ "$DCPX_SHARD" == xstats ]] || { echo "config.yihou.dcpx.sh: DCPX_PACKNOWAIT=1 needs DCPX_SHARD=xstats" >&2; return 2 2>/dev/null || exit 2; }
    export PREFILL_MOUNTS_ADD="$WS/harness/mounts.yihou.dcpx-xstats2-packnowait.tsv"
    PREFILL_EXTRA_ENV+=" YIHOU_PACK_NOWAIT=1"
elif [[ "$DCPX_PACKNOWAIT" == 2 ]]; then   # dcpx10: + cached device metadata + priority -1 gather stream
    [[ "$DCPX_SHARD" == xstats ]] || { echo "config.yihou.dcpx.sh: DCPX_PACKNOWAIT=2 needs DCPX_SHARD=xstats" >&2; return 2 2>/dev/null || exit 2; }
    export PREFILL_MOUNTS_ADD="$WS/harness/mounts.yihou.dcpx-xstats2-packnowait2.tsv"
    PREFILL_EXTRA_ENV+=" YIHOU_PACK_NOWAIT=2"
elif [[ "$DCPX_PACKNOWAIT" != 0 ]]; then echo "config.yihou.dcpx.sh: DCPX_PACKNOWAIT must be 0|1|2" >&2; return 2 2>/dev/null || exit 2; fi
# DCPX_PMTP=1 (dcpx8, user 10-22): prefill carries the decode's EAGLE flags (llying 2p1d-v0522 example-prefill-mtp.diff):
# without the NextN layer on the prefill the transfer carries no draft KV and the decode draft layer reads uninitialised KV.
: "${DCPX_PMTP:=0}"
if [[ "$DCPX_PMTP" == 1 ]]; then
    export PREFILL_MTP=1                    # read by bench-harness/tools/agentx_env.py
    [[ " $PREFILL_EXTRA_ARGS " == *" --speculative-algorithm "* ]] ||
        PREFILL_EXTRA_ARGS+="${PREFILL_EXTRA_ARGS:+ }--speculative-algorithm EAGLE --speculative-num-steps $DECODE_SPEC_STEPS --speculative-eagle-topk $DECODE_SPEC_TOPK --speculative-num-draft-tokens $DECODE_SPEC_DRAFT_TOKENS"
elif [[ "$DCPX_PMTP" != 0 ]]; then echo "config.yihou.dcpx.sh: DCPX_PMTP must be 0|1" >&2; return 2 2>/dev/null || exit 2; fi

# DCPX_DCP=0 (dpax1, 10-22): keep the base DPA4 decode leg (only image/overlay/mounts/rails of dcpx), the DPA
# control for the DCP-vs-DPA transfer comparison.
if [[ "$DCPX_DCP" != 0 ]]; then
# --- decode: the DCP4 leg (r5b) --------------------------------------------------------------------------------------
DECODE_TP=4; DECODE_EP=4; DECODE_DP=1; DECODE_DPA=0        # DCP needs attn_tp % dcp == 0 -> no DPA
PD_DP_RANK_AFFINITY=0                                      # router: decode has one DP rank (affinity would 503 ranks 1-3)
[[ " $DECODE_EXTRA_ARGS " == *" --dcp-size "* ]] || DECODE_EXTRA_ARGS+=" --dcp-size $DCPX_DCP"
[[ " $DECODE_EXTRA_ARGS " == *" --disable-custom-all-reduce "* ]] &&
    { echo "config.yihou.dcpx.sh: decode must run custom AR ON (r5b)" >&2; return 2 2>/dev/null || exit 2; }
DECODE_EXTRA_ENV+=" SGLANG_DSA_DCP_KEEP_KV_B_BF16=1"
[[ "$DCPX_SPLIT" == 1 ]] && DECODE_EXTRA_ENV+=" SGLANG_DSA_DCP_SPLIT_DECODE_INDEXER=1 SGLANG_DSA_DCP_SPLIT_INDEXER_TRITON=1"
[[ "$DCPX_SEQLOG" == 1 ]] && DECODE_EXTRA_ENV+=" SGLANG_LOG_DECODE_SEQ_LENS=1"
if [[ "$DCPX_SDMA0" == 1 ]]; then PREFILL_EXTRA_ENV+=" HSA_ENABLE_SDMA=0"; DECODE_EXTRA_ENV+=" HSA_ENABLE_SDMA=0"; fi
fi

# DCPX_GRAPH_BS (dcpx11, user 10-22): explicit decode cuda-graph capture list, comma separated (YIHOU_OVERRIDES is
# space separated); pair it with DECODE_GRAPH_MAX_BS = its last entry.
: "${DCPX_GRAPH_BS:=}"
if [[ -n "$DCPX_GRAPH_BS" ]]; then
    [[ "$DCPX_GRAPH_BS" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo "config.yihou.dcpx.sh: DCPX_GRAPH_BS must be a comma list of ints" >&2; return 2 2>/dev/null || exit 2; }
    [[ "${DCPX_GRAPH_BS##*,}" == "$DECODE_GRAPH_MAX_BS" ]] || { echo "config.yihou.dcpx.sh: DCPX_GRAPH_BS must end at DECODE_GRAPH_MAX_BS" >&2; return 2 2>/dev/null || exit 2; }
    [[ " $DECODE_EXTRA_ARGS " == *" --cuda-graph-bs-decode "* ]] || DECODE_EXTRA_ARGS+=" --cuda-graph-bs-decode ${DCPX_GRAPH_BS//,/ }"
fi

# --- ppdc (10-23): PP4 prefill + llying prefill options + GMU decode patch -------------------------------------------
# DCPX_PP=4: prefill TP1 x PP4 (one 4-GPU prefill instance, stage k = GPU k), layer partition 20,20,20,18, chunk and
# max-prefill-tokens 8192, overlap off (llying 2p1d-v0522 kit-v6-pp4.diff); mounts the deepseek_v2 PP fix (ab7a898b77)
# next to the conn/dcp_pack patches (needs DCPX_SHARD=xstats DCPX_PACKNOWAIT=2).
: "${DCPX_PP:=0}" "${DCPX_SPF:=0}" "${DCPX_NOOFFLOAD:=0}" "${DCPX_DECODE_EC:=0}"
if [[ "$DCPX_PP" != 0 ]]; then
    [[ "$DCPX_PP" =~ ^[2-8]$ ]] || { echo "config.yihou.dcpx.sh: DCPX_PP must be 0 or 2..8" >&2; return 2 2>/dev/null || exit 2; }
    [[ "$DCPX_SHARD" == xstats && "$DCPX_PACKNOWAIT" == 2 ]] || { echo "config.yihou.dcpx.sh: DCPX_PP needs DCPX_SHARD=xstats DCPX_PACKNOWAIT=2" >&2; return 2 2>/dev/null || exit 2; }
    PREFILL_TP=1; PREFILL_EP=1; PREFILL_DP=1; PREFILL_DPA=0; PREFILL_CHUNK_SIZE=8192
    PREFILL_EXTRA_ARGS+=" --pp-size $DCPX_PP --disable-overlap-schedule --max-prefill-tokens 8192"
    PREFILL_EXTRA_ENV+=" SGLANG_PP_LAYER_PARTITION=${DCPX_PP_PARTITION:-20,20,20,18}"
    export PREFILL_MOUNTS_ADD="$WS/harness/mounts.yihou.dcpx-pp-prefill.tsv"
    PD_DP_RANK_AFFINITY=0   # DCPX_PP: one prefill DP rank -> affinity would pin every request to decode DP rank 0
fi
# DCPX_SPF=1: prefill --schedule-policy shortest-prefill-first (llying 2p1d-aus stage-b4-spf, +15.4 % at 2P1D C176).
[[ "$DCPX_SPF" == 1 ]] && PREFILL_EXTRA_ARGS+=" --schedule-policy shortest-prefill-first"
# DCPX_NOOFFLOAD=1: drop OffloaderV2 from the prefill (PP stages hold ~94-120 GB of weights; S3 with it was -19 %).
if [[ "$DCPX_NOOFFLOAD" == 1 ]]; then
    PREFILL_EXTRA_ARGS="$(echo " $PREFILL_EXTRA_ARGS " | sed -E 's/ --offload-(group-size|num-in-group|prefetch-step|mode) [^ ]+//g; s/^ +//; s/ +$//')"
fi
# DCPX_DECODE_EC=1: decode.py all-rank empty_cache when free < 6 GiB every 8 loop iterations (other session's
# glm52.dcp4-decode-gmu.packup_20261023 patch 03) + memlog on the TP0 Decode-batch line (patch 01).
[[ "$DCPX_DECODE_EC" == 1 ]] && export DECODE_MOUNTS_ADD="$WS/harness/mounts.yihou.dcpx-ec-decode.tsv"

# --- KV rails: each prefill DPA rank i (single NIC ionic_i via the per-GPU map) now sends to ALL 4 decode ranks
# (common/conn.py:1369-1377: decode attn_tp 4 > prefill attn_tp 1 -> required_dst_info_num 4). Each decode rank
# registers its KV on all four ionic NICs, so prefill ionic_i always finds decode ionic_i (Mooncake dest affinity,
# MC_ENABLE_DEST_DEVICE_AFFINITY=1 from bench-harness/config.sh:88, present on both legs). role is set by engine.sh
# and by audit_live (config_env) before sourcing; the prefill keeps the base per-GPU map.
# DCPX_RAIL=destlocal (needs image ...-destpin: Mooncake + llying's dest-local-rail / pinned-HCA / IB-timeout patches,
# llying cross-rank-rca-20260921/fix): BOTH legs register all four rails (shared list); the sender posts each WRITE on the
# HCA the receiver pinned for the destination GPU, so every decode GPU is fed by exactly one NIC (no condition-B fan-in).
: "${DCPX_RAIL:=affinity}"
case "$DCPX_RAIL" in
    affinity) ;;
    destlocal)
        RDMA_DEVICE='ionic_0,ionic_1,ionic_2,ionic_3'
        PREFILL_EXTRA_ENV+=" MC_ENABLE_DEST_LOCAL_RAIL=1 MC_IB_TIMEOUT=${DCPX_IB_TIMEOUT:-18}"
        DECODE_EXTRA_ENV+=" MC_ENABLE_DEST_LOCAL_RAIL=1 MC_IB_TIMEOUT=${DCPX_IB_TIMEOUT:-18}" ;;
    hip) ;;   # same-node (DCPX_BASE_CONFIG = a non-xnode chain): keep the base per-GPU map + no-such-device filter
    *) echo "config.yihou.dcpx.sh: DCPX_RAIL must be affinity|destlocal|hip" >&2; return 2 2>/dev/null || exit 2 ;;
esac
if [[ "$DCPX_RAIL" == affinity && "${role:-}" == decode ]]; then
    case "$DCPX_DECODE_RDMA" in
        list) RDMA_DEVICE='ionic_0,ionic_1,ionic_2,ionic_3' ;;
        map4) RDMA_DEVICE='{"0":"ionic_0,ionic_1,ionic_2,ionic_3","1":"ionic_0,ionic_1,ionic_2,ionic_3","2":"ionic_0,ionic_1,ionic_2,ionic_3","3":"ionic_0,ionic_1,ionic_2,ionic_3"}' ;;
        *) echo "config.yihou.dcpx.sh: DCPX_DECODE_RDMA must be list|map4" >&2; return 2 2>/dev/null || exit 2 ;;
    esac
fi
[[ "$DCPX_RAIL" == hip || "$MC_TE_FILTERS" == ionic_0,ionic_1,ionic_2,ionic_3 ]] ||
    { echo "config.yihou.dcpx.sh: MC_TE_FILTERS must list all four rails (whitelist is ANDed with the device list)" >&2; return 2 2>/dev/null || exit 2; }

# Mooncake MR size cap (llying 2p1d-v0522 README:98-112: ionic EINVAL on 1 GiB-aligned 2 GiB MRs; more MRs now that the
# decode registers its pools on four NICs). Both legs. The v0522 engine.so has no MC_ENABLE_DEST_LOCAL_RAIL /
# MC_IB_TIMEOUT (strings check 10-22), so each decode GPU receives on up to four NICs (llying condition B: NIC drops
# possible) - watch "transfer failed" lines and ionic rx counters.
: "${DCPX_MC_MAX_MR:=1073741824}"
[[ "$DCPX_MC_MAX_MR" == 0 ]] || { PREFILL_EXTRA_ENV+=" MC_MAX_MR_SIZE=$DCPX_MC_MAX_MR"; DECODE_EXTRA_ENV+=" MC_MAX_MR_SIZE=$DCPX_MC_MAX_MR"; }
YIHOU_EXP="${YIHOU_EXP:-E}+dcpx${DCPX_DCP}-shard${DCPX_SHARD}-pmtp${DCPX_PMTP}-pnw${DCPX_PACKNOWAIT}${DCPX_GRAPH_BS:+-gbs${DCPX_GRAPH_BS##*,}}-pp${DCPX_PP}-spf${DCPX_SPF}-nooff${DCPX_NOOFFLOAD}-ec${DCPX_DECODE_EC}-${DCPX_RAIL}-${DCPX_DECODE_RDMA}-split${DCPX_SPLIT}"
unset _dx_base _dx_pe _i
