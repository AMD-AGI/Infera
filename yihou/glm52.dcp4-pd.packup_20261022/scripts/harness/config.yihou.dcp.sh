#!/usr/bin/env bash
# dcp_pd_20261021 overlay (pdrun): GLM-5.2 same-node 1P1D P4D4 on infera-sglang:v0522-1020-yihou-dcp-mc4340
# (v0521-mc4340 recipe on lmsysorg/sglang-rocm:v0.5.22-rocm720-mi35x-20261020), optionally with a sglang
# python overlay (dir mount over /sgl-workspace/sglang/python/sglang) and a DCP decode leg.
#
# Chain: this -> config.yihou.E2-target.sh -> config.yihou.v0521-base.sh -> bench-harness/config.sh
# Knobs (all ":=", so YIHOU_OVERRIDES / env win):
#   SGLANG_OVERLAY   overlay python/sglang dir built by tools/build_overlay.yihou.sh (recipe patches +
#                    R001 noblock + R003 noabsorbcfg re-applied). EMPTY = the image's own sglang, which
#                    then gets the two v0522-cut runtime-fix files (mounts.yihou.dcp-image.tsv).
#   DCP_DECODE       1 = baseline decode (TP4 DP4 DPA EP1, as e2g); N>1 = decode TP4 EP4 DP1 no-DPA
#                    + --dcp-size N (DCP needs attn_tp_size % dcp_size == 0, so no DPA) and router
#                    PD_DP_RANK_AFFINITY=0 (decode has one DP rank; affinity would 503 ranks 1-3).
#   PREFILL_ARM      dpa (TP4 DP4 DPA, e2g) | tp (TP4 DP1 no DPA; router affinity off).
#   Real-draft (correctness) runs: DECODE_SIMULATE_ACC_LEN= (empty) in YIHOU_OVERRIDES.
#   PREFILL_HICACHE_SIZE  small for debugging (user): default 48 GB/rank (e2g: 450).
# Container prefix yihou-dcp-pd-<node>-<round> (team rule: yihou-dcp-*).

_dcp_node_key="${YIHOU_NODE##*-}"
: "${CONTAINER_PREFIX:=yihou-dcp-pd-${_dcp_node_key}-${YIHOU_ROUND:-x}}"
: "${PREFILL_HICACHE_SIZE:=48}"
: "${CONC:=4}"
: "${DCP_DECODE:=1}"
: "${SGLANG_OVERLAY:=}"
[[ "$DCP_DECODE" =~ ^[1-9][0-9]*$ ]] || { echo "DCP_DECODE must be a positive integer" >&2; return 2 2>/dev/null || exit 2; }
# PREFILL_ARM: dpa (default; e2g leg TP4 DP4 DPA) | tp (TP4, no DPA; design section 7 main E2E arm).
: "${PREFILL_ARM:=dpa}"
case "$PREFILL_ARM" in
    dpa) ;;
    tp) : "${PREFILL_DP:=1}"; : "${PREFILL_DPA:=0}"; : "${PD_DP_RANK_AFFINITY:=0}" ;;
    *) echo "PREFILL_ARM must be dpa|tp" >&2; return 2 2>/dev/null || exit 2 ;;
esac
if (( DCP_DECODE > 1 )); then
    : "${DECODE_TP:=4}"; : "${DECODE_EP:=4}"; : "${DECODE_DP:=1}"; : "${DECODE_DPA:=0}"
    : "${PD_DP_RANK_AFFINITY:=0}"
fi

source "$(dirname "${BASH_SOURCE[0]}")/config.yihou.E2-target.sh"

# HARD (after the chain): image, mounts, DCP flag.
IMAGE=infera-sglang:v0522-1020-yihou-dcp-mc4340
# Per-node image IDs (each node builds its own copy: tools/build_image_on_node.yihou.sh appends a row).
_dcp_ids="$WS/harness/image_ids.dcp.tsv"   # node<TAB>image<TAB>id
EXPECTED_IMAGE_ID="$(awk -F"	" -v n="${YIHOU_NODE:-}" -v i="$IMAGE" "\$1==n && \$2==i {id=\$3} END{print id}" "$_dcp_ids" 2>/dev/null)"
# Unknown node: warn only (dryrun still renders); run.v2 up/retry-decode refuse an ID without sha256:.
[[ "$EXPECTED_IMAGE_ID" == sha256:* ]] ||
    echo "config.yihou.dcp.sh: WARNING no $IMAGE ID for YIHOU_NODE='${YIHOU_NODE:-}' in $_dcp_ids (run tools/build_image_on_node.yihou.sh on the node)" >&2
unset _dcp_ids
export IMAGE EXPECTED_IMAGE_ID
if [[ -n "$SGLANG_OVERLAY" ]]; then
    [[ -f "$SGLANG_OVERLAY/__init__.py" && "$SGLANG_OVERLAY" == */python/sglang ]] ||
        { echo "SGLANG_OVERLAY must be an overlay .../python/sglang dir: $SGLANG_OVERLAY" >&2; return 2 2>/dev/null || exit 2; }
    _dcp_tag="$(basename "$(dirname "$(dirname "$SGLANG_OVERLAY")")")"
    YIHOU_MOUNTS="$WS/harness/mounts.yihou.dcp-overlay.$_dcp_tag.tsv"
    _dcp_row="$(printf '%s\t%s' "$SGLANG_OVERLAY" /sgl-workspace/sglang/python/sglang)"
    if [[ "$(cat "$YIHOU_MOUNTS" 2>/dev/null)" != "$_dcp_row" ]]; then
        printf '%s\n' "$_dcp_row" > "$YIHOU_MOUNTS.tmp.$$" && mv -f "$YIHOU_MOUNTS.tmp.$$" "$YIHOU_MOUNTS"
    fi
    unset _dcp_tag _dcp_row
else
    YIHOU_MOUNTS="$WS/harness/mounts.yihou.dcp-image.tsv"
fi
if (( DCP_DECODE > 1 )); then
    DECODE_EXTRA_ARGS="$DECODE_EXTRA_ARGS --dcp-size $DCP_DECODE"
fi
# Extra container env as COMMA-separated KEY=VALUE lists (YIHOU_OVERRIDES is space-split, so no spaces):
#   BOTH_ENV_LIST   both legs (e.g. HSA_ENABLE_SDMA=0 on v3 nodes, amdgpu 6.16.13 weight-load hang)
#   DECODE_ENV_LIST decode only (e.g. SGLANG_DSA_DCP_KEEP_KV_B_BF16=1,SGLANG_DSA_DCP_SPLIT_DECODE_INDEXER=1)
: "${BOTH_ENV_LIST:=}"; : "${DECODE_ENV_LIST:=}"; : "${PREFILL_ENV_LIST:=}"
for _l in BOTH_ENV_LIST DECODE_ENV_LIST PREFILL_ENV_LIST; do
    [[ -z "${!_l}" || "${!_l}" =~ ^[A-Za-z_][A-Za-z0-9_]*=[^,[:space:]]*(,[A-Za-z_][A-Za-z0-9_]*=[^,[:space:]]*)*$ ]] ||
        { echo "$_l must be KEY=VAL[,KEY=VAL...]" >&2; return 2 2>/dev/null || exit 2; }
done; unset _l
[[ -n "$BOTH_ENV_LIST" ]] && { PREFILL_EXTRA_ENV+=" ${BOTH_ENV_LIST//,/ }"; DECODE_EXTRA_ENV+=" ${BOTH_ENV_LIST//,/ }"; }
[[ -n "$PREFILL_ENV_LIST" ]] && PREFILL_EXTRA_ENV+=" ${PREFILL_ENV_LIST//,/ }"
[[ -n "$DECODE_ENV_LIST" ]] && DECODE_EXTRA_ENV+=" ${DECODE_ENV_LIST//,/ }"
# DECODE_CUSTOM_AR=1: drop v0521-base's --disable-custom-all-reduce from the decode leg (r3; mixrun's mixed DCP4
# runs used custom AR ON). Default 0 = as e2g.
: "${DECODE_CUSTOM_AR:=0}"
if [[ "$DECODE_CUSTOM_AR" == 1 ]]; then
    DECODE_EXTRA_ARGS="${DECODE_EXTRA_ARGS//--disable-custom-all-reduce/}"
    DECODE_EXTRA_ARGS="$(echo $DECODE_EXTRA_ARGS)"
fi
YIHOU_EXP="p${PREFILL_ARM}-dcp${DCP_DECODE}${SGLANG_OVERLAY:+-overlay}"
unset _dcp_node_key
