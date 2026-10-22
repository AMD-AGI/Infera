#!/usr/bin/env bash
# E2 (mixrun, 2026-10-22): PD with a DP2 x DCP2 decode on top of config.yihou.dcp.sh (pdrun, unchanged).
# Decode: TP4 EP4, --dp-size 2 --enable-dp-attention --dcp-size 2 (attn_tp 2 % dcp 2 == 0), router affinity off
# (decode has 2 DP ranks, prefill 4). Custom AR ON on the decode. v3 burst nodes: HSA_ENABLE_SDMA=0 on BOTH legs
# (amdgpu 6.16.13 hangs in SDMA H2D during weight load; mixrun ledger R4-R8). Decode envs: kv_b bf16 (q-rep on 78
# layers), split decode indexer with the triton pack/merge. All ":=" so YIHOU_OVERRIDES win.
: "${DECODE_TP:=4}"; : "${DECODE_EP:=4}"; : "${DECODE_DP:=2}"; : "${DECODE_DPA:=1}"
: "${PD_DP_RANK_AFFINITY:=0}"
: "${DCP_DECODE:=2}"
: "${DECODE_CUSTOM_AR:=1}"
: "${E2_SDMA_OFF:=1}"
: "${E2_KVB:=1}"      # 1 = SGLANG_DSA_DCP_KEEP_KV_B_BF16=1 on the decode
: "${E2_SPLIT:=1}"    # 1 = split decode indexer + triton pack/merge on the decode
source "$(dirname "${BASH_SOURCE[0]}")/config.yihou.dcp.sh"
if [[ "$E2_SDMA_OFF" == 1 ]]; then
    PREFILL_EXTRA_ENV+=" HSA_ENABLE_SDMA=0"
    DECODE_EXTRA_ENV+=" HSA_ENABLE_SDMA=0"
fi
[[ "$E2_KVB" == 1 ]] && DECODE_EXTRA_ENV+=" SGLANG_DSA_DCP_KEEP_KV_B_BF16=1"
[[ "$E2_SPLIT" == 1 ]] && DECODE_EXTRA_ENV+=" SGLANG_DSA_DCP_SPLIT_DECODE_INDEXER=1 SGLANG_DSA_DCP_SPLIT_INDEXER_TRITON=1"
YIHOU_EXP="${YIHOU_EXP}-dp${DECODE_DP}dpa${DECODE_DPA}-e2-kvb${E2_KVB}-split${E2_SPLIT}"
