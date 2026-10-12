#!/usr/bin/env bash
# E2b: E2 target with the gfx950 fused DSA indexer OFF (exactly E1's H3 knobs), everything
# else E2 (450 GB + CHUNK 512, decode mem 0.95, CONC 44, index_share false, legacy IPC both
# legs, debug agent). Isolates H3 against E2. Run on crsuse2-m2m-102.

# H3 (gfx950 fused DSA indexer): OFF on both legs, both switches
#   (--no-enable-dsa-fused-indexer AND SGLANG_DISABLE_AITER_FUSED_FP8_DSA_INDEXER=1).
: "${DSA_FUSED_INDEXER:=off}"
: "${AITER_FUSED_FP8_INDEXER_WRITER:=off}"

source "$(dirname "${BASH_SOURCE[0]}")/config.yihou.E2-target.sh"
YIHOU_EXP=E2b
