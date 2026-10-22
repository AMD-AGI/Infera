#!/usr/bin/env bash
# Overlay (task v0521cap2, user 2026-10-15 "decode侧cpu解码都是用4个worker"): decode leg runs
# --tokenizer-worker-num ${DECODE_TOKENIZER_WORKERS:-4} --detokenizer-worker-num ${DECODE_DETOKENIZER_WORKERS:-4}.
# v0.5.21: detokenizer_worker_num is forced back to 1 only with --skip-tokenizer-init (not used here; serving_hook.py:723).
# Prefill leg unchanged. Chain: TOKW2_BASE_CONFIG (default config.yihou.mc4340n6.sh). Applies to all rounds from 2026-10-15.
_tw2_base="${TOKW2_BASE_CONFIG:-$(dirname "${BASH_SOURCE[0]}")/config.yihou.mc4340n6.sh}"
[[ -r "$_tw2_base" ]] || { echo "config.yihou.tokw2.sh: base not readable: $_tw2_base" >&2; return 2 2>/dev/null || exit 2; }
source "$_tw2_base"
[[ " $DECODE_EXTRA_ARGS " == *" --tokenizer-worker-num "* ]] || DECODE_EXTRA_ARGS+=" --tokenizer-worker-num ${DECODE_TOKENIZER_WORKERS:-4}"
[[ " $DECODE_EXTRA_ARGS " == *" --detokenizer-worker-num "* ]] || DECODE_EXTRA_ARGS+=" --detokenizer-worker-num ${DECODE_DETOKENIZER_WORKERS:-4}"
YIHOU_EXP="${YIHOU_EXP:-E}+tokw${DECODE_TOKENIZER_WORKERS:-4}detok${DECODE_DETOKENIZER_WORKERS:-4}"
unset _tw2_base
