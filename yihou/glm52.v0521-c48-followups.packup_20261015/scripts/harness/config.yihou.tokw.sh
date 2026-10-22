#!/usr/bin/env bash
# Overlay (task v0521cap2, user 2026-10-14 ~08:00): decode leg with more tokenizer workers: --tokenizer-worker-num
# ${DECODE_TOKENIZER_WORKERS:-4} (sglang v0.5.21 multi-tokenizer HTTP front-end; rejected only with --grpc-port, Elastic EP
# scale-up, and it leaves detokenizer_worker_num at its default 1). Prefill leg unchanged. Chain: TOKW_BASE_CONFIG.
_tw_base="${TOKW_BASE_CONFIG:-$(dirname "${BASH_SOURCE[0]}")/config.yihou.mc4340n4.sh}"
[[ -r "$_tw_base" ]] || { echo "config.yihou.tokw.sh: base not readable: $_tw_base" >&2; return 2 2>/dev/null || exit 2; }
source "$_tw_base"
_tw_flag="--tokenizer-worker-num ${DECODE_TOKENIZER_WORKERS:-4}"
[[ " $DECODE_EXTRA_ARGS " == *" --tokenizer-worker-num "* ]] || DECODE_EXTRA_ARGS+=" $_tw_flag"
YIHOU_EXP="${YIHOU_EXP:-E}+tokw${DECODE_TOKENIZER_WORKERS:-4}"
unset _tw_base _tw_flag
