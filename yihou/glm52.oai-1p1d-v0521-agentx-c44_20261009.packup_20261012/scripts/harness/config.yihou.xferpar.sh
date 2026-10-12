#!/usr/bin/env bash
# Overlay (R033): KV-transfer instrumentation + k-way layer-group parallel submission on the PREFILL sender.
#   XFERPAR_BASE_CONFIG  chain to source first (default config.yihou.mc4340.sh)
#   YIHOU_XFER_LOG       0|1 (default 1)  -> PREFILL env (YIHOU-XFER lines, 1/min/shard)
#   YIHOU_XFER_PAR       k (default 4; 1 = original single batch_transfer_sync) -> PREFILL env
#   XFER_THREAD_POOL     SGLANG_DISAGGREGATION_THREAD_POOL_SIZE (default 16 -> 4 executor threads per queue)
# Mount: conn.yihou-regbase-xferlog-par.py (REGBASE + log + par; default-off = image behaviour), repro, unit-tested.
_xp_base="${XFERPAR_BASE_CONFIG:-$(dirname "${BASH_SOURCE[0]}")/config.yihou.mc4340.sh}"
[[ -r "$_xp_base" ]] || { echo "config.yihou.xferpar.sh: base not readable: $_xp_base" >&2; return 2 2>/dev/null || exit 2; }
source "$_xp_base"
YIHOU_MOUNTS="$WS/harness/mounts.yihou.v2-xferpar.tsv"
: "${YIHOU_XFER_LOG:=1}"; : "${YIHOU_XFER_PAR:=4}"; : "${XFER_THREAD_POOL:=16}"
[[ "$YIHOU_XFER_PAR" =~ ^[1-9][0-9]*$ && "$XFER_THREAD_POOL" =~ ^[1-9][0-9]*$ ]] ||
    { echo "config.yihou.xferpar.sh: bad YIHOU_XFER_PAR/XFER_THREAD_POOL" >&2; return 2 2>/dev/null || exit 2; }
PREFILL_EXTRA_ENV+=" YIHOU_XFER_LOG=$YIHOU_XFER_LOG YIHOU_XFER_PAR=$YIHOU_XFER_PAR SGLANG_DISAGGREGATION_THREAD_POOL_SIZE=$XFER_THREAD_POOL"
YIHOU_EXP="${YIHOU_EXP:-E}+xferpar$YIHOU_XFER_PAR"
unset _xp_base
