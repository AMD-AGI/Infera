#!/usr/bin/env bash
# Overlay (g4tune 2026-10-21, transfer-time instrumentation): base (default config.yihou.idlefix.sh = the 10-21
# baseline) plus ONLY the LD_PRELOAD shim libhipxfer.yihou.so in the PREFILL container. The shim times every
# Mooncake HipTransport batch (hipMemcpyAsync kind Default >= 64 KiB, hipEventSynchronize): host enqueue /
# sync time and GPU copy time from its own events; one line per batch in $NODE_RD/hipxfer.<pid>.log.
# Source: mounts-v0521/src/hipxfer_trace.yihou.c. Log-only; no change to what is transferred.
#   XFERTRACE_BASE_CONFIG  chain to source (default config.yihou.idlefix.sh)
_xt_base="${XFERTRACE_BASE_CONFIG:-$(dirname "${BASH_SOURCE[0]}")/config.yihou.idlefix.sh}"
[[ -r "$_xt_base" ]] || { echo "config.yihou.xfertrace.sh: base not readable: $_xt_base" >&2; return 2 2>/dev/null || exit 2; }
source "$_xt_base"
case "$PREFILL_MOUNTS_ADD" in
    "$WS/harness/mounts.yihou.offload-attrs.tsv") PREFILL_MOUNTS_ADD="$WS/harness/mounts.yihou.offload-attrs-hipxfer.tsv" ;;
    *) echo "config.yihou.xfertrace.sh: unexpected PREFILL_MOUNTS_ADD=$PREFILL_MOUNTS_ADD" >&2; return 2 2>/dev/null || exit 2 ;;
esac
export PREFILL_MOUNTS_ADD
PREFILL_EXTRA_ENV+=" LD_PRELOAD=/opt/yihou/libhipxfer.yihou.so HIPXFER_DIR=$NODE_RD HIPXFER_MIN_BYTES=${HIPXFER_MIN_BYTES:-65536}"
YIHOU_EXP="${YIHOU_EXP:-E}+hipxfer"
unset _xt_base
