#!/usr/bin/env bash
# Overlay (R036): fix the KV-transfer head-of-line wait on the PREFILL sender.
#   XFERFIX_BASE_CONFIG   chain to source first (default config.yihou.mc4340.sh)
#   YIHOU_NONFINAL_EVENT  orig|launch|off (default launch): when the non-final-chunk wait_event is recorded
#   YIHOU_XFER_SHARD      dest|room (default room): transfer-queue sharding (room = bootstrap_room % queues)
#   YIHOU_XFER_LOG        0|1 (default 1): YIHOU-XFER stats incl. sync/idle
#   KV_CKSUM              0|1 (default 0): built-in --disaggregation-enable-kv-checksum on BOTH legs
# Mounts (repro, unit-tested): mounts.yihou.v2-xferfix.tsv (noblock, noabsorbcfg, conn xferfix, prefill event).
_xf_base="${XFERFIX_BASE_CONFIG:-$(dirname "${BASH_SOURCE[0]}")/config.yihou.mc4340.sh}"
[[ -r "$_xf_base" ]] || { echo "config.yihou.xferfix.sh: base not readable: $_xf_base" >&2; return 2 2>/dev/null || exit 2; }
source "$_xf_base"
YIHOU_MOUNTS="$WS/harness/mounts.yihou.v2-xferfix.tsv"
: "${YIHOU_NONFINAL_EVENT:=launch}"; : "${YIHOU_XFER_SHARD:=room}"; : "${YIHOU_XFER_LOG:=1}"; : "${KV_CKSUM:=0}"
case "$YIHOU_NONFINAL_EVENT:$YIHOU_XFER_SHARD:$KV_CKSUM" in
  orig:*|launch:*|off:*) ;; *) echo "config.yihou.xferfix.sh: bad YIHOU_NONFINAL_EVENT" >&2; return 2 2>/dev/null || exit 2 ;; esac
case "$YIHOU_XFER_SHARD" in dest|room) ;; *) echo "config.yihou.xferfix.sh: bad YIHOU_XFER_SHARD" >&2; return 2 2>/dev/null || exit 2 ;; esac
PREFILL_EXTRA_ENV+=" YIHOU_XFER_LOG=$YIHOU_XFER_LOG YIHOU_XFER_SHARD=$YIHOU_XFER_SHARD YIHOU_NONFINAL_EVENT=$YIHOU_NONFINAL_EVENT"
if [[ "$KV_CKSUM" == 1 ]]; then
    PREFILL_EXTRA_ARGS+=" --disaggregation-enable-kv-checksum"
    DECODE_EXTRA_ARGS+=" --disaggregation-enable-kv-checksum"
fi
_xf_ck=""; [[ "$KV_CKSUM" == 1 ]] && _xf_ck="-cksum"
YIHOU_EXP="${YIHOU_EXP:-E}+xferfix-$YIHOU_NONFINAL_EVENT-$YIHOU_XFER_SHARD$_xf_ck"
unset _xf_base _xf_ck
