#!/usr/bin/env bash
# Overlay (g4tune 2026-10-19): copy of config.yihou.xferfix.sh (R036) with conn.yihou-xferfix2.py, which adds
# YIHOU_XFER_SHARD=mix (hashed room). Defaults here are LOG-ONLY: dest + orig + LOG=1 (upstream behaviour).
#   XFERFIX_BASE_CONFIG   chain to source first (default config.yihou.mc4340.sh)
#   YIHOU_NONFINAL_EVENT  orig|launch|off (default orig): when the non-final-chunk wait_event is recorded
#   YIHOU_XFER_SHARD      dest|room|mix (default dest): transfer-queue sharding (room = bootstrap_room % queues)
#   YIHOU_XFER_LOG        0|1 (default 1): YIHOU-XFER stats incl. sync/idle
#   KV_CKSUM              0|1 (default 0): built-in --disaggregation-enable-kv-checksum on BOTH legs
# Mounts: mounts.yihou.v2-xferfix2.tsv (noblock, noabsorbcfg, conn xferfix2, prefill event).
_xf_base="${XFERFIX_BASE_CONFIG:?set XFERFIX_BASE_CONFIG}"
[[ -r "$_xf_base" ]] || { echo "config.yihou.xferfix2.sh: base not readable: $_xf_base" >&2; return 2 2>/dev/null || exit 2; }
source "$_xf_base"
YIHOU_MOUNTS="$WS/harness/mounts.yihou.v2-xferfix2.tsv"
: "${YIHOU_NONFINAL_EVENT:=orig}"; : "${YIHOU_XFER_SHARD:=dest}"; : "${YIHOU_XFER_LOG:=1}"; : "${KV_CKSUM:=0}"
case "$YIHOU_NONFINAL_EVENT:$YIHOU_XFER_SHARD:$KV_CKSUM" in
  orig:*|launch:*|off:*) ;; *) echo "config.yihou.xferfix2.sh: bad YIHOU_NONFINAL_EVENT" >&2; return 2 2>/dev/null || exit 2 ;; esac
case "$YIHOU_XFER_SHARD" in dest|room|mix) ;; *) echo "config.yihou.xferfix2.sh: bad YIHOU_XFER_SHARD" >&2; return 2 2>/dev/null || exit 2 ;; esac
PREFILL_EXTRA_ENV+=" YIHOU_XFER_LOG=$YIHOU_XFER_LOG YIHOU_XFER_SHARD=$YIHOU_XFER_SHARD YIHOU_NONFINAL_EVENT=$YIHOU_NONFINAL_EVENT"
if [[ "$KV_CKSUM" == 1 ]]; then
    PREFILL_EXTRA_ARGS+=" --disaggregation-enable-kv-checksum"
    DECODE_EXTRA_ARGS+=" --disaggregation-enable-kv-checksum"
fi
_xf_ck=""; [[ "$KV_CKSUM" == 1 ]] && _xf_ck="-cksum"
YIHOU_EXP="${YIHOU_EXP:-E}+xferfix2-$YIHOU_NONFINAL_EVENT-$YIHOU_XFER_SHARD$_xf_ck"
unset _xf_base _xf_ck
