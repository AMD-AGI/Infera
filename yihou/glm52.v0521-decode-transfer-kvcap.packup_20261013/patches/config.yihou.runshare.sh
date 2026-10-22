#!/usr/bin/env bash
# Overlay (task v0521cap Track B): decode-only mount of decode.yihou-runshare.py + YIHOU_DECODE_RUNNING_KV_SHARE
# (default 0.5 = running/arrived requests always keep >= 50 % of each decode DP rank's KV pool; in-flight KV of
# prealloc->transfer requests capped at 50 %). Chain: RUNSHARE_BASE_CONFIG (default config.yihou.mc4340.sh).
_rs_base="${RUNSHARE_BASE_CONFIG:-$(dirname "${BASH_SOURCE[0]}")/config.yihou.mc4340.sh}"
[[ -r "$_rs_base" ]] || { echo "config.yihou.runshare.sh: base not readable: $_rs_base" >&2; return 2 2>/dev/null || exit 2; }
source "$_rs_base"
DECODE_MOUNTS_ADD="$WS/harness/mounts.yihou.decode-runshare.tsv"
DECODE_EXTRA_ENV+=" YIHOU_DECODE_RUNNING_KV_SHARE=${RUNNING_KV_SHARE:-0.5}"
YIHOU_EXP="${YIHOU_EXP:-E}+runshare${RUNNING_KV_SHARE:-0.5}"
unset _rs_base
