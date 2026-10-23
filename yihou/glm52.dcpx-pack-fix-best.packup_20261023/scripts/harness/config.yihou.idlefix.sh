#!/usr/bin/env bash
# Overlay (idxshare fix, 2026-10-20): the user's g4new rule (PR42553 + index_share true + decode custom AR on) plus
# ONLY the root-cause fix eager_runner.yihou-idlefix.py (EagerRunner._execute_idle also initializes the active
# forward-context attention backend for padded IDLE batches). No debug agent, no guards, no seed guard.
#   IDLEFIX_BASE_CONFIG  chain to source (default config.yihou.g4new.sh)
_if_base="${IDLEFIX_BASE_CONFIG:-$(dirname "${BASH_SOURCE[0]}")/config.yihou.g4new.sh}"
[[ -r "$_if_base" ]] || { echo "config.yihou.idlefix.sh: base not readable: $_if_base" >&2; return 2 2>/dev/null || exit 2; }
source "$_if_base"
case "$YIHOU_MOUNTS" in
    "$WS/harness/mounts.yihou.v2-xferfix2-gtt.tsv") YIHOU_MOUNTS="$WS/harness/mounts.yihou.v2-xferfix2-gtt-idlefix.tsv" ;;
    *) echo "config.yihou.idlefix.sh: unexpected YIHOU_MOUNTS=$YIHOU_MOUNTS" >&2; return 2 2>/dev/null || exit 2 ;;
esac
YIHOU_EXP="${YIHOU_EXP:-E}+idlefix"
unset _if_base
