#!/usr/bin/env bash
# rm_old (task cleanup 2026-10-23, user-approved): docker rm OUR exited containers on THIS node only.
# Guards: name prefix glm52-v0521-yihou-* or yihou-packtest-*; State must be exited/created (never running);
# never the round(s) given in KEEP; engine containers (prefill/decode) only when the round's server logs exist on NFS
# (non-empty). No -v: named volumes (caches) are kept. Prints every decision. Usage: rm_old.yihou.sh [dry] (KEEP env)
set -uo pipefail
DRY=${1:-}; R=/shared_nfs/yihou_glm52/v0521dbg_20261010/runs; KEEP=" ${KEEP:-ppda3} "
docker ps -a --format '{{.Names}} {{.State}}' | while read -r name state; do
  case "$name" in glm52-v0521-yihou-*|yihou-packtest-*) ;; *) echo "SKIP not-ours $name"; continue ;; esac
  [[ "$state" == exited || "$state" == created ]] || { echo "SKIP $state $name"; continue; }
  round=$(sed -nE 's/^glm52-v0521-yihou-[0-9a-z]+-([^-]+)-.*/\1/p' <<<"$name")
  [[ -n "$round" && "$KEEP" == *" $round "* ]] && { echo "SKIP keep $name"; continue; }
  if [[ "$name" == *-prefill-* || "$name" == *-decode-* ]]; then
    logs=$(find "$R/$round/server-logs" -maxdepth 1 -name '*.log' -size +0 2>/dev/null | wc -l)
    [[ "$logs" -ge 1 ]] || { echo "SKIP no-nfs-logs($round) $name"; continue; }
  fi
  if [[ "$DRY" == dry ]]; then echo "WOULD rm $name"; else docker rm "$name" >/dev/null && echo "RM $name" || echo "FAIL rm $name"; fi
done
