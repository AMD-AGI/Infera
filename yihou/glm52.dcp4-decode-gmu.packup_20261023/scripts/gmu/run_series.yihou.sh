#!/usr/bin/env bash
# Run the decode-only gmu experiments as two parallel queues (slot a = GPUs 0-3, slot b = GPUs 4-7).
# Usage on the node: NK=<nk> bash run_series.yihou.sh <series-file>   (detach with setsid nohup)
# Series file lines: <slot> <tag> <mem> | <extra sglang args> | <extra env K=V ...>   ('#' comments allowed)
set -uo pipefail
: "${NK:?}"; SER=${1:?series file}
T=/shared_nfs/yihou_glm52/dcp_pd_20261021/gmu/tools
run_queue() {
  local slot=$1
  grep -vE '^\s*(#|$)' "$SER" | while IFS='|' read -r head extra envx; do
    read -r s tag mem <<<"$head"
    [[ "$s" == "$slot" ]] || continue
    NK=$NK SLOT=$s TAG="$tag" MEM="$mem" EXTRA="$(echo $extra)" ENVX="$(echo $envx)" bash $T/decode_only.yihou.sh
  done
}
run_queue a & run_queue b & wait
echo "$(date -u +%FT%TZ) series $SER done"
