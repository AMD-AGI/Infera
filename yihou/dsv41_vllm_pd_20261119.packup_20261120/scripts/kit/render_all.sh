#!/usr/bin/env bash
# Dry-run render of every configs/*.env with a fake GPU list and NODE_IP placeholder -> configs/renders/<id>.txt (no side effects).
# PD configs (PD_CONFIG=1) render through serve_pd.sh (P GPUs 0,1, D GPUs 2,3, same node); role files -> configs/renders/pd_roles/.
set -euo pipefail
KIT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WS=$(cd "$KIT_DIR/../.." && pwd)
mkdir -p "$WS/configs/renders"
for f in "$WS"/configs/*.env; do
  id=$(basename "$f" .env)
  pd=$(bash -c 'source "$1"; echo "${PD_CONFIG:-0}"' _ "$f")
  if [ "$pd" = 1 ]; then
    DSV41_NODE_IP=NODE_IP bash "$KIT_DIR/serve_pd.sh" "$id" 0,1 2,3 "render-$id" --dry-run > "$WS/configs/renders/$id.txt"
    echo "rendered PD $id p=0,1 d=2,3 -> configs/renders/$id.txt (+ configs/renders/pd_roles/render-$id/)"
    continue
  fi
  n=$(bash -c 'source "$1"; echo "$GPUS_NEEDED"' _ "$f")
  gl=$(seq -s, 0 $((n - 1)))
  DSV41_NODE_IP=NODE_IP bash "$KIT_DIR/serve.sh" "$id" "$gl" "render-$id" --dry-run > "$WS/configs/renders/$id.txt"
  echo "rendered $id gpus=$gl -> configs/renders/$id.txt"
done
DSV41_NODE_IP=NODE_IP bash "$KIT_DIR/serve.sh" r2-dp2-umbp-c384 4,5 render-r2-off10 --dry-run --port-offset 10 --aiter-jit dir \
  > "$WS/configs/renders/r2-dp2-umbp-c384.offset10-aiterdir.txt"
echo "rendered r2-dp2-umbp-c384 example with --port-offset 10 --aiter-jit dir (gpus 4,5)"
