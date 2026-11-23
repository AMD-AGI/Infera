#!/bin/bash
# Host or container side: build one campaign dir under $FORGE_ROOT/runs/<op>: a copy of the workset (ws/) and an isolated
# git workspace (engine/) holding the editable baseline module at the edit_target path (aiter/ops/mega_mhc.py), seeded
# from the Definition's `baseline` (aiter's module verbatim + run delegation). usage: prep_campaign.sh OP
set -euo pipefail
OP=${1:?op}; R=${FORGE_ROOT:?}; C=$R/runs/$OP
mkdir -p "$C"
[ -d "$C/ws" ] || cp -r "$R/ws" "$C/ws"
if [ ! -d "$C/engine/.git" ]; then
  mkdir -p "$C/engine/aiter/ops"
  python3 - "$C/ws/definitions/mhc/$OP.json" "$C/engine/aiter/ops/mega_mhc.py" <<'PY'
import json, sys
open(sys.argv[2], "w").write(json.load(open(sys.argv[1]))["baseline"])
PY
  git -C "$C/engine" init -q && git -C "$C/engine" add -A && \
    git -C "$C/engine" -c user.name="${GIT_AUTHOR_NAME:-yihou}" -c user.email="${GIT_AUTHOR_EMAIL:-yihou@amd.com}" commit -q -m "baseline: aiter mega_mhc (image 9c1d45c2) + workset run()"
fi
echo "$C"
