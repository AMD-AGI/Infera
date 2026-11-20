#!/bin/bash
# Stage kit + InferenceX e2e onto /perf_apps (docker on the nodes cannot read HOME NFS). Run on a16-19 before node runs.
set -euo pipefail
KIT_DIR=$(cd "$(dirname "$0")" && pwd); WS=$(cd "$KIT_DIR/../.." && pwd)
ST=/perf_apps/yihou/dsv41_vllm_pd_20261119/stage
mkdir -p "$ST"
rsync -a --delete "$KIT_DIR/" "$ST/kit/"
rsync -a --delete --exclude .git "$WS/source/InferenceX/inferencex-e2e/" "$ST/inferencex-e2e/"
chmod -R go-w "$ST"
echo "staged to $ST"
