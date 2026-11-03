#!/usr/bin/env bash
# Unit checks (scripts/test_fix.py) inside $IMAGE on this node, the six fix files mounted as in a deployment. One GPU is
# made visible because some sglang modules probe the device on import. IMAGE / MODEL come from config/site.local.env.
# Usage: run_unit.sh [GPU]
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG="${PKG:-$(cd "$here/.." && pwd)}"
[[ -n "${IMAGE:-}" && -n "${MODEL:-}" ]] || { set -a; . "$PKG/config/site.local.env"; set +a; }
SB=/sgl-workspace/sglang/python/sglang/srt
m=()
for f in layers/attention/graph_variants.py model_executor/runner/decode_cuda_graph_runner.py \
    managers/scheduler_components/dp_attn.py managers/schedule_batch.py model_executor/forward_batch_info.py \
    batch_overlap/two_batch_overlap.py; do
    m+=(-v "$PKG/patches/fix/$f:$SB/$f:ro")
done
docker run --rm --network none --device /dev/kfd --device /dev/dri --group-add video --group-add render \
    -e HIP_VISIBLE_DEVICES="${1:-0}" -v "$MODEL:$MODEL:ro" -v "$here/test_fix.py:/t/test_fix.py:ro" "${m[@]}" \
    --entrypoint python3 "$IMAGE" /t/test_fix.py "$MODEL"
