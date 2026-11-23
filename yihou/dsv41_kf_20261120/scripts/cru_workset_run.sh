#!/bin/bash
# Run ON a crusoe node (spur exec): correctness + performance of the mHC workset for both operators in parallel
# (bf16 on GPU0, mxfp8 on GPU1), ephemeral containers of the serving image. Output: $D/<tag>/{c,p}-<op>.json + logs.
set -uo pipefail
TAG=${1:?tag}
D=/shared_nfs/yihou/dsv41_kf_20261120/workset_runs
IMG=vllm/vllm-openai-rocm@sha256:694108707f333787b1041caa0e948dd18b86f2450365a94e38819f026275ee32
SRC=/shared_nfs/yihou/dsv41_kf_20261120/forge/ws
O=$D/$TAG; mkdir -p "$O"; chmod 1777 "$O"
docker image inspect "$IMG" >/dev/null 2>&1 || docker pull "$IMG" > "$O/pull.log" 2>&1

g=2
for op in mhc_seam_bf16_decode mhc_seam_bf16_prefill mhc_seam_mxfp8_decode mhc_seam_mxfp8_prefill; do
  mkdir -p "$O/ws-$op" && cp -r "$SRC/." "$O/ws-$op/" && chmod -R a+rwX "$O/ws-$op"
  docker run -d --rm --name "yihou-dsv41kf-ws-$op" --network host --device /dev/kfd --device /dev/dri --group-add video \
    -e HIP_VISIBLE_DEVICES=$g -e PYTHONDONTWRITEBYTECODE=1 -e E2E_NODE="$(hostname)" -v "$O:$O" -w "$O/ws-$op" \
    --entrypoint bash "$IMG" -c "./run_correctness.sh --operator $op --json $O/c-$op.json > $O/c-$op.log 2>&1; \
      ./run_performance.sh --operator $op --json $O/p-$op.json > $O/p-$op.log 2>&1; echo done > $O/done-$op" >/dev/null
  g=$((g + 1))
done
echo "started $TAG"
