#!/usr/bin/env bash
# Unit tests of the three new fork PRs (llying-001/vllm#15 #16 #17) inside the C1 image on one GPU of this node.
# Test files come from a checkout of base/nightly-de7cf7f8-C0 + the three diffs (/perf_apps/.../upstream/vllm-C1-tests);
# the installed (patched) vllm package is what they import (--import-mode=importlib keeps the source tree off sys.path).
#   test_C1.sh [GPU]
set -uo pipefail
. /perf_apps/liyingli/bench_agentx/vllm-p4d4-aus-20261127/site/env.sh
gpu=${1:-0} T=/perf_apps/liyingli/bench_agentx/upstream/vllm-C1-tests IMG=infera-vllm:nightly-de7cf7f8-C1
out=$W/results/test-C1-$(hostname -s)
mkdir -p "$out"
docker run --rm --name "vpd-test-c1-$gpu" --network host --device /dev/kfd --device /dev/dri --group-add video --group-add render \
    --security-opt seccomp=unconfined -e HIP_VISIBLE_DEVICES="$gpu" -e VLLM_ROCM_USE_AITER=1 -v "$T:$T:ro" -w /tmp \
    --entrypoint bash "$IMG" -c "pip install -q pytest > /dev/null 2>&1; cd /tmp && python3 -m pytest -v --import-mode=importlib -p no:cacheprovider \
      $T/tests/kernels/moe/test_naive_dp_ep_mxfp4_dispatch.py $T/tests/kernels/moe/test_naive_dp_ep_mxfp4_dispatch_pack.py \
      $T/tests/kernels/attention/test_rocm_triton_indexer_topk.py 2>&1 | tail -60" > "$out/new-tests.log" 2>&1
echo "$(ts) new tests: $(grep -a -E '^(=+ .*(passed|failed|error).* =+)$' "$out/new-tests.log" | tail -1)"
docker run --rm --name "vpd-test-c1-$gpu" --network host --device /dev/kfd --device /dev/dri --group-add video --group-add render \
    --security-opt seccomp=unconfined -e HIP_VISIBLE_DEVICES="$gpu" -e VLLM_ROCM_USE_AITER=1 -v "$T:$T:ro" -w /tmp \
    --entrypoint bash "$IMG" -c "pip install -q pytest > /dev/null 2>&1; cd /tmp && python3 -m pytest -v --import-mode=importlib -p no:cacheprovider \
      $T/tests/kernels/moe/test_naive_dp_ep_dbo.py $T/tests/kernels/attention/test_rocm_paged_mxfp4_indexer.py \
      $T/tests/kernels/attention/test_rocm_aiter_mla_sparse_metadata_sync.py 2>&1 | tail -60" > "$out/regression.log" 2>&1
echo "$(ts) regression: $(grep -a -E '^(=+ .*(passed|failed|error).* =+)$' "$out/regression.log" | tail -1)"
